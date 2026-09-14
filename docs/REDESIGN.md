# pimm-data redesign — flat-prefixed parts, roles, one transform list

> **Boundary note (2026-09-13).** Where this document places the LArTPC forward
> model — `AddNoise`, `Digitize`, the noise spectrum, `add_intrinsic_noise` — in
> pimm-data, that is no longer true. Those moved to **helix** (pimm-data
> `b592600` / helix `8075a53`): pimm-data keeps `densify`, which every detector
> family needs, and gave up the physics only one family applies. The transform
> registry raises on duplicate registration, so the two packages are lockstep.
> See helix `docs/ARCHITECTURE.md` §3. The design reasoning below is kept as
> written; only the ownership changed.

**Status:** design, pre-implementation. Supersedes the multi-modality/namespacing
parts of `RESTRUCTURE.md` (which documents already-shipped work: `labels=`,
nested namespaced `Collect`, the post-collate dense path). This document is the
authoritative target for the next rewrite.

## 0. One-paragraph summary

A batch is a **flat dict of underscore-prefixed keys** (`step_coord`,
`sensor_offset`), bare when there is a single part. The pipeline is **one list of
transforms** modelling **map → reduce → map**: per-event transforms (a `map`),
then `collate` (the `reduce` — aggregation), then per-batch transforms (a `map`).
Every transform has **one format** — it declares the part(s) it operates on via
`on=` and returns prefixed keys tagged with a **role** (`point` / `instance` /
`edge` / `label` / `event`). `collate` is role-driven and is the only thing that
"knows about" batching. There is **no separate batch-transform concept**:
GPU/dense ops (densify/noise/digitize) are ordinary `scope='sample'` transforms
placed after a `ToDevice` step and run by the ordinary `Compose` (no runner). Views, graphs, fusion, tokenization are all just transforms or
producers over this contract.

---

## 1. The execution model: map → reduce → map

```
[ per-event maps … , Collect ]   ─ collate (reduce) ─   [ ToDevice , per-batch maps … ]
        1→1 per event                  N→1                     1→1 per batch
        (workers, numpy/cpu)        (worker→main)            (main, on device)
```

- **Per-event maps**: ordinary transforms over a single event. Run in DataLoader
  workers, CPU, numpy. `Collect` is the *last* per-event map (project + tensorize).
- **`collate` is the reduce** — role-driven aggregation, the implicit step at the
  head/tail seam. The user never writes it.
- **Per-batch maps**: transforms over the assembled batch, on device. `ToDevice`
  opens this segment. Densify/noise/digitize live here (by cost, see §7).

The framework **splits the one list at `Collect`**: head → `Dataset.__getitem__`,
reduce → `collate_fn`, tail → the ordinary `Compose` run by the trainer
(`on_after_batch_transfer` for Lightning, or a one-line loop). There is NO runner.

**Two hard fences (the only placement constraints):**
1. **No CUDA before the reduce** — workers are forked; per-event transforms cannot
   touch the GPU. Anything device-bound is post-`Collect`.
2. **`scope='batch'` (cross-sample) must be post-reduce** — see §4. None exist today.

Everything else is free placement by cost.

---

## 2. The batch structure

- **Single part → bare flat**: `{coord, feat, offset, segment}`. The 95% / new-user
  case. Never nested. Byte-identical to the current single-modality output.
- **Multiple parts → flat, underscore-prefixed**: `{step_coord, step_feat,
  step_offset, sensor_wire, sensor_offset, …}`. A *part* is the set of keys sharing
  a prefix.
- **Unprefixed keys = whole-event**: `name`, `split`, a regression `target`. Collate
  **stacks** tensors to `(B, …)` and **lists** non-tensors. The event owns these;
  they are not under any part. (A per-event quantity *about* a part — e.g. a count —
  is prefixed, `step_count`, and stacks via the catch-all role.)
- **`offset` is `(B,)` cumulative, NO leading 0**, per part (`step_offset`) — the
  *existing* convention (`offset[-1] == ΣN`, `len == B` or `B·num_views`). **Do NOT
  add a leading 0.** Every pimm model (`offset2bincount` prepends its own 0), the
  MultiCrop golden (`global_offset[-1] == N`), and pimm-data's own
  `dense_ops.offset2batch` assume no-leading-0; changing it silently corrupts all of
  them. Helpers (`to_batched_coords`, `batch[i]`) prepend the 0 *internally*.

**Separator is `_` (underscore), not `.`** — pimm models already read flat
`global_coord`/`global_offset`; `.` collides with `Point.coord` attribute access and
with the `_`-prefixed-metadata convention. Dotted keys are rejected.

---

## 3. Parts and roles

A part = its keys + `<part>_offset` + a small, droppable **`_roles`** map naming the
**non-default** keys (everything unlisted is `point`). Roles drive both `collate`
and subsampling:

| role | meaning | collate | subsample (`index_operator`) |
|---|---|---|---|
| **point** (default) | one row per point | concat by `offset` | slice |
| **raw** | per-point but immutable (the densify COO: `wire/time/value/plane_gid`) | concat by `offset` | **NOT sliced** (densify needs the immutable raw COO) |
| **instance** | rows in the part's SECOND row-space (e.g. `bbox (K,8)`; packed waveform samples) | concat; counted by the role-declared offset key `ok` | slice by `ok` |
| **edge** | index array into a cloud (`edge_index`) | concat + **shift** by referenced part's running node count | **remap** (drop edges to removed rows, reindex) |
| **label** | categorical grouping id (`cluster_id`, `group_id`) | **compact per event to 0..K-1, then** concat + add running distinct-count base, joint across the declared group | slice |
| **event** | NOT per-point (a whole-event scalar `target`, a part-summary `step_count`, or a pre-collate dense grid) | **stack** → `(B,…)` / list | leave |

`edge` role value: `'self'` (indexes its own part) or `(src, dst)` (cross-store
bipartite — row 0 shifts by `src`'s node count, row 1 by `dst`'s). Cross-store edges
are **producer-only** (a single-part `Apply` can't see the other part). `label` role
preserves the hierarchy by **compacting per-event ids to `0..K-1` first**, then adding
a single running distinct-count base to the *whole declared block* (raw FK ids are
not dense, so the compaction is required — a cluster still maps to one group after).

**Instance role = a part's second row-space.** A part has one row-space by default
(points, counted by `<part>_offset`); it may carry a second, independent one counted by its
own offset key (same `(B,)` no-leading-0 convention). The `('instance', ok)` role declares
its rows live there, **naming the offset key `ok`** (role-declared, not suffix-bound). The
second space can be coarser than points (instance `bbox`, `ok=<part>_inst_offset`) or finer
(packed per-chunk waveform samples, `ok=<part>_wave_offset` — the optical loader, emitted
via `Collect(offset_keys_dict=dict(offset='pmt_id', wave_offset='adc'))`).
`instance`-role keys are concatenated at collate and `split_event` slices them by `ok`'s
span — **never** the point `offset`. Distinct from the per-point
`instance` *index* column (which is `point`-role: one row per point, naming each point's
instance). Three contracts the producer must honour: (1) emit per-event instance ids
compacted to `0..K-1` so each point indexes that event's `bbox` rows directly — global
`instance`→`bbox` indexing is a *collate output* (add `node_bases(inst_offset)[event]`),
not renumbered in place like `label`; (2) `split_event` undoes the concat so the
`0..K-1` ↔ row correspondence holds again per event; (3) **build instances LAST** — after
all point subsampling, else dropping points (point space) without the matching `bbox` rows
(instance space) desyncs the two. pimm-data carries instances through collate/split; the
producer (InstanceParser) is pimm-side, fed in as config.

**Default role = `point`.** A key is `point` if its first dim matches the part's
`offset` total **and** it's not on the part's `raw`/`event` declaration. `_roles`
lists only the non-default keys (raw/edge/label/instance/event). This generalizes
today's `index_valid_keys`, BUT note: a `(1,)`-shaped per-event-in-part key
(`step_count`) does **not** match the point total — it MUST be explicitly tagged
`event` (it is *not* auto-`point`), so collate stacks it and exempts it from the
offset concat. The densify COO keys MUST be tagged `raw` so subsampling never slices
them.

---

## 4. The transform contract (one format)

A transform is a callable. It declares **`on`** — the part(s) it reads/operates on —
and returns prefixed keys (plus `_roles` for non-`point` outputs, plus
`<part>_offset` for a new cloud).

Two shapes:

- **Blind transform** (the augmentation library: `GridSample`, `RandomRotate`,
  `RemapSegment`, `InstanceParser`, …): part-agnostic, reads **bare** canonical keys
  (`coord`, `segment`). Targeted at a part by **`Apply(on='step', transforms=[…])`**,
  which exposes `step_*` as bare keys, runs the inner transforms, and re-prefixes
  (translating `_roles`/per-point membership bare↔prefixed). **A multi-part
  `Apply(on=('step','sensor'), …)` is IMPLICITLY shared** — it runs the transforms
  once per part with the SAME RNG draw (restore RNG state before each), co-registering
  them (same rotation/flip). There is **NO `shared` flag.** The config *shape* carries
  the intent: **bundled (`on=tuple`) = shared/together; separate `Apply` blocks =
  independent.** Views never use this path — their independent per-crop augmentation
  is `MultiCrop`'s job.
- **Producer** (`MultiCrop`, `SetupGraph`, `BuildNexus`, fusion): part-aware,
  declares `on=` (one part, or a tuple to gather), returns NEW prefixed keys + roles.

**Geometric transforms already scope to `coord`/`vertex`/`normal`** and leave
`origin_coord` (the un-augmented match target) untouched — verified in
`_rotate_about_center`. So `origin_coord` is safe without any extra flag;
transforms that take values per key (`RandomJitter(keys=…)`) already expose it.

**`scope` = arity, not placement:** `'sample'` (per-single, placeable anywhere
fork-safety allows — densify/noise/digitize are here) vs `'batch'` (cross-sample —
mixup/batch-stats; must be post-reduce; none exist today).

`index_operator` is **roles-aware**: subsampling a part slices `point`/`instance`,
**remaps** `edge` (drop+reindex), **does not touch** `raw` (the densify COO) or
`event`. For this to hold, **(a)** every blind transform that drops/reorders rows
MUST route them through `index_operator` (no ad-hoc `coord = coord[mask]`), and
**(b)** `Apply(on=…)` injects the part's `_roles` into the bare view so
`index_operator` can see them, then re-prefixes on exit. Cross-store edge values
(`(src,dst)` naming *other* parts) have no bare form, so they are producer-only and
never pass through a single-part `Apply`. Given (a)+(b), "subsample then build graph"
is order-independent; without (a) it is not — this is a contract, not an emergent
property.

---

## 5. collate — the role-driven reduce

`collate_fn` is a dumb structural reduction. Per part, per key, dispatch on role
(§3): concat-by-offset (point/raw), concat-by-inst-offset (instance), concat+shift
(edge), compact+concat+distinct-renumber (label), stack/list (event/unprefixed).
`offset` stays `(B,)` no-leading-0 (§2).

**`_roles` SURVIVES collate** — carried on the batch (e.g. `batch['_roles']`), NOT
dropped — because `batch[i]`/`split_event` (rebasing edges, slicing instances) and
the post-collate dense transforms (`on=`-targeted) need to know each key's role
*after* the reduce. Other `_`-prefixed temporaries are dropped.

**Typed internally, bare at the boundary:** collate builds lightweight typed parts
so it can assert invariants (`coord.shape[0] == counts.sum()`, monotonic offset,
edge indices in range), then `.to_dict()` (+ the `_roles` map) — the consumer
receives a **plain dict of tensors**, no wrapper type to import (pimm-data hands
batches to *someone else's* model).

FD-IPC preserved: `Collect` tensorized every leaf before collate, so worker→main is
file-descriptor sharing, not pickling.

---

## 6. Helpers (shipped, not user-reimplemented)

- **`to_batched_coords(batch, part) → [N, 1+D]`** — reconstruct the
  `[batch_id, x, y, z]` column MinkowskiEngine/spconv require. Sparse-conv has no
  offset-only path; this lives in one place, not every model.
- **`batch[i]` / `split_event(batch, i)`** — one event, with all index columns
  **rebased** to that event. Hand-rolled index-rebasing is the #1 bug source; ship it.
- **`content_seed(name, …)`** — `AddNoise` SELF-seeds per event from
  `batch['name']` (no runner injects seeds).

---

## 7. The post-collate tail (no batch-transform concept)

densify/noise/digitize are **`scope='sample'` transforms** (each event independent —
offset-driven scatter, per-event seed, elementwise). They are the same format as any
transform. Placement is a **cost choice**:

| placement | cost |
|---|---|
| post-`Collect`, after `ToDevice` (GPU) | born-on-GPU; only sparse crosses PCIe — **default** |
| post-`Collect`, before `ToDevice` (CPU main) | CPU densify, then move the dense grid |
| pre-`Collect` (per-event, workers) | ships dense over IPC; collate **stacks** the per-event grids (`event` role) |

`ToDevice(device=…)` is an ordinary transform — the explicit `.to(device)` step that
opens the per-batch segment, which the ordinary `Compose` runs (dense ops are
`scope='sample'`, self-seeding). There is NO `apply_batch_transforms`/`build_batch_transforms`
runner surface — `build_sensor_gpu_stages` just returns a `Compose`.

The only thing that would ever need a true batch transform is a genuinely
cross-sample op (mixup, batch statistics, cross-event edges) — `scope='batch'`,
post-reduce. pimm-data has none; they are model-side.

---

## 8. Cross-part alignment & label hierarchy

- **`Align(to='image', parts=('cluster',))`** — a first-class directive that aligns
  parts row-for-row (reference-tensor dedup, à la SPINE `clean_sparse_data`).
  Multi-task voxel heads silently misalign without it.
- **Label block + joint renumber** — id columns of a part (`cluster_id`, `group_id`,
  `interaction_id`, …) are kept as an aligned block and declared as a `label` role
  group. The **FK join stays in the dataset** (`get_data` decorates raw per-point
  ids, as today — `_label_decorate.py`); only the **cross-event renumber** is new and
  lives in `collate`: compact each event's ids to `0..K-1`, then add a running
  distinct-count base to the whole declared block (raw FK ids aren't dense, so
  compaction is required) → the cluster→group→interaction hierarchy stays consistent.
  Every per-point transform must permute/drop all of a part's per-point keys
  atomically (tested). *There is no current renumber to keep green — a new multi-event
  golden is the only oracle.*

---

## 9. Config-load validation

`Collect` validates only what is **statically declarable** at build: `namespaces`
parts exist in declared `modalities`, `on=` names a known part, the EITHER/OR Collect
form. Roles that are shape-derived at runtime are *not* build-checkable. `Compose`
validates ordering (`Collect` present and last per-event; `scope='batch'` not
pre-reduce — dormant today). This is net-new (today `Collect` validates almost
nothing at build); keep it to the statically-knowable to avoid false promises.

---

## 10. Worked examples

### Single cloud (new user) — bare
```python
JAXTPCDataset(modalities=('step',), labels='pdg', transform=[
    dict(type='Apply', on='step', transforms=[
        dict(type='GridSample', grid_size=0.5, mode='train'),
        dict(type='RemapSegment', scheme='motif_5cls')]),
    dict(type='Collect', keys=('coord','segment'), feat_keys=('coord','energy'))])
# -> {coord, feat, segment, offset, name, split}
```

### C4 — sparse step + dense sensor (the one list)
```python
transform = [
    dict(type='Apply', on='step', transforms=[
        dict(type='GridSample', grid_size=0.5, mode='train'),
        dict(type='RemapSegment', scheme='motif_5cls')]),
    dict(type='Collect', parts={'step': dict(keys=('coord','segment')),  # last per-event map
                                'sensor': dict(keys=('wire','time','value','plane_gid'))}),
    # ── collate (reduce) ──
    dict(type='ToDevice', device='cuda'),                              # opens per-batch segment
    dict(type='Densify',  geom=_geom, modality='sensor'),             # scope='sample'
    dict(type='AddNoise', geom=_geom, modality='sensor', coherent=True),
    dict(type='Digitize', geom=_geom, modality='sensor', n_bits=12),
]
# -> {step_coord, step_segment, step_offset, sensor_wire, sensor_offset, sensor_dense, name, split}
```

### panda mode — global/local multi-crop SSL (config-only)
```python
transform = [
    dict(type='Apply', on='step', transforms=[
        dict(type='GridSample', grid_size=0.001, mode='train'),
        dict(type='Copy', keys={'coord': 'origin_coord'})]),       # clean reference
    dict(type='MultiCrop', on='step',                              # OWNS shared-pre + local-within-global
         view_keys=('coord','origin_coord','energy','segment_motif'),
         global_view_num=2, global_view_scale=(0.4,1.0),
         local_view_num=6,  local_view_scale=(0.1,0.4),
         global_shared_transform=[dict(type='MultiplicativeRandomJitter', keys=('energy',))],
         global_transform=[dict(type='CenterShift'),
                           dict(type='RandomRotate', axis='z'),  # only coord/vertex/normal -> origin_coord SAFE
                           dict(type='RandomFlip')],
         local_transform =[dict(type='CenterShift'),
                           dict(type='RandomRotate', axis='z')]),
    dict(type='Collect', namespaces=('global','local'),
         feat_keys=('coord','energy'), carry=('origin_coord','segment_motif'))]
# -> {global_coord, global_offset, global_feat, global_origin_coord, global_segment_motif,
#     local_coord, ..., name, split}   (global_offset has B*num_global entries — packed crops)
```

### spine mode — multi-product multi-task (config-only)
```python
modalities = ('image','cluster','particle')
transform = [
    dict(type='Apply', on='image', transforms=[dict(type='Voxelize', grid_size=0.5)]),
    dict(type='Align', to='image', parts=('cluster',)),            # row-for-row alignment
    dict(type='Collect', namespaces={
        'image':   dict(keys=('coord',), feat_keys=('coord','value')),
        'cluster': dict(keys=('coord',),
                        labels=('cluster_id','group_id','interaction_id','particle_id','pid','shape'),
                        renumber=('cluster_id','group_id','interaction_id')),   # joint distinct-count
        'particle':dict(keys=('points',))})]
```

### Producers
```python
class SetupGraph:                  # graph on one cloud
    on = 'step'
    def __call__(self, d):
        return {'step_edge_index': knn(d['coord']), '_roles': {'step_edge_index': ('edge','self')}}

class BuildNexus:                  # cross-store bipartite edges
    on = ('hit', 'sp')
    def __call__(self, d):
        return {'nexus_edge_index': bipartite(d['hit_pos'], d['sp_pos']),
                '_roles': {'nexus_edge_index': ('edge', ('hit','sp'))}}
```

### Model consumption
```python
batch = next(loader)                                   # plain dict; offset (B+1)
xb = to_batched_coords(batch, 'image')                 # [N, 1+3] -> ME.SparseTensor
z  = backbone(coord=batch['global_coord'], feat=batch['global_feat'], offset=batch['global_offset'])
ev = batch[3]                                          # one event, indices rebased
```

---

## 11. Author-review findings folded in

From the pimm + SPINE author reviews of an earlier draft:
- **`MultiCrop` keeps its internal couplings** — `global_shared_transform` (globals-only,
  pre-crop) and local-within-global center-sampling/cover-mask — NOT split into `Apply`
  blocks (pimm).
- **Geometric transforms `keys=`-scoped; `origin_coord` protected** (pimm) — silent-SSL-
  degradation guard.
- **Per-instance ragged role** (`bbox`) + **`(B+1)` packed-offset invariant**
  `len(part_offset)==B*num_views` (pimm).
- **`to_batched_coords` + `batch[i]` shipped; typed-internal/bare-at-boundary** (SPINE) —
  sparse-conv needs the batch-id column; per-event slicing is the #1 bug source.
- **Label ids = jointly-renumbered block by distinct-count, not `'self'`** (SPINE) — the
  scalar inc tag was insufficient for the cluster/group/interaction hierarchy.
- **`Align` cross-part directive + config-load validation** (SPINE).
- **Underscore separator, not dotted** (pimm) — models already read flat `global_*`.

---

## 12. What breaks / migration

- **Multimodal output: nested `{step:{…}}` → flat `step_*`.** Every consumer reading
  `batch['step']['coord']` → `batch['step_coord']`. Single-cloud and pimm's existing
  flat `global_*` are unaffected.
- **`offset` convention is UNCHANGED** (`(B,)` no-leading-0). No model migration for
  offset, and **single-cloud (C1/C2) byte-identity holds** (PF1 golden unchanged) —
  the earlier `(B+1)` idea is dropped (it would have broken every consumer silently).
- **`Streams` deleted; `ApplyToModalities` folded into `Apply(on=)` (multi-part = implicitly shared, no flag);
  `from_`→`on`; `build_batch_transforms`/`apply_batch_transforms` REMOVED — dense ops
  run via `Compose`; `build_sensor_gpu_stages` returns a `Compose`.
- **The numpy `Densify`/`AddNoise`/`Digitize` trio in `detector_transforms.py` is
  REWRITTEN, not relabeled** — today they operate on a nested `sensor` sub-dict keyed
  by string plane label (`sub['raw']`/`sub['shape']`), with per-event seed from
  `sub['name']`; the new ops are `on='sensor'` transforms over flat keys. The
  per-sub-dict `name` carry (`jaxtpc.py` injects `data['sensor']['name']`) must be
  preserved for noise seeding.
- **`MultiCrop` is a cross-repo EXTRACTION**, not a clean addition — it lives in
  `particle-imaging-models/pimm/datasets/transforms.py` (the `global_*`/`local_*`
  packing + couplings). Moving it into pimm-data means deleting pimm's copy and
  repointing the lejepa/panda configs. Coordinate cross-repo.
- **`offset_keys_dict=dict()` empty-suppress semantics must be preserved** when
  unifying the two `Collect` forms — the single form treats `{}` as "emit no offset"
  (lejepa configs depend on this because `global_offset`/`local_offset` come from
  `MultiCrop`), but the namespaced form's `or dict(offset="coord")` would override
  `{}`. Unify on "explicit `{}` = suppress."
- **Test files to rewrite/delete** (so "keep the suite green" is honest):
  `test_streams.py` (delete), `test_apply_to_modalities.py` (port to `Apply(on=,
  )`), `test_batch_transforms.py` (rewrite — `Batch*`/`offset2batch` contract),
  `test_user_transforms.py` (port — drops `build_batch_transforms`/`ToDevice` imports),
  `test_phase1.py` (labels= survives; namespaced-Collect assertions flip to flat keys),
  `test_pf1_golden.py` (multimodal goldens re-frozen to flat keys; single-cloud
  unchanged).
- pimm config/model migration (nested→flat keys) is cross-repo and must be coordinated.

---

## 13. Phased implementation plan

0. **Lock the spec** (this doc).
1. **Roles + collate rewrite** (`collate.py`, new `_roles.py`): role-driven reduce,
   `offset` UNCHANGED `(B,)`. Dual-path = must handle BOTH the still-nested current
   datasets AND flat keys until Phase 2 lands (the burden is on collate, not Collect).
   *Highest risk; net-new — no existing roles/edge code to evolve from.*
2. **Flat-prefixed representation** (`Collect`, datasets): namespaced `Collect` emits
   `part_*`; single stays bare. Working repr stays nested; flatten at `Collect`.
3. **Transform contract** (`detector_transforms.py`, `transform.py`): `ApplyToModality`
   → `Apply(on=)` (multi-part implicitly shared, NO flag), fold `ApplyToModalities`; geometric `keys=` scope;
   `index_operator` roles-aware (remap edges).
4. **Producers & dense as transforms**: `MultiCrop` (cross-repo extraction, keep
   couplings), `SetupGraph`, `BuildNexus`, `Align`; **rewrite** the numpy
   `Densify`/`AddNoise`/`Digitize` trio (sub-dict → `on='sensor'` flat transforms,
   `scope='sample'`); `dense_ops`/`offset2batch` unchanged (offset stays `(B,)`).
5. **Helpers**: `to_batched_coords`, `batch[i]`, typed-internal collate. Dense ops run
   via `Compose` (no runner); `build_sensor_gpu_stages` returns a `Compose`.
6. **Validation** (`Collect`/`Compose` at build).
7. **Migration + cleanup**: delete `Streams`/`Batch*`; migrate configs/models;
   empty-part/feat/val contracts.
8. **Tests/goldens**: per-role, per-helper, per-mode (panda packed invariant, spine
   align+renumber), single-cloud byte-identity, real-data.

Order: 0 → 1 → 2 → 3 → (4 ∥ 5) → 6 → 7 → 8. Keep the suite green incrementally
(dual-path collate; flip goldens at Phase 2).

---

## 14. Decided defaults

- Separator `_`; `offset` `(B,)` **no-leading-0** (unchanged from today); bare-when-1;
  unprefixed=whole-event.
- `on=` (the part a transform operates on); `scope`=arity; default role `point`;
  `raw` role for the immutable densify COO; `event` role explicitly tagged (not
  shape-derived) for per-event-in-part keys.
- `_roles` **survives collate** (carried on the batch); other `_`-temporaries dropped.
- Multi-part `Apply(on=tuple)` is **implicitly shared** (one RNG draw, co-registration) —
  **no `shared` flag**; independent augmentation = separate `Apply` blocks.
- No batch-transform concept and NO runner; `ToDevice` opens the per-batch segment,
  which the ordinary `Compose` runs (dense ops are `scope='sample'`, self-seeding).
- Typed-internal/bare-at-boundary; ship `to_batched_coords`/`batch[i]`.
- `MultiCrop` owns crop couplings; geometric transforms `keys=`-scoped.
- Label renumber: FK join stays in dataset; compact-then-distinct-count-shift in collate.

---

## 15. Design-review resolutions (two-agent pass)

Blocking issues found and resolved in this revision:
- **`(B+1)` offset → reverted to `(B,)` no-leading-0.** It silently broke every pimm
  model, `dense_ops.offset2batch` + its test, the MultiCrop golden, and contradicted
  single-cloud byte-identity. Helpers prepend the 0 internally. (§2, §5, §13, §14)
- **`event` role disambiguated + `raw` role added.** `event` is explicitly tagged
  (not shape-derived) and exempt from offset-concat; `raw` (densify COO) is per-point
  but never sliced. (§3)
- **`_roles` survives collate** — needed by `batch[i]` and post-collate dense ops. (§5)
- **`Apply(on=)` ↔ `index_operator` contract specified**: blind transforms must route
  row-drops through `index_operator`; cross-store edges are producer-only. (§4)
- **Label renumber = compact-per-event then distinct-count base; FK join stays in
  dataset.** (§3, §8)
- **§12 expanded** with the numpy-trio rewrite, the cross-repo `MultiCrop` extraction,
  the `offset_keys_dict=dict()` suppress semantics, and the test delete/rewrite manifest.

Still-open items to pin during Phase 1 (not blocking the start, but resolve before the
relevant phase):
- **`instance` role mechanics** — who emits `<part>_inst_offset`, and how
  `index_operator` slices instances vs points (different cardinalities). (Phase 1/4)
- **Empty tail (sparse-only) = no-op**: an empty post-`Collect` segment means the
  trainer's normal `.to(device)` applies; no dense Compose runs. (Phase 5)
- **`Align` must be the last pre-`Collect` structural op** for the part it aligns
  (a later subsample would desync). (Phase 4)
- **`name` is guaranteed present + list-typed post-collate** (seed identity). (Phase 5)

---

## 16. Implementation status (branch `redesign-flat-parts`)

**DONE — committed, suite 370 passed:**
- Roles + role-driven `collate` (`_roles.py`, `collate_with_roles`); `offset` `(B,)`.
- Flat-prefixed `Collect` (namespaced → `step_*` + `_roles`); single bare unchanged
  (PF1 golden intact); `offset_keys_dict={}` suppress fixed; build-time validation.
- `Apply(on=)` implicit-shared (no flag); `ApplyToModalities` folded/deleted;
  `ApplyToModality` back-compat alias.
- `Streams` deleted → `MultiCrop` (packed `global_*`/`local_*` views, all in pimm-data).
- Producers: `SetupGraph` (self edges), `BuildNexus` (cross-store), `Align`
  (multi-task row alignment) — graph ops in **torch** (`cdist`/`topk`).
- `index_operator` roles-aware self-edge remap (subsample/graph order-independent).
- Dense path: `Densify`/`AddNoise`/`Digitize` are a single dispatching trio each
  (the `Batch*` variants were deleted) — they take `geom=`/`modality=`, run
  `scope='sample'` on flat `sensor_*`, and dispatch numpy (per-event sub-dict) vs
  torch (post-collate batch) on whether `<part>_offset` is present.
- One-list split / post-collate tail: a single `transform` list is split at the
  terminal `Collect` (`ShardEventDataset._split_at_collect`) into
  `dataset.transform` (per-event head) + `dataset.batch_transform` (the
  `ToDevice`→`Densify`→`AddNoise`→`Digitize` tail a trainer runs post-collate,
  e.g. `on_after_batch_transfer`). Recipe: `configs/jaxtpc/sensor_dense_gpu.py`;
  bench `examples/bench_dense.py` (GPU ~29× CPU, bit-exact vs numpy oracle).
- Helpers: `to_batched_coords`, `split_event`/`batch[i]`.

**Deliberately deferred (not omissions):**
- numpy `Densify/AddNoise/Digitize` still work as pre-collate transforms via `Apply`
  scoping (sub-dict); a flat `on=` rewrite is optional cleanup, not correctness.
- Geom auto-injection: the dense tail still takes an explicit `geom=` (config-derived
  registry); having the dataset fill its own `plane_geometry()` when `geom` is omitted
  is a future nicety.
- pimm-side config/model migration (nested→flat keys, `Apply(on=)`, `MultiCrop`) —
  pimm's repo, config-level (pimm calls pimm-data).
