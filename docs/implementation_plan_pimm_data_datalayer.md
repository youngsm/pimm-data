# Implementation Plan — pimm-data data layer

Status: **build spec**, derived from `engagement_plan_transform_dataset_placement.md`
(v2, decisions D1–D38) + a final 4-agent implementation-readiness review. This
is the executable plan; the engagement doc remains the decision record (spine of
truth). Near-term structure is **single-stream-per-task (D35)**.

Repos: pimm-data `/sdf/group/neutrino/omara/pimm-data` (installed **editable**);
pimm (Pointcept fork, branch `research`) `/sdf/home/o/omara/.claude/jobs/21ffc656/particle-imaging-models`.

---

## 1. Architecture (recap)

pimm-data owns the entire data layer; pimm keeps DDP/model/hooks. A
**`MultiModalEventDataset` base** owns event selection (multi-source mixture,
hash-on-identity holdout, cheap-`n_hits` min-points, `event_identity`/`split`);
`LUCiDDataset`/`JAXTPCDataset` inherit it; `LUCiDEventSSLDataset` dissolves into
base + config. Datasets emit **nested per-stream** dicts; each task selects **one**
stream via `Collect(stream=)` → the **existing single-stream collate** → `Point`.
That stream carries its own per-point labels, **decorated from `labl` in the
dataset** via an extensible `label_config`. **Multi-stream-in-batch (namespaced
collate), cross-stream batch joins, and aggregation are FUTURE**; **Track B
(densify/noise) is JAXTPC-only, deferred.**

**Validated (final review):** all near-term tasks (SSL/sensor, semseg+instance/hits,
3D/step) pass end-to-end under single-stream, modulo the four concrete gaps in §3.
Collate is a byte-identical REPLACE again (verified `diff`).

---

## 2. Build order (rollout runbook)

Each step is independently landable + revertable. **Invariant: existing
PILArNet/panda/hmae training works at every step.**

- **Step 0 — Test matrix (gate, no code move).** Stand up the parity/determinism
  harness in pimm-data using `src/pimm_data/testing.py` fixtures (§6). Pure gate.
- **Step 1 — Additive build in pimm-data (no pimm change):** (a) transform merges
  §3.1; (b) `index_operator` prefix-match §3.2; (c) `MultiModalEventDataset` +
  `TestModeMixin` §3.3 (internal refactor of existing pimm-data datasets, gated on
  Step 0); (d) reader `read_meta` + stream surfacing §3.4; (e) `label_config`
  decorator §3.5. Gate: Step-0 suite green.
- **Step 2 — Re-export shim** in `pimm/datasets/__init__.py`: re-export +
  re-register pimm-data classes into pimm's `DATASETS`/`TRANSFORMS` (start with the
  byte-identical REPLACE files). Gate: PILArNet 1-step smoke through the shim.
- **Step 3 — Flip transforms + PILArNet** (after the Rb pilarnet merge §5):
  PILArNet/panda/hmae are transform-compatible (`segment_motif`/`PDGToSemantic`
  resolve from pimm-data). Gate: identical first-batch tensors vs vendored.
- **Step 4 — Migrate JAXTPC configs (Risk Ra, §4), then flip `JAXTPCDataset`;
  dissolve `LUCiDEventSSLDataset`** → base + LUCiD config. Gate: JAXTPC semseg +
  LUCiD SSL configs build and run 1 step.
- **Step 5 — Delete vendored files** (last commit, single `git revert` restores).
  Gate (D33): grep `seg|resp|corr|output_mode` in `configs/` clean; `jaxtpc_seg.py`
  migrated; `__init__.py` stale import fixed; full parity suite green; soaked ≥1
  full PILArNet run.

---

## 3. Component specs

### 3.1 Transform merges (pimm-data `transform.py`) — the only `transform.py` gaps

Everything else is byte-identical (`InstanceParser`, `HierarchicalMaskGenerator`,
`HMAECollate`, `ComputeAnchors`, `CropBoundary`, `LocalCovarianceFeatures`,
`RandomDrop`, color/jitter, crops). **Do NOT overwrite pimm-data's `Collect`** — it
is ahead of the branch (`stream=`, tensor autoconvert, passthrough); merge
direction is reversed for that one class.

1. **`RelativeLogNormalize`** (NET-NEW; D11/D31). `(keys=("time",), scale=50,
   max_val=4000, out_min=-1, out_max=1)`. Per-event `x -= x.min()` (**the D31
   negative-time correctness step** — WAND `T` reaches ~−240 ns), `clip(0,max_val)`,
   `log1p(x/scale)/log1p(max_val/scale)`, affine to `[out_min,out_max]`, clip, f32.
   Must run before `Collect`/any reorder on the time column (non-idempotent).
2. **`GridSample` reducers → `{key: op}` map** (D29), ops `sum/min/max/mean/first`,
   with `sum_keys`/`min_keys` back-compat shim (explicit `reducers` wins). `min` is
   the only one on the branch; `max/mean/first` are net-new. `mean` = sum/count
   (promote int→float); `first` = sorted-order representative
   (`idx_sort[cumsum(insert(count,0,0)[:-1])]`) for determinism (≠ random survivor —
   document). Int/float fill split for min/max. Back-compat parity: `min_keys` path
   must equal the branch byte-for-byte.
3. **`LogTransform.clip`** (D11/D31): add `clip=False`; when set, clamp pre-`log10`
   domain in both `log_transform`/`linear_transform`. Default off (no regression).
4. **`MultiViewGenerator.get_view` guard** (D11): `if max_size<=0: raise`;
   `size=max(1,min(max_size,size))`. Only `get_view` changes.
5. **v3 vertex/`is_primary` plumbing** (D11): port module helpers `_valid_vertex_mask`
   / `_apply_to_v3_vertex` / `_translate_axis`; add `_apply_to_v3_vertex(...)` calls
   inside `NormalizeCoord` (refactor to compute centroid/scale once), `PositiveShift`,
   `CenterShift`, `ConditionalRandomTransform`, `RandomShift`, `RandomRotate`,
   `RandomRotateTargetAngle`, `RandomScale`, `RandomFlip`; add `"vertex"` to default
   `index_valid_keys` + the v3 `is_primary` append. **`PointClip` stays vertex-blind.**
   All guarded by `_valid_vertex_mask` → no-op when no `vertex` key (dormant until a
   v3 dataset stamps `revision`).
6. **`MixedScaleGeometryMultiViewGenerator`** (NET-NEW subclass; D11): port verbatim
   (fine-scale local crops via kNN-PCA `_directional_complexity`/`_geometry_pool`).
   Optional — not on the SSL critical path.

### 3.2 `index_operator` prefix-match (D25)

Current default `index_valid_keys` is a hardcoded list omitting `time`,
`sensor_idx`, `particle_idx`, `plane_id` → N-changing transforms (`GridSample`
train, `RandomDropout`, `SphereCrop`, `ShufflePoint`) silently desync those
columns. **Fix:** after the default-list init, append keys present in the dict
matching `segment`/`instance`/`target` (underscore-boundary match, not bare
`startswith`) + `{particle_idx, sensor_idx, plane_id}`. **Exclude per-event
`target_*`** via a leading-dim ≠ `n_points` shape check (per-event targets must not
be point-subset). This makes the per-config `Update(index_valid_keys=...)` no longer
load-bearing. (`ApplyToStream` doesn't propagate the list, so any explicit `Update`
must itself be inside `ApplyToStream`.)

### 3.3 `MultiModalEventDataset` (new `pimm_data/multimodal.py`)

Inherits `DefaultDataset` via a factored **`TestModeMixin`** (D30) — extract the
test-mode/`prepare_test_data` fragment-list/`inverse` path (3 identical copies in
`defaults.py`/`lucid.py`/`jaxtpc.py`) so seg eval's TTA works. Subclasses supply
`_build_readers(source_root, split, **kw)` + per-modality builders + the FK chain.

**Constructor (frozen):**
```python
MultiModalEventDataset(
    sources,                 # str | list[str|dict{root,label,config_id?,split?,weight?}]  (D9; "config"/"run" axis)
    modalities=('sensor',),
    *, split='train',        # holdout role: train|val|test|all
    holdout=None,            # {seed, fractions=(tr,va,te), strata='config'} | {n_per_config:int}
    min_points=None,         # int | {threshold, modality='sensor', op='>='}
    max_events=-1, mixture=None,   # {weights:'uniform'|list, mode:'replicate'|'sampler'}
    label_config=None,       # §3.5
    dataset_name='wc', transform=None, test_mode=False, test_cfg=None,
    loop=1, ignore_index=-1, cache=False, **reader_kwargs)
```
`data_root=` is a back-compat alias → `sources=[data_root]` (every existing config
unchanged). `__init__` order (D8 filter-then-hash-split): normalize sources →
per-source `_build_readers` → build/load manifest cache (§3.4) → min-points filter
(`>=`) → hash holdout → `max_events` → mixture → standard tail.

**Holdout (D26):** `bucket = blake2b(struct.pack('<qqq', seed, config_id,
source_event_idx)) / 2**64` → 3-way fractions; **config-stratified by folding
`config_id` into the digest** (no per-config bookkeeping). DDP rank-identical (pure
function of file-discovered `source_event_idx` + seed; **replaces the colleague's
`np.random.permutation`** which isn't rank/version-stable). `n_per_config` mode:
take the `k` smallest-`u` events per config. **Fallback** `(config_id, positional)`
+ one warning when `source_event_idx` absent.

**API the eval hook needs (preserve exactly):** `event_identity(idx) ->
(config_id, source_event_idx)` (modality-independent), public `self.split` (survives
`Subset`), `data_list` as `(source_idx, local_idx)` tuples, a `datasets`-equivalent
source list. `get_data_name` moves to the base **with a source prefix**
(`config_1/wc_sensor_0000.h5_evt007`) — fixes the cross-config filename collision.

**Common (base) vs different (subclass)** per D37: base owns selection/holdout/
min-points/mixture/identity/`get_data` dispatch/`event_label`+`config_id`
materialization; subclasses own readers (PMT vs wire/pixel), geometry, FK chains
(`particle_idx→category` vs `group_id→track→label`), and detector sub-selectors
(**JAXTPC `volume` is orthogonal, NOT the mixture axis**).

### 3.4 Reader additions (`pimm_data/readers/`)

Add a uniform **`read_meta(idx) -> {source_event_idx, n_hits}`** to every reader
reading **only `evt.attrs`** (never datasets) — the base's cheap selection probe;
`read_event` (hot path) untouched.

| reader | `n_hits` source (attr-only) | extra `read_event` surfacing |
|---|---|---|
| lucid_sensor | `evt.attrs['n_hits']` | — |
| lucid_step | `evt.attrs['n_segments']` (already read for `min_segments`) | — |
| lucid_hits | `evt.attrs['n_particle_hits']` | **+`T_reco`** |
| lucid_labl | (n/a) | **+`per_interaction`** scope (`source_type,t0,vertex_*,neutrino_pdg,neutrino_energy,contained`, CSR primaries) |
| jaxtpc_sensor | Σ plane `n_pixels` (walk plane groups) | — |
| jaxtpc_step | Σ vol `n_actual` | — |
| jaxtpc_hits | Σ vol `n_actual` | group_id already present |
| jaxtpc_labl | (n/a) | `track_interaction` already present |

`config_id` is **not** a reader field (assigned by the base per source). Names are
hardcoded (confirmed + uniform at `format_version=5`); values read at runtime;
absence → D26 fallback. **WAND schema confirmed (one-shard check, config_000001):**
- **`source_event_idx`** present as BOTH a per-file vector `config/source_event_idx`
  `uint32 (n_events,)` (sensor *and* labl) — use this (O(1)/file) for identity — AND
  a per-event attr `event_NNN.attrs['source_event_idx']` (fallback).
- **`n_hits`** = per-event attr `event_NNN.attrs['n_hits']`; **no** `config/n_hits`
  vector → cheap per-event scalar-attr walk (manifest-cached). Optional writer ask:
  add a `config/n_hits` vector for O(1)/file. (JAXTPC sensor: Σ`n_pixels` attr-walk.)
- **`per_interaction`** group fields: `source_type, t0, vertex_x, vertex_y, vertex_z,
  n_primaries, n_particles, neutrino_pdg, neutrino_energy_MeV, contained` + CSR
  primaries (`primary_{track_ids,pdgs,energies}_{data,offsets}`). **`target_vertex`
  = stack `vertex_{x,y,z}`** (three scalars, not a `(3,)` dset); surface CSR raw.
- **`instance_interaction` is one-hop**: `per_particle.interaction_idx` exists →
  `particle_idx → interaction_idx` directly (no per_track detour).
- `format_version=5` (reader docstrings say 3 — **fix the docstrings**);
  `per_particle.category` is `uint8`; `per_track` matches existing reader keys.

### 3.5 Label decoration (`label_config` + generic decorator)

**Decision (resolves the bare-vs-named gap):** the dataset decorator emits the
**named schema keys** (`segment_pid`, `instance_particle`, `instance_interaction`,
`target_vertex`, …) directly — this is what the configs/evaluators consume
(`panda/panseg` uses `segment_pid`/`instance_particle`). Bare `segment`/`instance`
remain a back-compat single-axis alias (or a config `Copy`). Readers emit **raw
FKs**; the dataset decorates (D20/D28).

`label_config` = list of axis specs (D38, open/extensible):
```python
dict(out="segment_pid",          scope="point", fk="particle_idx", source=("particle","category"), fill=-1)
dict(out="instance_particle",    scope="point", fk="particle_idx", source="self")
dict(out="instance_interaction", scope="point", fk="particle_idx", source=("track","interaction"))
dict(out="target_vertex",        scope="event", source=("interaction","vertex"))
dict(out="target_energy",        scope="event", source=("interaction","neutrino_energy_MeV"))
dict(out="event_label",          scope="event_broadcast", source="<mixture label>")   # per-point for the probe
```
Generic `_decorate_from_labl(sub, labl, fk_resolver)` generalizes the existing
`_decorate_*_from_labl`/`_lookup_per_*`; `fk_resolver` is the only per-subclass piece
(LUCiD `particle_idx`/`track_idx`; JAXTPC `deposit_to_track`/`group_to_track[gid]`).
**Per-event `target_*` stay per-event** (length-1/`(D,)`, not broadcast) → carried as
`_`-prefixed list-collated metadata, excluded from `index_valid_keys`.
**`event_label`/`config_id` are `scope="event_broadcast"` → per-point arrays** inside
the stream so `Collect(keys=[...,'event_label'])` lifts them and the probe slices by
offset. New axis families (edge/graph for NuGraph) = new spec + a prefix in §3.2.

### 3.6 Collate (single-stream — REPLACE)

Near-term collate is the **existing `collate_fn`/`point_collate_fn`** (byte-identical
to pimm's `utils.py`, verified) — REPLACE, no change. `event_label`/`config_id` reach
the batch as ordinary per-point columns (§3.5). **Namespaced multi-stream collate +
cross-stream offset rebasing (D23/D24) is a FUTURE additive path** that reuses the
same nested dataset + decoration; not built now.

### 3.7 Eval-hook rewiring (`pimm/engines/hooks/lucid_event_probe.py`)

Replace `_event_keys`'s reach into `data_list`/`datasets`/`source_root` (lines
~150–190) with the stable `event_identity(idx)` API (peel `Subset`, then
`{event_identity(i)}`); generalize `_format_event_key` to a multi-component tuple;
keep `_dataset_split`. The probe needs only per-point `event_label` collated
(§3.5) + model `point.feat`/`offset`.

---

## 4. Config migration (Risk Ra)

Only **two** configs break (grep-confirmed; `output_mode` has zero hits):
- `configs/detector/_base_/jaxtpc_seg.py` (+ child `semseg-pt-v3m2-jaxtpc-5cls.py`):
  `modalities=("seg",)` → `("step","labl")`; wrap per-stream ops in
  `ApplyToStream(stream='step', [...])`; replace `PDGToSemantic` with `label_key='pdg'`
  + `RemapSegment(scheme="motif_5cls")`; terminal `Collect(stream='step',
  keys=("coord","grid_coord","segment"), feat_keys=("coord","energy"))`. No model
  change (`in_channels=4`).
- `pimm/datasets/__init__.py:9` stale `from .lucid_dataset import LUCiDDataset` +
  the unimported `lucid_event_ssl` registration → fix in the shim so
  `LUCiDEventSSLDataset`'s successor registers.

**PILArNet/panda/hmae/voltmae/polarmae/lejepa (~14 configs) are unaffected** —
`type="PILArNetH5Dataset"` + `segment_motif`/`PDGToSemantic` resolve from pimm-data
identically once Rb (§5) lands.

---

## 5. Packaging & de-fork mechanics

- **Submodule, not loose editable install.** Add pimm-data as a git submodule under
  pimm (`libs/pimm-data`), `pip install -e` from `environment.yml`. Extend
  `scripts/train.sh:228` to `cp -r ... libs/pimm-data` (or record its `git rev-parse
  HEAD` into the experiment dir) so the code-snapshot captures the data-layer SHA —
  fixes the repro hole (editable + uncopied today).
- **Torch pin:** pin pimm-data `pyproject.toml` `torch` to the env (`==2.5.0` or
  `>=2.5,<2.6`) so `pip -e` never pulls a different CUDA build.
- **Registry (Rf):** keep pimm's `DATASETS`/`TRANSFORMS` authoritative for config
  lookup; the `__init__.py` shim imports each pimm-data class and **re-registers**
  it via `register_module(module=Cls)` (the pattern `lucid_event_ssl` already uses).
  Don't share one registry object across the boundary. Guard double-registration.
- **Rb — pilarnet v2→v3 merge into pimm-data:** add `revision="v3"` + the 6-wide
  `cluster_extra`/`is_primary` branch + `is_primary` propagation/mask/emit/concat-key;
  add the shared-`rotations` param to `_apply_random_90_rotation` (so overlaid events
  share orientation).
- **Stays in pimm:** `MultiDatasetDataloader` (`dataloader.py`, DDP/`comm`),
  `pimm/utils/registry.py` (model/hook registries), hooks/evaluators, and the thin
  `__init__.py` re-export shim (must keep exporting `point_collate_fn`/`collate_fn`/
  `inseg_collate_fn`/`DefaultDataset`/`ConcatDataset`/`build_dataset`/dataset classes/
  `MultiDatasetDataloader`).

---

## 6. Test matrix (Step 0; on `testing.py` synthetic fixtures — no GPU, no WAND)

Branch outputs are the **reference** for transform parity (import both modules,
seed `random`/`np.random`/`torch` identically, `assert_array_equal` per key):
- **Transform parity:** `RelativeLogNormalize` (incl. negatives, no NaN); `GridSample`
  `min_keys` back-compat byte-equal + new `max/mean/first` vs hand-rolled groupby +
  `first` determinism across seeds; `LogTransform.clip`; `get_view` guard (empty
  raises, 1-pt returns size-1); v3 vertex co-transform (+ `PointClip` unchanged);
  `MixedScale` equal arrays/offsets.
- **`index_operator` prefix-match** (new-behavior, not parity): per-point
  `segment_*`/`instance_*`/`*_idx` subset to new N; per-event `target_vertex` (len 3)
  not subset.
- **Holdout determinism:** same `(config_id, source_event_idx)` → same split under
  shard reorder / add-remove / worker count / machine.
- **Label decoration:** `segment_pid`/`instance_particle`/`instance_interaction` match
  a hand FK-gather using the fixture's known FK arrays (invariants in `testing.py`
  docstring); per-event `target_*` length-1.
- **Cheap == array min-points:** `n_hits`-attr filter selects the identical event set
  as array counting.
- **Migration smoke:** PILArNet/panda/hmae + migrated JAXTPC + LUCiD SSL configs build
  + one `__getitem__`/1-step through the shim.

Gate transform-parity **assertions** (not the design) on the colleague's branch being
fetchable; placeholder fixtures meanwhile.

---

## 7. Reversible defaults (D34) & verification items

Reversible (decide at code time, documented in code): collate fn decomposition;
mix-up disabled under (future) multi-stream; `replicate` vs `sampler` mixture;
per-point vs per-event materialization layout; the exact prefix-match token rule;
manifest-cache key/dir. **Verify before coding:** WAND per-event attr names
(`n_hits`/`source_event_idx`/`per_interaction`) on one shard; whether JAXTPC needs a
writer-side `n_hits`. **Risks:** `min_points` `>=` vs colleague's `>` (intentional,
parity diff at the boundary); `target_mask` has **no producer** (hmae config↔
`HMAECollate` drift — flag for the hmae owner, don't invent one); fill-value leakage
for empty `min`/`max` voxels; `RelativeLogNormalize` non-idempotent ordering.

---

## 8. Future extensions (deferred, designed-for)

- **Multi-stream-in-batch (D23/D24):** namespaced collate `{stream:{coord,feat,offset}}`
  + per-event-resolved cross-stream joins + model-side primary-stream→`Point` adapter.
  Additive; reuses the nested dataset + decoration. Build when a task consumes >1
  stream per forward.
- **`AggregateBySensor` (D32):** registered transform on `hits` (sum PE / min-or-pe-
  weighted t); built-but-unused until a PMT-merged-input task appears.
- **Track B — densify + noise (D1):** wire-TPC-only; lives on the JAXTPC track
  (`gpu_batch_transforms_plan.md`, retired 2026-09-13); sequenced after the base + readers land.
- **Per-interaction event unit (D36):** opt-in split mode for GENIE/pile-up;
  `per_interaction` already surfaced (§3.4) so it's additive.

---

## 9. Resolutions from final questions (D39–D41)

**9.1 Multi-stream — design-for-extension, NOT build-now (D39).** Single-stream
is the build; multi-stream is deferred because no concrete consumer is specified
and the cross-stream collate is the highest-risk piece. **Lock these four seams now**
so the future add is a bounded additive change (a new collate path + a model-side
primary/aux adapter), never a rewrite:
1. **Nested dataset output** — never flatten in the dataset (§3.3); a second stream
   is just present.
2. **Per-event label decoration** (§3.5) — each stream self-contained; cross-stream
   joins resolved before batching, so there is *never* a join-at-collate.
3. **Stream-aware collate structure** — the single-stream collate (§3.6) is written
   to operate on a *named* stream, so adding streams is a loop, not a rewrite.
4. **Stable `event_identity`** (§3.3) — future multi-stream batches align streams by
   event via this key.
Flips to build-now only if a concrete multi-stream model is committed this cycle.

**9.2 JAXTPC is in scope from the start (D40).** The base, readers, label decoration,
and the `jaxtpc_seg` migration all cover JAXTPC in this build (not LUCiD-first).
JAXTPC-specific build items: `volume` stays an **orthogonal sub-selector** applied in
`_build_readers` (not the mixture/holdout axis); sensor `n_hits` computed via
Σ`g.attrs['n_pixels']` over plane groups (cheap attr-walk) with a **writer-side ask**
to stamp a per-event `n_hits` (and per-file vector) for O(1); `source_event_idx` is in
JAXTPC files (`save.py:344/391/625`); `track_interaction` is already surfaced by the
labl reader. Holdout/identity at the **event** level (one readout = one sample, D36);
volume does not subdivide the holdout. The parity test matrix (§6) runs both LUCiD and
JAXTPC fixtures.

**9.3 Eval reproducibility baked in (D41).** The eval/probe path must be a faithful
replay of training data conditions:
- **Per-run record**: persist the holdout spec (`seed`, fractions or `n_per_config`,
  identity scheme) and the resolved transform pipeline (config-hashed) next to the
  experiment (alongside `config.py` in the `train.sh` snapshot, §5).
- **train≡eval transforms**: the val/probe transform pipeline must be the *same
  registered transforms* with the same params as pretraining (no re-specified or
  drifted GridSample/normalize); enforce by construction (shared `base_*_transform`
  config fragment) and assert in the test matrix (§6) that the val pipeline's
  registered transform list matches the train pipeline's deterministic subset.
- The holdout is selected by the same hash (§3.3), so train/val/test are provably
  disjoint and stable; the probe's leakage guard (§3.7) verifies disjointness at
  runtime via `event_identity`.
