# pimm-data data layer — Design & Decisions (authoritative)

> **Boundary note (2026-09-13).** Where this document places the LArTPC forward
> model — `AddNoise`, `Digitize`, the noise spectrum, `add_intrinsic_noise` — in
> pimm-data, that is no longer true. Those moved to **helix** (pimm-data
> `b592600` / helix `8075a53`): pimm-data keeps `densify`, which every detector
> family needs, and gave up the physics only one family applies. The transform
> registry raises on duplicate registration, so the two packages are lockstep.
> See helix `docs/ARCHITECTURE.md` §3. The design reasoning below is kept as
> written; only the ownership changed.

Status: **live design**, canonical "what & why" for the pimm-data data layer. This
consolidates the engagement record (decision log D1–D48) and the implementation
series (`implementation_plan_pimm_data_datalayer.md` + `impl/00..07`) into one
reference. It presents the **live** design only.

- **Spine of truth for decisions:** `engagement_plan_transform_dataset_placement.md`
  Part VIII (D1–D48). This doc defers to it; §8 below indexes every decision with
  its live/superseded/future status so nothing is lost.
- **Executable detail:** the `impl/0[0-7]_*.md` part specs (file-by-file, code-grounded).
- **Repos:** pimm-data `/sdf/group/neutrino/omara/pimm-data` (the data layer);
  pimm (Pointcept fork, branch `research`) keeps trainer/DDP/model/hooks; JAXTPC the
  simulator; WAND the water-Cherenkov LUCiD dataset.

> **Superseded — do not propagate.** Two early decisions were superseded and are
> mentioned here once only: **D3 → D6** (`LUCiDEventSSLDataset` dissolves into the
> base instead of being a standalone selection-cap class), and **D4 → D23** (canonical
> nested output + `Collect(stream=)` was reframed; collate is namespaced in the
> *future* path, and that future was then itself downgraded — see §7). Neither superseded
> form is part of the live design below.

---

## 1. Goal & principles

Make **pimm-data own the entire data layer** — datasets, transforms, collate,
readers, registries — cleanly enough to drive **broad multi-task** work (SSL/pretrain,
semantic seg, instance/panoptic, vertex/energy/PID/containment) on **both** detectors:
water-Cherenkov (LUCiD/WAND) and wire-TPC (JAXTPC). pimm keeps only trainer / DDP /
model / hook code. The near-term build is **single-stream-per-task** (D35): the dataset
is multi-modal and emits a nested per-stream dict; each task picks **one** stream and
runs it through the existing single-stream collate.

**Placement principle — put the right thing in the right place** (D-scope, Part II):

| Layer | Owns | Examples |
|---|---|---|
| **Dataset** | *which/whether an event exists* + identity + label sourcing | selection, holdout, min-points, source mixture, label decoration from `labl` |
| **Transform** | per-event / per-stream data manipulation | normalize, voxelize, augment, encode labels (`RemapSegment`/`InstanceParser`) |
| **Collate** | packing samples into a batch | single-stream `collate_fn` (offset rebasing, `_`-prefix drop) |
| **Driver (pimm)** | when/where post-collate stages run + RNG | DDP sampler, model forward, hooks/evaluators |

Everything below is a consequence of keeping each capability in its layer.

---

## 2. Architecture overview

**Ownership boundary (D17/D18):** all of the data layer moves to pimm-data; pimm keeps
`MultiDatasetDataloader` (DDP), the model/hook registries, hooks/evaluators, and a thin
re-export/re-register shim in `pimm/datasets/__init__.py`. Config `type=` strings resolve
against **pimm's** registries, so pimm-data classes are *re-registered* into them by the
shim (the two repos keep separate `Registry` objects; class objects are copied by name —
never a shared registry).

```
                        pimm (Pointcept fork, branch research)
   ┌───────────────────────────────────────────────────────────────────┐
   │  Trainer / DDP (MultiDatasetDataloader)   Model registry            │
   │  Hooks / evaluators (lucid_event_probe)   thin __init__.py shim ────┼──┐
   └───────────────────────────────────────────────────────────────────┘  │ re-register
                                                                           │ into pimm's
   pimm-data  (owns the data layer)                                        │ DATASETS / TRANSFORMS
   ┌───────────────────────────────────────────────────────────────────┐  │
   │  MultiModalEventDataset (base)  ── selection / identity / holdout   │◄─┘
   │      ├── LUCiDDataset            (PMT readers, particle-idx FKs)    │
   │      └── JAXTPCDataset           (wire/pixel readers, track FKs)    │
   │  readers/  (8: lucid|jaxtpc × sensor|step|hits|labl)               │
   │  transform.py  detector_transforms.py (ApplyToStream/RemapSegment) │
   │  collate.py (single-stream, byte-identical REPLACE)                │
   │  label decoration (label_config + generic decorator)               │
   └───────────────────────────────────────────────────────────────────┘
```

**De-fork mechanics (D33, rollout runbook):** build the new code *behind* the existing
vendored datasets in pimm-data → prove parity/determinism on synthetic fixtures (Step 0
gate) → drop the re-export shim → flip transforms+PILArNet → flip JAXTPC+LUCiD configs →
delete vendored files last (single `git revert` restores). Invariant at every step:
**existing PILArNet/panda/hmae training still works.** Packaging: pimm-data is a git
submodule under `pimm/libs/pimm-data` (editable install), its SHA snapshotted by
`train.sh` for reproducibility; torch pinned to the env. Details: `impl/06`.

---

## 3. Dataset layer — `MultiModalEventDataset` base

The keystone. `MultiModalEventDataset` (new `src/pimm_data/multimodal.py`) is the single
owner of **which events exist and which split they belong to** (D6). `LUCiDDataset` and
`JAXTPCDataset` inherit it; `LUCiDEventSSLDataset` dissolves into base + config (D6,
ex-D3). It inherits `DefaultDataset`'s test-mode TTA path via a factored `TestModeMixin`
(D30), so seg-eval fragment/`inverse` test-time augmentation keeps working and the npy
PILArNet path stays byte-identical.

**Common (base) vs different (subclass)** (D37):

- **Common (base):** source mixture, hash-identity holdout, min-points, nested output,
  the label-decoration framework, `event_identity`/`split`, `get_data` dispatch,
  `event_label`/`config_id` materialization, the source-prefixed `get_data_name`.
- **Different (subclass):** readers (PMT vs wire/pixel), geometry, FK chains
  (`particle_idx→category` vs `group_id→track→label`), and detector sub-selectors.
  **JAXTPC `volume` is an orthogonal sub-selector, NOT the mixture/holdout axis** — two
  `volume=` views of the same files map to the *same* `(config_id, source_event_idx)` and
  the *same* split (D37/D40/D44).

**`__init__` order is load-bearing — filter-then-hash-split (D8):** normalize sources →
build readers (subclass hook, applies `volume` etc.) → build/load manifest cache →
min-points filter (`>=`) → hash holdout → `max_events` cap → mixture → standard tail.
Filtering before splitting means the three splits partition the *surviving* population.

### 3.1 Source mixture (D9 / D45 — three distinct axes, do not collapse)

`sources=` is the mixture axis: `str | [str|dict{root,label,config_id?,split?,weight?}]`.
Each source gets a stable integer `config_id` (assigned by the base, never a reader
field), an `event_label` (probe target), and a `weight`. Mixture default is `replicate`
(integer copies by weight; rank-deterministic, no sampler); `sampler` mode records
per-entry weights for a pimm-side `WeightedRandomSampler`.

**Three axes that must stay distinct (D45):**

1. **`sources=`** — the D9 mixture: labeled, config-stratified, the holdout/identity axis.
2. **multi-run / shard-union** — same physics, one label, run-as-identity (e.g. doraemon's
   four `run_*` dirs are **one** logical dataset, *not* four sources). A within-config
   multi-run + shard-tag sub-selector is a **Phase B** gap (see §7), not built now.
3. **shard-tag** — an orthogonal within-run sub-selector, like JAXTPC `volume`.

### 3.2 Hash-identity holdout (D7 / D26)

Deterministic 3-way (`train`/`val`/`test`) holdout keyed on **stable identity**
`(config_id, source_event_idx)` — never the dense `local_idx`, which shifts when shards
are reordered/added/removed.

- `bucket = blake2b(struct.pack('<qqq', seed, config_id, source_event_idx)) / 2**64`,
  thresholded by `fractions`. **Config-stratified by folding `config_id` into the digest**
  — each config independently gets ~the same split, with no per-config bookkeeping.
- blake2b (stdlib) is process-, NumPy/Python-version-, and machine-stable (fixed byte
  order); this **replaces** the colleague's `np.random.permutation`, which is not
  rank/version-stable.
- `n_per_config` mode: take the `k` smallest-`u` events per config (exact count).
- **Fallback** when `source_event_idx` is absent: identity `(config_id, positional)` +
  one warning per source. Deterministic given a fixed shard set, but degrades — reorder
  then changes membership (documented).

`event_identity(idx) -> (config_id, source_event_idx)` is modality-independent, public,
survives `Subset` wrapping, and is the exact surface the eval probe and the repro
contract read. The headline invariant: for fixed `(sources, seed, fractions, modalities,
min_points)` the identity set is identical across process restart, machine, DDP world
size, version, shard reorder, and shard add/remove (only adding/removing *events* changes
membership, and only for those events).

### 3.3 Min-points (D8) + manifest cache (D27)

Min-points is a cheap, attr-only `n_hits` threshold, inclusive `>=` (intentional
one-event parity diff vs the colleague's `>`), applied before the split. `n_hits` comes
from `read_meta(idx)` (§5 of the readers part), which reads **only `evt.attrs`** (and
per-file `config/` vectors) — never decodes arrays. Per-detector cheap proxies: LUCiD
sensor `n_hits` / step `n_segments` / hits `n_particle_hits`; JAXTPC sensor Σ`n_pixels`,
step/hits Σ`n_actual` (deposit count). `min_deposits`/`min_segments` are intercepted at
the base and routed through the joint-index min-points so they cannot desync (D44);
volume-aware min-points may scope the filter but never the holdout.

**Manifest cache (D27/D46):** the scanned `(source_event_idx, n_hits)` triples are
persisted per source as an `.npz`, keyed by package version + modalities + per-shard
(path,size,mtime). **Rank-0 builds it under a DDP barrier with an atomic write**
(`os.replace`); every rank reads the identical file → byte-identical `data_list` across
ranks. Steady state never reopens event groups. The A1 `_read_shard_meta` `lru_cache`
(dedup the cross-reader file opens during the rank-0 scan) is an impl detail under this
build, complementary to it.

### 3.4 Joint cross-modality index (D42 — the desync fix, LIVE and central)

The base **must build a joint cross-modality index**. This closes a real, latent bug the
naive base would inherit (handoff §4): a dataset that passes the **same global `idx` to
every reader** and sets `_n_events = min(len(r) …)` is only correct if every reader's
index is the *same* contiguous list of physics events. Two ways it breaks:

1. **`min_deposits>0` desync** — when only the step reader masks its index to a
   non-contiguous subset, `get_data(k)` reads step event `valid[k]` but sensor/hits/labl
   event `present[k]` — different physics events; `bridges`/`deposit_to_track`/
   `group_to_track` joins become meaningless. (Triggers the moment anyone uses it with
   >1 modality; no current test covers it.)
2. **Gap-induced desync** — once each reader indexes its own *present* event keys
   (gap-tolerant indexing, already landed), a gap in some-but-not-all modalities
   misaligns them silently.

**Fix (D42):** intersect the present `event_*` keys across the loaded modalities into one
canonical map; `_read_event` translates `local_idx` per modality; replace `min(...)` with
the **intersected** size; the manifest/identity are built over the intersected index.
`read_meta`'s `source_event_idx` is the join key (already surfaced). This makes holdout
and identity depend on alignment the base **enforces**, not on alignment it merely
assumes. Single-stream (D35) dodges desync only for single-modality unlabeled runs;
**label decoration re-exposes it** for every labeled task (stream-reader + labl-reader
joined at the same `local_idx`) — so the joint index is required, not optional.

**Phasing (D43/D47):** Phase A lands this as a **standalone bug-fix PR** on the loader
branch *before* the de-fork — A1 meta-cache, A2 joint index, A3 volume-aware+raise, A4
length-mismatch warn (+`strict_lengths`), A5 cross-modality regression test (fails on
HEAD today). The base then factors A2's joint index up. Identity correctly keys on
`source_event_idx` (not `local_idx`/`event_num`); no branch drift.

### 3.5 LUCiD vs JAXTPC subclasses

| | LUCiD | JAXTPC |
|---|---|---|
| Readers | PMT (`sensor/step/hits/labl`) | wire/pixel (`sensor/step/hits/labl`) |
| FK chain | `particle_idx → per_particle`; `track_idx → per_track → particle_idx` (positional gather) | `group_id → group_to_track → track_id`; `deposit_to_track → track_id` (value-keyed searchsorted) |
| `source_event_idx` | per-file `config/source_event_idx` **vector** (sensor+labl, O(1)) + per-event attr | per-event attr only (`save.py`); no config vector |
| `n_hits` source | per-event `n_hits`/`n_segments`/`n_particle_hits` attrs | Σ`n_pixels` (sensor) / Σ`n_actual` (step/hits) attr walk |
| Sub-selector | none | `volume` (orthogonal, §3) |

---

## 4. Streams & labels

**Nested per-stream dataset output (Seam 1 — never flatten in the dataset).**
`get_data(idx)` returns `{name, split, sensor:{…}, step:{…}, hits:{…}, labl:{…},
bridges:{…}, _targets:{…}}` — each modality a self-contained sub-dict with its own
`coord`/labels; never bare top-level `coord`/`segment`. (The legacy flat `resp/corr/seg`
prefixing is the anti-pattern being migrated away from.)

**Single-stream-per-task (D35/D36).** The event unit is configurable; default = one stored
detector readout (`event_NNN`) = one sample (per-interaction/pile-up splitting is an
opt-in future mode). Each task selects **one** stream via `Collect(stream=…)` → the
existing single-stream collate → `Point`. That stream carries its own per-point labels.

**Task → stream → label (Part X):**

| Task | Primary stream | Label (from `labl`) |
|---|---|---|
| SSL pretrain | `sensor` / `step` | none |
| Event PID / classification | `sensor`/`step` | `event_label` / `target` from interaction |
| Semantic seg | **`hits`** (unaggregated) / `step` | `segment_pid ← per_particle.category` |
| Instance / panoptic | **`hits`** / `step` | `segment_pid` + `instance_particle`/`instance_interaction`/`instance_ancestor` |
| Vertex / energy / containment regression | `sensor`/`step` | per-event `target_vertex`/`target_energy`/`target_contained` |

**Label decoration from `labl` (D20/D28/D38) — in the dataset, not a transform.** Readers
emit **raw FKs** (`particle_idx`/`track_idx`/`group_id` + chain tables); the dataset's
**generic decorator** `_decorate_from_labl(sub, labl, fk_resolver)` builds the named
schema keys via a declarative `label_config`. The `fk_resolver` is the *only* per-subclass
piece (positional gather for LUCiD; value-keyed searchsorted for JAXTPC) — the decorator
itself has zero detector branches. The framework is **open/extensible** (D38): a new axis
(edge/graph for NuGraph, hierarchy labels) is a new `label_config` spec + a prefix entry
in `index_operator`, not a decorator rewrite. This collapses the four hand-written
single-axis decorators that exist today into one.

**The named-key schema** (what configs/evaluators actually consume —
`panda`/`panseg` read `segment_pid`/`instance_particle`):

- **`scope="point"`** → per-point `(N,1)` column (`segment_pid`, `segment_interaction`,
  `instance_particle`, `instance_interaction`, `instance_ancestor`). Subset by N-changing
  transforms (kept aligned by the §5 prefix-match).
- **`scope="event"`** → per-event target (`target_vertex (3,)`, `target_energy`,
  `target_contained`), length-1/`(D,)`, carried as `_`-prefixed list-collated metadata,
  **NOT** subset.
- **`scope="event_broadcast"`** → per-event value materialized to a `(N,1)` per-point
  column (`event_label`, `config_id`) so `Collect` lifts it and the probe slices by
  offset. Owned by the base.

Bare `segment`/`instance` survive only as the back-compat single-axis alias (the JAXTPC
`label_key=` path), byte-identical to today. `fill=-1` (ignore-index) on any unresolved
FK; a missing labl column omits that one axis (never fabricated). `target_mask` has **no
producer** — do not invent one (hmae config↔`HMAECollate` drift; flag to the hmae owner).

---

## 5. Transforms

pimm-data's `transform.py` is brought up to (and slightly past) the colleague's
`research`-branch set. The merge delta (everything else is byte-identical to the branch;
**do NOT overwrite pimm-data's `Collect`** — it is ahead with `stream=`/autoconvert/
passthrough):

- **`RelativeLogNormalize`** (NET-NEW, D11/D31) — per-event PMT-time normalization with
  the **negative-time correctness step** (`x -= x.min()` before `log1p`; WAND `T` reaches
  ~−240 ns). Non-idempotent and order-sensitive → must run once, before any reorder/subset
  of the time column. The window params (`scale`/`max_val`) are a model/config choice
  (D13); only the subtract-min/clip correctness is fixed here (D31).
- **`GridSample` `{key: op}` reducers** (D29) — `sum/min/max/mean/first` with a
  `sum_keys`/`min_keys` back-compat shim (explicit `reducers` wins, and must byte-match
  the branch for the shimmed ops). `mean` promotes int→float; `first` is the deterministic
  hash-sorted representative (not the random survivor).
- **`LogTransform.clip`** (D11/D31) — optional pre-`log` domain clamp; default off (no
  regression).
- **`MultiViewGenerator.get_view` guard** (D11) — empty-cloud raise + size clamp.
- **v3 `vertex`/`is_primary` plumbing** (D11) — vertex co-transform on every geometric op,
  each guarded by `_valid_vertex_mask` so it is a **no-op until a v3 dataset stamps
  `vertex`** (`PointClip` stays vertex-blind by design). `MixedScaleGeometryMultiViewGenerator`
  ported verbatim (off the SSL critical path).

**`index_operator` prefix-match (D25 — the N-changing-safety keystone).** N-changing
transforms (`GridSample` train, `RandomDropout`, `SphereCrop`, `ShufflePoint`,
`CropBoundary`) subset only the keys in `index_valid_keys`. Any per-point column not in
that list silently desyncs (keeps the old N while `coord` shrinks), corrupting labels.
**Fix:** after the default list, append dict keys matching the **underscore-boundary**
prefixes `segment*`/`instance*`/`target*` (so `segment_pid` carries, `segmentation_meta`
does not) plus explicit `particle_idx`/`sensor_idx`/`plane_id`, and **exclude per-event
`target_*`** by a leading-dim≠`n_points` shape check. This makes the per-config
`Update(index_valid_keys=…)` no longer load-bearing for the standard decorated axes.

**`ApplyToStream` (D25).** Per-stream N-changing transforms run inside
`ApplyToStream(stream='step', […])`, which passes the stream sub-dict to the inner
`Compose` (so `index_operator` sees that stream's own `coord`/labels/`index_valid_keys`).
The list does not propagate across the wrapper boundary, so any explicit `Update` must sit
*inside* the same `ApplyToStream`.

---

## 6. Collate & reproducibility

**Single-stream collate is a byte-identical REPLACE (D35/D23-reverted).** Near-term collate
*is* the existing `collate_fn`/`point_collate_fn`/`inseg_collate_fn` (verified `diff`-clean
against pimm's `utils.py`). After `Collect(stream=…)` the per-sample dict is exactly the
single-cloud shape the existing collate already handles — the nested structure is gone by
collate time, so there is nothing stream-shaped left to pack. Touching collate now would
risk the Step-3 parity gate for zero functional gain. Two mechanisms it already provides
are reused by both the single-stream and future-multi-stream designs:

- **offset diff/cumsum rebasing** — each sample's `offset=[n_b]` → `.diff(prepend=0)` →
  concatenate counts → outer `cumsum` → a global batch `offset`; lets one flat point cloud
  carry batch boundaries.
- **`_`-prefix key drop** — any key starting with `_` is silently dropped at collate; this
  is how ragged per-event metadata (`labl`/`bridges`/per-event `target_*`) rides to the
  sample but never gets tensor-collated. `event_label`/`config_id` reach the batch as
  ordinary per-point columns.

**Eval / probe via `event_identity` (D41).** The `lucid_event_probe` hook is rewired off
dataset internals (`data_list`/`datasets`/`source_root`) onto the stable
`event_identity(idx)` + public `split` API: `_event_keys` becomes
`{event_identity(i) for i in indices}`, `_format_event_key` generalizes to an N-tuple,
`_dataset_split` is unchanged. The train/val disjointness guard then checks the *same*
stable identity the holdout hash splits on — disjoint by construction. Per-point
`event_label` (the `event_broadcast` materialization) is recovered per event by the
offset-slice branch.

**train ≡ eval (D41).** The val/probe pipeline must be the *same registered transforms with
the same params* as training's deterministic subset (NormalizeCoord, GridSample,
LogTransform, RelativeLogNormalize) — enforced by a shared `base_*_transform` config
fragment both splice in, asserted in the test matrix. Per run, persist the holdout spec
(seed + fractions/`n_per_config` + identity scheme) and the resolved transform pipeline
next to the experiment snapshot (alongside the submodule SHA), so the exact held-out set
and preprocessing are reconstructible.

---

## 7. What's deferred / future seams

**Multi-stream-in-batch is design-for-extension, NOT built (D39, downgrading
D14/D19/D23/D24).** No concrete multi-stream consumer is specified, and the
namespaced-collate + cross-stream-join is the highest-risk piece (coherent cross-stream
mix-up is a research problem). The design makes the future add a **bounded additive
change** (a new `multistream_collate_fn` + a model-side primary/aux adapter), never a
rewrite. **Four seams are locked now:**

1. **Nested dataset output** — never flatten in the dataset; a second stream is just present.
2. **Per-event label decoration** — each stream self-contained, cross-stream joins resolved
   *before* batching and carried `_`-prefixed; there is never a join-at-collate.
3. **Stream-aware collate structure** — the single-stream collate operates on a *named*
   stream, so the future path is a loop over streams (each with its own offset), not a rewrite.
4. **Stable `event_identity`** — future multi-stream batches align streams by event via this
   key.

Flips to build-now only if a concrete multi-stream model is committed this cycle.

**Track B — densify + noise (D1/D33).** Wire-TPC-only; re-homed to the JAXTPC track
(`gpu_batch_transforms_plan.md`, retired 2026-09-13), sequenced after the base + readers land. NOT part of the
LUCiD/WAND path. Two decoupled, device-agnostic stages: `Densify` (sparse→dense scatter,
per-plane, CPU=GPU code) and a separate sensor-only noise step faithful to JAXTPC
`tools/noise.py`. Stored sensor is signal-only; nothing is converted between `sensor` and
`hits`. Noted, not detailed here.

**Aggregation (`AggregateBySensor`, D32).** Built-but-unused, default off; seg uses `hits`
unaggregated (D21). Revisit when a PMT-merged-input task appears.

**Per-interaction event unit (D36).** Opt-in split mode for GENIE/pile-up; `per_interaction`
is surfaced so it is additive.

**Shard/event filtering — Phase B/C (D45/D47/D48).** The within-config multi-run +
shard-tag sub-selector (handoff B1/B2), `event_filter` registered-class form (B3), and
`manifest=` include/exclude curation contract + overflow-CSV + CLI tooling (B4/C) are
deferred. Distinct from the internal `.npz` manifest cache (machine-generated, for speed).
The *snapshot* half of the old `resolved_manifest` is superseded by D41's per-run record;
the include/exclude *curation contract* is a real gap scoped to Phase B. **Open (D48, owed
by user):** how far to go now (Phase A only / A+B1 / A+full B / +CLI) and the multi-run
mechanism (`runs=` one dataset, recommended, now a no-structural-change `sources=`
extension / ConcatDataset / one-run-at-a-time). Phase A proceeds regardless.

---

## 8. Decision index (D1–D48)

Status legend: **live** (part of the current design), **superseded** (replaced; do not
present as current), **future** (locked seam / deferred, not built now), **open** (owed).

| ID | One-line | Status |
|----|----------|--------|
| D1 | LUCiD fully sparse; densify/noise (Track B) out of scope for LUCiD, wire-TPC-only | future (Track B) |
| D2 | Build broad / multi-task from day one; carry streams + labels | live |
| D3 | Selection caps in shared base; `LUCiDEventSSLDataset` dissolves | **superseded → D6** |
| D4 | Canonical nested output + `Collect(stream=)`; no in-dataset flattening | **superseded → D23** |
| D5 | De-fork mechanical bucket in parallel | live (sequencing → D33) |
| D6 | New `MultiModalEventDataset` base owns selection + nested output; LUCiD/JAXTPC inherit | live |
| D7 | Holdout = hash on stable `(config_id, source_event_idx)`; 3-way; config-stratified | live |
| D8 | min-points via cheap `n_hits`; inclusive `>=`; filter-then-hash-split | live |
| D9 | Config-mixture native: `sources=[{root,label,weight}]`, explicit labels | live |
| D10 | Surface ALL stored streams/labels now (sensor `n_hits`; hits `T_reco`; labl `per_interaction`) | live |
| D11 | Upstream branch transforms (GridSample reducers, `RelativeLogNormalize`, `LogTransform.clip`, `get_view`, v3 vertex) | live |
| D12 | Aggregation = registered transform on `hits`, not `sensor` | live placement (→ deferred build D32) |
| D13 | Time window / `RelativeLogNormalize` params = model/config choice | live (correctness split → D31) |
| D14 | Multi-stream collation REQUIRED | **future** (downgraded by D35/D39) |
| D15 | Support all eval tasks (vertex/energy/PID/seg/containment); extensibility first-class | live |
| D16 | Build the segmentation path now (`hits` + labels + `InstanceParser`) | live (aggregation struck → D32) |
| D17 | No early implementation; de-fork replaces MOST of `pimm/datasets/` | live (early-exit → D33) |
| D18 | De-fork boundary: data layer → pimm-data; pimm keeps dataloader/registries/hooks/shim | live |
| D19 | Multi-stream batch namespaced/nested; model names primary stream | **future** (→ D23, downgraded) |
| D20 | Labeling in reader/dataset from `labl` (not a transform) | live (split → D28) |
| D21 | Seg input task-dependent; LUCiD per-point seg → `hits` unaggregated; 3D → `step` | live |
| D22 | Generic `segment_*`/`instance_*`/`target_*` schema | live (names reconciled → D28) |
| D23 | Flattening topology: namespaced `multistream_collate_fn`, model-side flatten; `Collect(stream=)` demoted | **future** (downgraded by D35/D39) |
| D24 | Cross-stream joins resolved per-event in reader/dataset; ragged carried `_`-prefixed | **future** seam (live as Seam 2 contract) |
| D25 | N-changing safety: `index_operator` prefix-match + per-stream `ApplyToStream` | live |
| D26 | Holdout specifics: blake2b, config-stratified, fallback+warning; `event_identity` in base | live |
| D27 | Reader surfacing + persisted manifest cache (rank-0 build under DDP barrier) | live |
| D28 | Reader emits raw FKs → dataset decorates via `label_config`; per-event targets per-event; named keys | live |
| D29 | GridSample `{key:op}` reducers (sum/min/max/mean/first) + back-compat shim | live |
| D30 | Base inherits `DefaultDataset` via factored `TestModeMixin` | live |
| D31 | Negative-time policy = transform-correctness requirement (subtract-min/clip) | live |
| D32 | Aggregation deferred; `AggregateBySensor` built-but-unused (default off) | future (deferred) |
| D33 | Rollout: build-behind → parity matrix → shim → flip → delete vendored; Track B re-homed | live |
| D34 | Rounds settle STRUCTURE; reversible impl details deferred with documented defaults | live |
| D35 | Single-stream-per-task is the near-term structure (supersedes D14; downgrades D19/D23) | live |
| D36 | Event unit configurable; default one stored readout = one sample; per-interaction opt-in | live |
| D37 | LUCiD/JAXTPC common (base) vs different (subclass); `volume` orthogonal, not mixture axis | live |
| D38 | Open/extensible decoration framework (axes = `label_config` entries, not hardcoded) | live |
| D39 | Multi-stream: design-for-extension; build single-stream now, lock the four seams | live (the seams) / future (the build) |
| D40 | JAXTPC in scope from the start (base + readers + decoration + `jaxtpc_seg` migration) | live |
| D41 | Eval reproducibility: per-run holdout+pipeline record; train≡eval transforms enforced | live |
| D42 | Cross-modality desync is real; base builds a JOINT cross-modality index (identity on `source_event_idx`) | live (central) |
| D43 | Phase A lands first as a standalone bug-fix PR before the de-fork (A1–A5) | live (sequencing) |
| D44 | Intercept `min_deposits`/`min_segments` at the base on the joint index; volume-aware min-points | live |
| D45 | Three distinct axes: `sources=` ≠ multi-run/shard-union ≠ shard-tag; doraemon runs are not 4 sources | live |
| D46 | A1 `_read_shard_meta` lru_cache adopted under the manifest-cache build; no branch drift | live |
| D47 | Manifest-as-INPUT (include/exclude) + overflow-CSV + CLI = Phase B/C, deferred | future (Phase B/C) |
| D48 | OWED BY USER: how far now (Phase A / +B1 / +B / +CLI); multi-run mechanism | **open** |
