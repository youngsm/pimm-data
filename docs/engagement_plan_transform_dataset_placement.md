# Engagement & Working Plan — pimm-data ↔ pimm: right things in the right places (v2)

Status: **living document, v2** (rewritten after the Round-4 4-agent cadence
review). Append-only **decision log (Part VIII)** is the spine of truth: D1–D22
verbatim from v1; D23–D33 added by the review; superseded rows annotated, never
deleted. Companions: `gpu_batch_transforms_plan.md` and `_handoff.md` — both retired
2026-09-13; the forward model they place in pimm-data is helix's now.

Repos / data:
- `pimm-data`: `/sdf/group/neutrino/omara/pimm-data`
- `pimm` (Pointcept fork, branch `research`): `/sdf/home/o/omara/.claude/jobs/21ffc656/particle-imaging-models`
- `JAXTPC`: `/sdf/group/neutrino/omara/JAXTPC`
- **WAND** (water-Cherenkov LUCiD): `/sdf/data/neutrino/cjesus/DORAEMON/WAND/SK_like`

---

## Part 0 — Converged design (TL;DR)

**Goal:** make pimm-data own the entire data layer (datasets, transforms,
collate, readers, registries) cleanly enough to drive **broad multi-task**
work (SSL, semseg, panoptic/instance, vertex/energy/PID/containment) on **both**
water-Cherenkov (LUCiD/WAND) and wire-TPC (JAXTPC), with pimm keeping only
trainer/DDP/model/hook code.

The architecture:
- **`MultiModalEventDataset` base** (pimm-data) owns event **selection**:
  multi-source mixture, hash-on-stable-identity holdout, cheap-`n_hits`
  min-points, `event_identity(idx)`/`split` API. `LUCiDDataset`/`JAXTPCDataset`
  inherit it; `LUCiDEventSSLDataset` dissolves into base + config.
- **Nested per-stream dataset output** `{sensor, hits, step, labl, …}`.
- **Namespaced multi-stream collate** (NET-NEW): `{stream: {coord, feat,
  offset, labels…}}`, each stream its own offset. Primary-stream → flat `Point`
  happens **model-side**. `Collect(stream=)` is demoted to legacy single-stream.
- **Labeling in the reader/dataset from `labl`** (not transforms): readers emit
  raw FKs; the dataset's generic decorator builds `segment_<name>`/
  `instance_<name>`/`target_<name>` via a `label_config` map. Cross-stream
  joins are resolved **per-event before batching**.
- **LUCiD is fully sparse** (CPU per-sample transforms). Densify/noise (Track B)
  is **wire-TPC-only**, re-homed to the JAXTPC track, deferred.

---

## Part I — Process (re-scoped by the review)

Slow, question-driven, agent-checked, colleague-input first-class. **Decision
log is immutable**; the narrative is rewritten each cycle. **Convergence, not a
100-question quota, terminates the rounds** (22 decisions closed in 4 rounds;
the substrate already exists, so the remaining surface is small and concrete).

Re-scoped remaining rounds (was "5–20 generic"; now ~5 themed):
- **R5 — Namespaced multi-stream collate mechanics + cross-stream joins** (the keystone, D23–D25).
- **R6 — `MultiModalEventDataset` API + holdout/identity/min-points specifics** (D26–D27, D30).
- **R7 — Label decoration + multi-task schema + N-changing safety** (D25, D28–D29).
- **R8 — JAXTPC parity** (multi-volume axis, missing `n_hits` attr, identity field; Track B re-home).
- **R9 — Eval/repro contract, testing matrix, DDP scale, rollout order** (D33).
- **Early-exit gate:** once the collate contract (R5) and base API (R6) are pinned and the test matrix (R9) agreed, cut to implementation. 4-agent review after the re-scoped rounds → v3; final synthesis → implementation doc.

Cadence: 4-agent review after every theme block; doc version bumps with each.
Colleague's `research` branch is fetched/checked out; transform **parity tests**
gate on his reference outputs, the **design** does not.

---

## Part II — Scope & guiding principle

**Put the right things in the right places.** Dataset = *which/whether an event
exists* + identity + label sourcing (selection, holdout, min-points, mixture,
label decoration from `labl`). Transform = per-event / per-stream data
manipulation (normalize, voxelize, augment, encode labels via `InstanceParser`).
Collate = packing into the namespaced multi-stream batch. Driver (pimm) =
when/where post-collate stages run + RNG.

**Track B (densify + noise) is wire-TPC-only and re-homed to the JAXTPC track**
(D1, D33); it is NOT part of the LUCiD/WAND path and is sequenced after the base
+ collate land. The (retired) `gpu_batch_transforms_plan.md` v3 "near-term consumer"
framing predates D1 and is superseded for LUCiD.

---

## Part III — Colleague input (requirements; reference on `research` branch)

Asks: (1) registered PMT-hit aggregation transform [→ `hits`, deferred per D32];
(2) deterministic holdout (file/event) [→ D7/D26]; (3) min-points filter in the
Dataset [→ D8/D27]; (4) config-mixture selection [→ D9]. He built a working
`LUCiDEventSSLDataset` (sensor-only, flat, positional holdout, sensor-aggregation
no-op) + a Sonata SSL config + transforms (`GridSample.min_keys`,
`RelativeLogNormalize`, `LogTransform.clip`, `get_view` guard, v3 vertex
plumbing) + an `EventLinearProbeEvaluator`. All reconciled below. His code is
"not optimized for readability/extendability — there are better ways."

---

## Part IV — Placement decisions (regenerated from the log)

| Capability | Layer | Decision |
|---|---|---|
| PMT-hit aggregation | Transform (on `hits`) | D12 → **deferred, built-unused** (D32); seg uses unaggregated |
| Deterministic holdout | Dataset base | **hash on `(config_id, source_event_idx)`** (D7→D26) |
| min-points filter | Dataset base | **cheap `n_hits` + manifest cache** (D8→D27) |
| Config-mixture | Dataset base | **native multi-source, explicit labels, weights** (D9) |
| Label decoration | reader emits FKs → dataset decorates | **from `labl`, `label_config` map** (D20→D28) |
| Multi-stream collate | pimm-data collate | **namespaced, net-new** (D14/D19→D23) |
| Cross-stream joins | reader/dataset (per-event) | **resolve before batching** (D24) |
| Primary-stream→`Point` | model-side | **adapter** (D19→D23) |
| Densify / AddIntrinsicNoise | — | **deferred, JAXTPC-only** (D1/D33) |
| `LUCiDEventSSLDataset` | — | **dissolve → base + config** (D3→D6) |

---

## Part V — Round log

| Round | Theme | Status |
|---|---|---|
| R1 | Scope & process (LUCiD sparse, broad, base, nested, de-fork) | closed → D1–D5 |
| R2 | Dataset-layer design (base, holdout, min-points, mixture, streams, transforms) | closed → D6–D12 |
| R3 | Time/multistream/seg/de-fork-scope | closed → D13–D18 |
| R4 | Multi-stream shape, label join, seg stream, label schema, aggregation | closed → D19–D22 |
| R4-review | 4-agent cadence → **v2** | closed → D23–D33 |
| R5 | Collate mechanics + cross-stream joins | open |
| R6 | Base API + holdout/identity/min-points | open |
| R7 | Label decoration + multi-task schema + N-changing | open |
| R8 | JAXTPC parity + Track B re-home | open |
| R9 | Eval/repro + testing + DDP + rollout | open |

---

## Part VI — WAND findings (scan)

18 configs (`config_000001`…`000018`), full 4-modality layout (`sensor/step/
hits/labl`), `wc_*` prefix, **~464k events** (symlinks into a ~hundreds-of-GB
tree). **sensor** (`sensor_idx`/`PE`/`T`, 10,764 PMTs, `sensor_positions`) is
**PMT-unique per event**; **hits** is per-particle (multi-hit per PMT). Every
event group has cheap **`n_hits`/`n_segments`/`n_particle_hits`** attrs and a
**`source_event_idx`** — but **no per-file count vector** and the readers don't
surface them. Cylinder, z half-height 18.1 (= `coord_scale`). Configs: 1=mu-,
3=e-, …, 13–14 GENIE ν, 15–18 pile-up. FLAGS: `time_log_max=4000` truncates the
muon-decay/Michel tail (config_1 → ~88 µs; 0.54% > 4000 ns); negative T to
~−240 ns; reader docstrings say v3 but files are v5; reader drops
`per_interaction`/`group_id`/`sensor_hits`/`T_reco`/`n_hits`.

## Part VII — Seed review (5 agents, pre-R1) — historical pointers

SR1 colleague config can't run on stock pimm-data (`min_keys`/`RelativeLogNormalize`
missing); SR2 aggregation no-op on sensor; SR3 densify/noise wire-only; SR4 time
truncation + negative-log; SR5 holdout determinism + `get_data_name` collision +
cheap `n_hits`; SR6 flat/nested; SR7 `per_interaction` dropped; SR8 process
(decision log, early-exit, de-fork ungated, colleague branch). All carried into
decisions below.

---

## Part VIII — Decision log (append-only; spine of truth)

| ID | Decision | Round | Status |
|----|----------|-------|--------|
| D1 | LUCiD fully sparse; Densify/AddIntrinsicNoise out of scope for LUCiD (wire-TPC-only, deferred); LUCiD ops on CPU. | R1 | decided |
| D2 | Build broad / multi-task from day one; carry streams + labels for future tasks. | R1 | decided |
| D3 | Selection caps in pimm-data shared base; `LUCiDEventSSLDataset` → dissolve. | R1 | **superseded → D6** |
| D4 | Canonical output nested + `Collect(stream=)`; no in-dataset flattening. | R1 | **superseded → D23** (collate now namespaces; `Collect(stream=)` demoted) |
| D5 | De-fork mechanical bucket in parallel. | R1 | decided (sequencing refined → D33) |
| D6 | New `MultiModalEventDataset` base owns selection + nested output; LUCiD/JAXTPC inherit. | R2 | decided |
| D7 | Holdout = hash on stable identity (`config_id`:`source_event_idx`); 3-way; config-stratified; file+event level. | R2 | decided (specifics → D26) |
| D8 | min-points via cheap per-event `n_hits`; inclusive `>=`; configurable modality; filter-then-hash-split. | R2 | decided (cache → D27) |
| D9 | Config-mixture native in base: `sources=[{root,label,weight}]`, explicit `{config:label}`, `config_id`/`event_label` top-level. | R2 | decided |
| D10 | Surface ALL stored streams/labels now (sensor+`n_hits`; hits `particle_idx`/`T_reco`; step `group_id`/`sensor_hits`; labl incl. `per_interaction`). | R2 | decided (+`source_event_idx`/`config_id` → D27) |
| D11 | Upstream his transforms: GridSample reducers, `RelativeLogNormalize`, `LogTransform.clip`, `get_view` guard. | R2 | decided (reducer API → D29; only `min_keys` exists to port) |
| D12 | Aggregation = registered transform on `hits`, not `sensor`. | R2 | decided (→ deferred, D32) |
| D13 | Time window / `RelativeLogNormalize` params = model/config choice, not dataset-layer. | R3 | decided (negative-time correctness split out → D31) |
| D14 | Multi-stream collation REQUIRED (multi-modality ⇒ multi-stream batches). | R3 | decided (mechanics → D23) |
| D15 | Support all eval tasks (vertex/energy/PID/seg/containment); extensibility first-class. | R3 | decided |
| D16 | Build the segmentation path now (`hits` stream + labels + `InstanceParser`). | R3 | decided (aggregation struck → D32) |
| D17 | No early implementation; de-fork replaces MOST of `pimm/datasets/`. | R3 | decided (early-exit gate added → D33) |
| D18 | De-fork boundary: all of `pimm/datasets/` → pimm-data; stays in pimm: `MultiDatasetDataloader`, model/hook registries, hooks/evaluators, thin `__init__` shim. | R3 | decided (collate is GAIN not REPLACE → D23; corrections in Part IX) |
| D19 | Multi-stream batch namespaced/nested post-collate; model names primary stream. | R4 | decided (→ D23) |
| D20 | Labeling in reader/dataset from `labl` (not a transform). | R4 | decided (reader-FK vs dataset-decorate split → D28) |
| D21 | Seg input task-dependent; LUCiD per-point seg → `hits` **unaggregated**; 3D → `step`. | R4 | decided |
| D22 | Generic `segment_<name>`/`instance_<name>`/`target_<name>` schema. | R4 | decided (names reconciled to real configs → D28) |
| D23 | **Flattening topology (supersedes D4):** dataset emits nested; **collate produces namespaced multi-stream batch** (`{stream:{coord,feat,offset,…}}`, per-stream offset) — **net-new `multistream_collate_fn`**, not byte-identical; **primary-stream→`Point` flattening is model-side** (multi-stream segmentor adapter); `Collect(stream=)` demoted to legacy single-stream convenience; mix-up disabled under multi-stream for v2. | R4-rev | decided |
| D24 | **Cross-stream joins resolved per-event in reader/dataset** (Option C): every join producing a per-point value computed before batching; ragged `labl`/`bridges`/per-event targets carried as `_`-prefixed list-collated metadata; live offset-shifted cross-stream indices deferred until a task needs them. | R4-rev | decided |
| D25 | **N-changing safety:** `index_operator` carries the D22 schema via **prefix-match** (`segment*`/`instance*`/`target*`) + `particle_idx`/`sensor_idx`/`plane_id` + explicit-append; N-changing transforms (GridSample…) run **per-stream via `ApplyToStream`** after joins resolved. | R4-rev | decided |
| D26 | **Holdout specifics (refines D7):** seeded **blake2b** of `config_id:source_event_idx`, config-stratified by including `config_id`; 3-way buckets; fallback `(file, positional)` + warning when `source_event_idx` absent. `event_identity(idx)` owned by the **base**, modality-independent; public `split` survives `Subset` wrapping. | R4-rev | decided |
| D27 | **Reader surfacing + cost (extends D10/D8):** add `source_event_idx` + `config_id` + per-event `n_hits`; request a per-file `n_hits` **vector** from writers; min-points/index via a **persisted manifest cache** (rank-0 build under DDP barrier), never array reads in steady state. | R4-rev | decided |
| D28 | **Label decoration (refines D20/D22):** reader emits **raw FKs** (`particle_idx`/`track_idx`/`group_id` + FK chains); dataset's generic decorator builds keys via a `label_config` map. **Per-event targets (vertex/energy/contained) are per-event, not per-point;** `event_label`/`config_id` materialized as **per-point arrays** in the primary stream (probe slices by offset). Names reconciled to real configs: `segment_pid`/`segment_interaction`, `instance_particle`/`instance_interaction`, `target_energy`/`target_vertex`/`target_contained`; `category→segment_pid`; `interaction` is BOTH a semantic and instance axis. | R4-rev | decided |
| D29 | **GridSample reducers (refines D11):** `{key: op}` map (sum/min/max/mean/first) with `sum_keys`/`min_keys` back-compat shim; only `min_keys` exists on the branch (max/mean/first net-new); define `first`/`mean`/fill-value semantics. | R4-rev | decided |
| D30 | **Base vs `DefaultDataset` (resolves Risk Re):** `MultiModalEventDataset` **inherits `DefaultDataset` via a factored `TestModeMixin`** (seg eval needs the fragment/`inverse` TTA path); npy `DefaultDataset` → thin subclass. | R4-rev | decided |
| D31 | **Negative-time policy (splits from D13):** a **transform-correctness requirement** (not a model knob): `RelativeLogNormalize`/`LogTransform.clip` define the pre-`log` offset/clip so the domain is valid. D13 keeps only the fidelity/window choice. | R4-rev | decided |
| D32 | **Aggregation deferred (amends D12/D16):** seg uses `hits` unaggregated (D21); `AggregateBySensor` built-but-unused (default off), struck from D16 "build now"; revisit when a PMT-merged-input task appears. | R4-rev | decided |
| D33 | **Rollout + process:** build base+collate+reducers **behind** existing datasets → parity/determinism **test matrix** on synthetic fixtures → re-export shim → **then** delete vendored (gate: grep `seg/resp/corr`, migrate `jaxtpc_seg.py`+semseg child first, fix `__init__.py:9` stale import). Re-scope rounds 5–9 (Part I); early-exit once collate+base pinned. Track B re-homed to JAXTPC track. | R4-rev | decided |
| D34 | **Rounds address only STRUCTURAL / hard-to-reverse questions** (what forces the shape of the architecture). **Reversible implementation details** — collate fn signature, mix-up handling, per-point vs per-event materialization layout, RNG/hash specifics, function decomposition, prefix-match rules — are **deferred to implementation** with documented reversible defaults (recorded in the impl doc, not litigated in rounds). The D23–D33 specifics stand as *default proposals*, not round questions. | R5 | user | decided |
| D35 | **Single-stream-per-task is the near-term structure** (supersedes D14 "required"; **downgrades D19/D23 to FUTURE**). Dataset stays multi-modal (nested); each task selects **one** stream via `Collect(stream=)` → existing single-stream collate → `Point`; that stream carries its own per-point labels (decorated from `labl`, D28). **Multi-stream-in-batch (namespaced collate + cross-stream joins) is a future additive extension** reusing the same nested dataset + decoration — not built now. Near-term **collate reverts to REPLACE** (existing single-stream); namespaced collate is a future GAIN. | R5 | user+claude | decided |
| D36 | **Event unit is configurable; default = one stored detector readout (`event_NNN`) = one sample.** Per-interaction splitting (pile-up/per-vertex) is an opt-in mode added when a task needs it. | R5 | user+claude | decided |
| D37 | **LUCiD/JAXTPC common vs different.** COMMON (in `MultiModalEventDataset`): source-mixture, hash-identity holdout, min-points, nested output, label-decoration framework, `event_identity`/`split`. DIFFERENT (per subclass): readers (PMT vs wire/pixel), geometry, label FK chains (`particle_idx→category` vs `group_id→track→label`), detector-specific sub-selectors. **"config"(LUCiD)/"run" is the generic mixture axis; JAXTPC `volume` is an ORTHOGONAL per-detector selector, not the mixture axis.** | R5 | user+claude | decided |
| D38 | **Label model: per-point `segment_*`/`instance_*` + per-event `target_*` now, via an OPEN/extensible decoration framework** (axes are registered entries in `label_config`, not hardcoded) so new axes (edge/relational/graph for future NuGraph, hierarchy labels) are additive, not a rewrite. | R5 | user+claude | decided |
| D39 | **Multi-stream: design-for-extension (build single-stream now; lock the seams).** Recommended over build-now: no concrete multi-stream consumer is specified, the namespaced-collate/cross-stream-join is the highest-risk piece (coherent cross-stream mix-up is a research problem), and the design makes it a **bounded additive** change. **Seams to lock now:** (1) nested dataset output (never flatten in dataset); (2) per-event label decoration (stream self-contained, joins resolved before batching); (3) stream-aware collate *structure* (operate on a named stream even with one); (4) stable `event_identity` for future cross-stream alignment. **Flips to build-now only if a concrete multi-stream model is committed this cycle.** | R-final | claude→user | decided |
| D40 | **JAXTPC is in scope for this build** (parallel to LUCiD): base + readers + label decoration + `jaxtpc_seg` migration all included. JAXTPC specifics: `volume` is an orthogonal sub-selector (D37), NOT the mixture axis; sensor `n_hits` via Σ`n_pixels` (+ a writer-side per-event `n_hits` ask); `track_interaction` already surfaced; `source_event_idx` present in files (`save.py`). | R-final | user | decided |
| D41 | **Eval reproducibility baked in:** per-run record of the holdout spec (seed + fractions/identity) and the exact transform pipeline; **train≡eval transforms enforced**; probe/evaluators consume the *same* registered transforms (not a re-specified pipeline). | R-final | user | decided |
| D42 | **Cross-modality desync (handoff §4) is REAL and our base INHERITS it.** Our `MultiModalEventDataset` re-implements `_n_events=min(...)` + one `local_idx` to every reader, builds NO joint index → holdout/identity (D26/D40) silently depend on alignment we never enforce; single-stream (D35) dodges it only for single-modality unlabeled runs, **label decoration re-exposes it** for every labeled task (stream-reader + labl-reader joined at same `local_idx`). **Fix: add a joint-index step to Part 02** — intersect present `event_*` keys across loaded modalities → one canonical map; `_read_event` translates `local_idx` per modality; replace `min(...)` with intersected size; manifest/identity built over the intersected index. `read_meta`'s `source_event_idx` is the join key (already surfaced). | R-handoff | claude→user | decided |
| D43 | **Phase A lands FIRST as a standalone bug-fix PR on `jaxtpc-loader-codec-opt`, before the de-fork.** Patch the current `src/pimm_data/jaxtpc.py` (the file the de-fork KEEPS; only the pimm-side vendored copy is deleted) — A1 meta-cache, A2 joint index, A3 volume-aware+raise, A4 length-mismatch, A5 cross-modality regression test. The base then factors A2 up. Fold A5 into Step-0 test matrix. Not throwaway work; not absorbed by Step 1. | R-handoff | claude→user | decided |
| D44 | **Intercept `min_deposits`/`min_segments` at the base** (route through dataset-level min-points on the joint index); deprecate/no-op the step-reader internal mask so it can't desync. **Carve out volume-aware min-points** — reconciles with D37: `volume` may scope min-points, but still must NOT affect holdout/identity. | R-handoff | claude→user | decided |
| D45 | **Three distinct axes; do NOT collapse:** `sources=` (D9 mixture, labeled, config-stratified) ≠ **multi-run/shard-union** (same physics, one label, run-as-identity) ≠ **shard-tag** (orthogonal within-run sub-selector, like `volume`). doraemon's 4 `run_*` dirs are NOT 4 sources. **Gap:** add a within-config multi-run + shard-tag sub-selector (handoff B1/B2) to the base — Phase B, after the base lands. | R-handoff | claude→user | decided |
| D46 | **A1 `_read_shard_meta` lru_cache adopted as an impl detail under the manifest-cache build** (dedup cross-reader file opens during the rank-0 scan); complementary to D27, not competing. **Branch drift: NONE** — specs grounded post-`0757ee0`; identity correctly keys on `source_event_idx` (not `local_idx`/`event_num`). | R-handoff | claude | decided |
| D47 | **Manifest-as-INPUT (B4 include/exclude) + overflow-CSV contract + CLI tooling (C) = Phase B/C, deferred.** Distinct from our internal `.npz` cache (machine-generated, speed). The *snapshot* half (`resolved_manifest`) is superseded by D41; the *include/exclude curation contract* is a real gap, scoped to Phase B. A4 length-mismatch warn + `strict_lengths` added to Part 02. | R-handoff | claude→user | decided |
| D48 | **OWED BY USER (gate Phase B scope, not Phase A):** (G1) how far now — Phase A only / A+B1 shard-selection / A+full B / +CLI; (G2) multi-run mechanism — `runs=` one dataset (recommended; now a no-structural-change `sources=` extension) / ConcatDataset / one-run-at-a-time. Phase A proceeds regardless. | R-handoff | user | open |

---

## Part IX — De-fork inventory (corrected by code-grounding)

**Large in file count, small in net new code — EXCEPT collate.** Corrections
from review: `defaults.py`/`builder.py`/`detector_transforms.py` are
**adapter/superset-equivalent, not byte-identical** (need a registry/logger
re-export shim, Rf); **collate is GAIN, not REPLACE** (D23 net-new
multi-stream); the transform delta on the branch is **only `min_keys`**
(max/mean/first are new design); **`target_mask` has no producer** (hmae
config↔`HMAECollate` drift — flag).

| pimm file | action |
|---|---|
| `utils.py` (collate) | **GAIN** — extend to namespaced multi-stream `multistream_collate_fn` (D23); current single-cloud `collate_fn` kept for legacy |
| `anchors.py` | REPLACE (byte-identical) |
| `defaults.py`, `builder.py`, `detector_transforms.py` | REPLACE via shim (adapter/superset-equivalent) |
| `transform.py` | MERGE delta: v3 vertex/`is_primary` plumbing, `RelativeLogNormalize`, `LogTransform.clip`, **`min_keys`** (+ new max/mean/first per D29), `get_view` guard, `MixedScaleGeometryMultiViewGenerator`. *Already identical:* `InstanceParser`, `LocalCovarianceFeatures`, `HierarchicalMaskGenerator`, `HMAECollate`, `RandomDrop`, `ComputeAnchors`, `CropBoundary` |
| `pilarnet.py` | MERGE: v2→**v3** `cluster_extra` (6-wide, `is_primary`) + **shared-`rotations`** overlay param |
| `jaxtpc_dataset.py`, `lucid_dataset.py`, old `readers/` | REPLACE — **but Ra**: migrate `configs/detector/_base_/jaxtpc_seg.py` + `semseg-...-jaxtpc-5cls.py` off `seg/resp/corr` → `step` first; fix `pimm/datasets/__init__.py:9` stale `lucid_dataset` import |
| `lucid_event_ssl.py` | DISSOLVE → `MultiModalEventDataset` + config |
| `dataloader.py`, `pimm/utils/registry.py`, `__init__.py` | KEEP-IN-PIMM (DDP; model/hook registries; re-export shim) |

**pimm-data must GAIN:** `MultiModalEventDataset` base (D6,D26–D30);
**namespaced multi-stream collate** (D23); transform merges (D11/D29); reader
surfacing incl. **`source_event_idx`/`config_id`/`n_hits`** (D27); registry
unification.

---

## Part X — Task → stream → label matrix

| Task | Primary stream | Label (from `labl`) | Eval |
|---|---|---|---|
| SSL pretrain (Sonata/MAE/JEPA) | `sensor` / `step` | none | `EventLinearProbe`/`Pretrain`/`OnlineLinearProbe` |
| Event PID / classification | `sensor`/`step` | `event_label` (config) / `per_interaction.neutrino_pdg` | `EventLinearProbeEvaluator` |
| Semantic seg | **`hits`** (unagg) / `step` | `segment_pid ← per_particle.category` | `SemSegEvaluator` |
| Instance / panoptic | **`hits`** (unagg) / `step` | `segment_pid` + `instance_particle`(`particle_idx`)/`instance_ancestor`/`instance_interaction` | `InstanceSegmentationEvaluator` |
| Vertex regression | `sensor`/`step` | `target_vertex ← per_interaction.vertex` (per-event) | regression |
| Energy regression | `sensor`/`step` | `target_energy ← neutrino_energy / initial_energy` | regression |
| Containment | `sensor`/`step` | `target_contained / segment_contained` | classifier |

**labl → schema-key map** (D28): `per_particle.category → segment_pid`;
`particle_idx → instance_particle`; `per_particle.ancestor_particle_idx →
instance_ancestor`; `per_track.interaction → instance_interaction` /
`segment_interaction`; `per_interaction.vertex/neutrino_energy →
target_vertex/target_energy` (per-event); `*.contained → target_contained`.
Emitters confirmed in code: `PDGToSemantic` (`segment_motif/segment_pid/
instance_*`), `_decorate_*_from_labl`, `pilarnet` v3 `cluster_extra`, `HMAECollate`
(`target_coords/energy`; **`target_mask` has no producer — config drift**).

---

## Part XI — Open items for the re-scoped rounds (R5–R9)

- **R5 collate (keystone):** input to collate (nested dict vs pre-shaped); per-stream offset ownership; cross-stream FK rebasing vs per-event resolution (D24 default); model-side primary-stream→`Point` adapter; mix-up/`inseg_collate_fn` under multi-stream.
- **R6 base API:** frozen `MultiModalEventDataset.__init__` (sources/holdout/min_points/modalities/label_config); `event_identity`/`split`; hash + fallback; manifest-cache build/read protocol under DDP.
- **R7 labels:** `label_config` shape; reader-FK vs dataset-decorate boundary; `index_valid_keys` prefix-match; per-event vs per-point targets; negative-time policy value (D31).
- **R8 JAXTPC parity:** multi-volume holdout granularity; missing sensor/hits `n_hits` attr (Rd); identity field naming (JAXTPC has `track_interaction`, lacks `source_event_idx`?); Track B re-home sequencing.
- **R9 eval/repro + testing + DDP + rollout:** per-run recording of split/transform/remap (train≡eval); test matrix (transform parity, holdout determinism, multi-stream collate, label decoration across configs, reducers, cheap==array min-points, migration smoke) on synthetic fixtures; DDP rank-identical index + h5py fd budget; rollout order (D33).

## Version log
- **v1** — process + scope + WAND scan + 5-agent seed review + decision skeleton.
- **v2** — Round-4 4-agent cadence review folded in: D23–D33; resolved D4/D19 flattening (namespaced collate, model-side flatten); cross-stream joins per-event (D24); orphaned aggregation deferred (D32); holdout/identity/min-points specifics (D26–D27); label decoration + schema reconciled to real configs (D28); GridSample reducer map (D29); base via `TestModeMixin` (D30); negative-time correctness split (D31); rollout + re-scoped rounds + Track B re-home (D33); corrected de-fork inventory (collate = GAIN; adapter-equivalent; Ra/Rb specifics).
