# Part 00 — Index, overview & build sequencing

> **Boundary note (2026-09-13).** Where this document places the LArTPC forward
> model — `AddNoise`, `Digitize`, the noise spectrum, `add_intrinsic_noise` — in
> pimm-data, that is no longer true. Those moved to **helix** (pimm-data
> `b592600` / helix `8075a53`): pimm-data keeps `densify`, which every detector
> family needs, and gave up the physics only one family applies. The transform
> registry raises on duplicate registration, so the two packages are lockstep.
> See helix `docs/ARCHITECTURE.md` §3. The design reasoning below is kept as
> written; only the ownership changed.

**Status:** navigational entry point for the pimm-data data-layer build. Read this
first. It maps the per-part specs, gives the dependency graph and build order,
pins the cross-part contracts that must not drift, traces decisions to parts, and
states the definition of done. It is the **map, not a re-derivation** — every
"how" lives in the cited part; every "why" lives in the decision log.

**The two upstream documents this series serves:**
- `implementation_plan_pimm_data_datalayer.md` — the **master build spec**
  (executable plan; §1 architecture, §2 rollout runbook Steps 0–5, §3 component
  specs, §6 test matrix, §9 D39–D41 resolutions).
- `engagement_plan_transform_dataset_placement.md` — the **decision record / spine
  of truth** (Part VIII decision log D1–D41, append-only; Part IX de-fork
  inventory; Part X task→stream→label matrix).

When a part spec and the master plan disagree on a reversible detail, the part
spec (later, code-grounded) wins; when anything disagrees with the decision log on
a *structural* choice, the decision log wins (D34: rounds settle structure, impl
settles reversible details).

> **Filename numbering is authoritative; internal "Part NN" labels drift.** The
> sibling specs were written before this file's numbering was fixed, so their
> in-text "Part NN" cross-references do **not** all match the filenames. Use the
> **§2 Part index** and **§5 Cross-part contracts** tables in this doc as the
> canonical mapping. See §2's note for the exact drift (e.g. `02_dataset_base.md`
> calls the readers part "Part 04" and label decoration "Part 05"; the real files
> are `03_readers.md` and `04_label_decoration.md`).

---

## 1. Orientation (what is being built)

**pimm-data is taking ownership of the entire data layer** — datasets,
transforms, collate, readers, registries — so it can drive broad multi-task work
(SSL/pretrain, semantic seg, instance/panoptic, vertex/energy/PID/containment) on
**both** detectors: water-Cherenkov (LUCiD/WAND) and wire-TPC (JAXTPC). pimm keeps
only trainer/DDP/model/hook code. The keystone is a **`MultiModalEventDataset`
base** that owns event *selection* (multi-source mixture, hash-on-stable-identity
holdout, cheap-`n_hits` min-points, a **joint event index** across modalities,
`event_identity`/`split`); `LUCiDDataset` and
`JAXTPCDataset` inherit it and `LUCiDEventSSLDataset` dissolves into base + config.
Datasets emit **nested per-stream** dicts; each task selects **one** stream via
`Collect(stream=)` → the existing **single-stream collate** → `Point`, and that
stream carries its own per-point labels decorated from `labl`.

Four scope facts frame everything below:
- **Cross-modality desync is real and the base must NOT inherit it (D42–D44).** The
  base builds a **joint event index** (intersect present `event_*` keys across
  modalities, keyed on `source_event_idx`) instead of the per-reader
  `_n_events=min(...)` + shared `local_idx` that silently misaligns modalities under
  `min_deposits>0` or a partial gap. The fix ships **first** as a standalone Phase-A
  bug-fix PR on `jaxtpc.py` (D43), then the base factors it up (02 §3.3a). The
  shard-tag / multi-run sub-selectors (≠ the `sources=` mixture axis, D45) and the
  manifest-as-input contract (D47) are **Phase B, deferred**.
- **Single-stream near-term (D35).** One stream per task; the existing single-
  stream collate is a byte-identical REPLACE (verified `diff`).
- **JAXTPC + LUCiD both in scope from the start (D40).** Not LUCiD-first.
- **Multi-stream is design-for-extension, NOT built (D39).** Four seams are
  locked now so the future namespaced-collate + model-side adapter is a bounded
  addition, never a rewrite.
- **Track B (densify/noise) is JAXTPC-only and deferred (D1/D33)** — re-homed to
  `gpu_batch_transforms_plan.md`, sequenced after the base + readers land. Not in
  this series.

**Doc map.** Seven implementation parts (this one is the index, 00) plus the two
upstream docs:

```
docs/
├── implementation_plan_pimm_data_datalayer.md   ← master build spec (§1–§9)
├── engagement_plan_transform_dataset_placement.md ← decision log D1–D41 (spine)
└── impl/
    ├── 00_index.md                  ← YOU ARE HERE (overview + sequencing)
    ├── 01_transforms.md             ← transform merges + index_operator prefix-match
    ├── 02_dataset_base.md           ← MultiModalEventDataset + TestModeMixin
    ├── 03_readers.md                ← read_meta + read_event surfacing
    ├── 04_label_decoration.md       ← label_config + generic decorator
    ├── 05_collate_streams_eval.md   ← collate + Collect(stream=) + eval rewire + repro
    ├── 06_defork_rollout_packaging.md   ← (expected, see path) — rollout/shim/submodule
    └── 07_test_matrix_fixtures.md       ← (expected, see path) — Step-0 gate + fixtures
```

> **06 and 07 not yet written at index time.** As of writing,
> `impl/06_defork_rollout_packaging.md` and `impl/07_test_matrix_fixtures.md` do
> **not exist on disk** (only 01–05 are present). Their scope is fully specified by
> the master plan (`implementation_plan_pimm_data_datalayer.md` §2 rollout Steps
> 0–5 and §5 packaging for Part 06; §6 test matrix and the per-part §6 test
> sections for Part 07), and every other part's "tests" and "dependencies"
> sections forward-reference them. Treat the entries below as **expected** and
> follow the cited master-plan sections until the files land.

---

## 2. Part index

| № | Title | File | One-line scope | Key decisions |
|---|---|---|---|---|
| 00 | Index / overview / build sequencing | `impl/00_index.md` | This map: part index, dep graph, build order, cross-part contracts, decision traceability, DoD | (all — navigational) |
| 01 | Transforms | `impl/01_transforms.md` | `RelativeLogNormalize`, `GridSample` `{key:op}` reducers, `LogTransform.clip`, `get_view` guard, v3 vertex/`is_primary` plumbing, `MixedScale…`, **`index_operator` prefix-match** | D11, D25, D29, D31, D34, D38 |
| 02 | Dataset base | `impl/02_dataset_base.md` | `MultiModalEventDataset` + `TestModeMixin`: source mixture, blake2b holdout, manifest cache, min-points, `event_identity`/`split`/`data_list`/`datasets`, `get_data_name` collision fix | D6, D7, D8, D9, D26, D27, D30, D34, D36, D37, D40 |
| 03 | Readers | `impl/03_readers.md` | Add `read_meta(idx)→{source_event_idx, n_hits}` (attr-only) to all 8 readers; `read_event` surfacing (`+T_reco`, `+per_interaction`); v5 docstring fix | D10, D27, D40 |
| 04 | Label decoration | `impl/04_label_decoration.md` | `label_config` axis-spec schema + one generic `_decorate_from_labl(sub, labl, fk_resolver)`; per-detector `fk_resolver`; named keys (`segment_pid`/`instance_*`/`target_*`); per-event vs per-point vs `event_broadcast` | D20, D22, D28, D38 |
| 05 | Collate, streams, eval & repro | `impl/05_collate_streams_eval.md` | Confirm single-stream collate is byte-identical REPLACE; `Collect(stream=)` contract; the 4 multi-stream seams (NOT BUILT); eval-hook rewire onto `event_identity`; repro contract | D19, D23, D24, D35, D39, D41 |
| 06 | De-fork, rollout & packaging | `impl/06_defork_rollout_packaging.md` *(expected, see `implementation_plan_pimm_data_datalayer.md` §2, §4, §5)* | Rollout Steps 0–5; re-export shim; submodule + SHA snapshot + torch pin; pilarnet v2→v3 (Rb); config migration (Ra); delete vendored | D5, D17, D18, D23(future), D32, D33 |
| 07 | Test matrix & fixtures | `impl/07_test_matrix_fixtures.md` *(expected, see `implementation_plan_pimm_data_datalayer.md` §6 + each part's §6)* | Step-0 parity/determinism harness on `testing.py` synthetic fixtures; fixture additions (`source_event_idx`/`n_hits`/`per_interaction`/`T_reco`/v5); gate for every flip | D33, D34, D41 (and every part's test §) |

> **Internal cross-reference drift (read before chasing a "Part NN" link).** The
> sibling specs predate this numbering. The mapping from each spec's in-text label
> to the real file:
> - `02_dataset_base.md` says **Part 04** = readers (real: `03_readers.md`),
>   **Part 05** = label decoration (real: `04_label_decoration.md`),
>   **Part 03** = collate (real: `05_collate_streams_eval.md`),
>   **Part 06** = eval/probe rewiring (real: inside `05_collate_streams_eval.md` §3.4).
> - `03_readers.md` says **Part 04** = label decoration (real: `04_label_decoration.md`),
>   **Part 02** = dataset base (correct).
> - `04_label_decoration.md` says **Part 05** = base (real: `02_dataset_base.md`),
>   **Part 06** = collate (real: `05_collate_streams_eval.md`),
>   **Part 02** = `index_operator` prefix-match (real: `01_transforms.md` §3.7).
> - `05_collate_streams_eval.md` says **Part 02** = base (correct),
>   **Part 04** = label decoration (correct), **Part 01** = transforms (correct),
>   **Part 06** = eval rewiring (it builds this itself, §3.4).
>
> When in doubt, match on **content** (the §-anchors are stable) not on the part
> number, and use §5 below for the contract owners.

---

## 3. Dependency graph

The data layer has one independent leg (transforms) and one mostly-linear leg
(readers → base → label-decoration → collate/eval). De-fork wraps everything; the
test matrix gates every flip.

```
                        ┌─────────────────────────────────────────────┐
                        │  07 TEST MATRIX & FIXTURES  (Step-0 gate)     │
                        │  gates EVERY flip; fixtures shared by 01–05    │
                        └───────────────▲──────────────▲────────────────┘
                                        │ gates         │ shared fixtures
   ┌──────────────────┐                 │               │
   │ 01 TRANSFORMS    │  (independent)  │               │
   │  - merges        │─────────────────┘               │
   │  - prefix-match ─┼── provides index_operator that carries the
   └──────────────────┘     named label keys 04 emits, through N-change
                                                        │
   03 READERS  ──read_meta(source_event_idx,n_hits)──►  02 BASE
       │             + raw FKs / per_interaction        (selection, identity,
       │                    │                            event_broadcast)
       │                    ▼                                 │
       │             04 LABEL DECORATION ◄──fk_resolver, label_config──┐
       │              (named keys from raw FKs)                        │
       │                    │ per-point segment_*/instance_*           │
       │                    │ per-event target_* (metadata)            │
       │                    ▼                                          │
       └──────────►  05 COLLATE / STREAMS / EVAL  ◄───────────────────┘
                      (Collect(stream=) → single-stream collate;
                       eval rewire onto base.event_identity;
                       4 multi-stream seams locked)

   06 DE-FORK / ROLLOUT / PACKAGING  ── wraps & sequences 01–05,
                                        deletes vendored last.
```

**Edges (X depends on Y = "Y must be correct for X to work"):**
- **03 → 02.** The base *consumes* `read_meta(idx) → {source_event_idx, n_hits}`
  for min-points + holdout; `event_identity` returns `(config_id,
  source_event_idx)` where `source_event_idx` comes from 03. The base also consumes
  03's per-modality **present-`event_*`-key sets** (`indices`/`cumulative_lengths` +
  `source_event_idx`-per-key, deduped via the `_read_shard_meta` lru_cache, D46) to
  build the **joint event index** (02 §3.3a, D42) — the desync fix. (`03_readers.md`
  §3.0/§8; `02_dataset_base.md` §3.3a/§8.)
- **02 → 04.** The base dispatches `get_data` to subclass builders that call
  `_decorate_from_labl`; it owns the `label_config=` constructor arg and the
  `event_broadcast` (`event_label`/`config_id`) materialization. (`04_label_decoration.md`
  §8; `02_dataset_base.md` §3.8.)
- **03 → 04.** Decoration consumes the raw FKs and `per_interaction` scope that 03
  surfaces (`target_vertex` ← `vertex_{x,y,z}`, `instance_interaction` one-hop,
  etc.). Several decoration axes are **blocked** until 03 surfaces them.
  (`04_label_decoration.md` §8; `03_readers.md` §8.)
- **04 → 05.** Collate carries the per-point `event_label`/`config_id` columns
  04/02 produce; per-event `target_*` ride as `_`-prefixed list-collated metadata.
  (`05_collate_streams_eval.md` §8.)
- **01 → 04/05.** The `index_operator` prefix-match (01 §3.7) is what keeps 04's
  named per-point keys aligned through N-changing transforms inside
  `ApplyToStream`, and excludes 04's per-event `target_*`. (`05_collate_streams_eval.md`
  §8; `04_label_decoration.md` §8.)
- **02 → 05.** The eval rewire (05 §3.4) hard-requires the base's
  `event_identity`/`split` (05 §8 "hard dependency").
- **07 → all.** The Step-0 matrix must be green before any pimm-side flip; the
  `testing.py` fixtures it owns are shared by 01/02/03/04/05 and need additions
  (`source_event_idx`, `n_hits`/`n_pixels`/`n_actual` attrs, `per_interaction`,
  `T_reco`, `format_version=5`).
- **06 → all.** De-fork re-exports/re-registers 01–05 into pimm, then deletes the
  vendored files; a single `git revert` restores.

**What can be built in parallel vs what blocks:**
- **Fully parallel from the start:** **01 (transforms)** and **03 (readers)** are
  independent of each other and of the base. **07 (fixtures + harness)** can be
  stood up in parallel and *should* lead (it is the Step-0 gate). 01 is on the
  critical path only for the LUCiD-SSL config (`RelativeLogNormalize`) and for the
  prefix-match that 04/05 rely on.
- **Serial spine:** **03 → 02 → 04 → 05**. The base (02) needs `read_meta` (03);
  decoration (04) needs both the base's dispatch (02) and the readers' raw FKs
  (03); collate/eval (05) needs 02's `event_identity` and 04's per-point columns.
- **06 wraps the lot** and runs last (shim mid-way, delete at the end).

---

## 4. Recommended build order

Aligned to the master plan's rollout runbook (`implementation_plan_pimm_data_datalayer.md`
§2, Steps 0–5). **Invariant at every step: existing PILArNet/panda/hmae training
works.** Each step is independently landable and revertable; the gate between steps
is concrete.

| Order | PR / step | Parts landed | Master-plan step | Gate to advance |
|---|---|---|---|---|
| 0 | **Phase A — cross-modality joint-index bug fix (standalone PR, D42/D43)** | patches current `src/pimm_data/jaxtpc.py` (A1 meta-cache, A2 joint index, A3 volume-aware+raise, A4 length-mismatch); A5 test folded into 07/02 §6.16. Detailed in 06 §4 (Phase A). | — (precedes Step 0; not a de-fork step) | A5 cross-modality regression green (both variants fail on HEAD today); existing robustness tests green. **Lands before Step 0**; the base (02 §3.3a) factors A2 up. |
| 1 | **Step 0 — Test matrix (no code move)** | 07 (harness + `testing.py` fixture additions) | §2 Step 0 | Step-0 suite stands up green on placeholder/branch fixtures; pure gate |
| 2 | **Step 1a — Transforms** | 01 (merges + prefix-match) | §2 Step 1(a)(b) | Transform parity + new-behavior tests green (01 §6); branch-parity skips cleanly if branch absent |
| 3 | **Step 1b — Readers** | 03 (`read_meta` + surfacing + v5 docstrings) | §2 Step 1(d) | Reader tests green (03 §6): `read_meta` == array count, `per_interaction`/`T_reco` surfaced, 3-tuple step guard |
| 4 | **Step 1c — Base** | 02 (`MultiModalEventDataset` + `TestModeMixin`) | §2 Step 1(c) | Holdout determinism / rank-identical / cheap==array tests green (02 §6); `DefaultDataset` byte-identical after mixin extraction |
| 5 | **Step 1d — Label decoration** | 04 (`label_config` + generic decorator + `fk_resolver`) | §2 Step 1(e) | Decoration == hand FK-gather; named keys present; per-event `target_*` length-1 (04 §6) |
| 6 | **Step 1e — Collate/eval contract** | 05 (collate REPLACE confirm; eval rewire onto `event_identity`; seams documented) | §2 Step 1 + §3.6/§3.7 | Collate byte-identity guard; probe disjointness via `event_identity`; train≡eval transform assertion (05 §6) |
| 7 | **Step 2 — Re-export shim** | 06 (shim in `pimm/datasets/__init__.py`; re-register into `DATASETS`/`TRANSFORMS`) | §2 Step 2 | PILArNet 1-step smoke through the shim |
| 8 | **Step 3 — Flip transforms + PILArNet** | 06 (after Rb pilarnet v2→v3 merge, §5) | §2 Step 3 | Identical first-batch tensors vs vendored |
| 9 | **Step 4 — Migrate JAXTPC configs, flip `JAXTPCDataset`, dissolve `LUCiDEventSSLDataset`** | 06 (Ra config migration, §4) | §2 Step 4 | JAXTPC semseg + LUCiD SSL configs build and run 1 step |
| 10 | **Step 5 — Delete vendored** | 06 (last commit; single `git revert` restores) | §2 Step 5 | D33 gate: grep `seg|resp|corr|output_mode` in `configs/` clean; `jaxtpc_seg.py` migrated; `__init__.py:9` stale import fixed; full parity suite green; ≥1 soaked PILArNet run |

Notes:
- **Phase A (order 0) lands FIRST as a standalone bug-fix PR (D43)**, before Step 0
  and outside the de-fork. It patches the current `src/pimm_data/jaxtpc.py` (the file
  the de-fork keeps) to fix the cross-modality desync (handoff §4): A2 builds the
  joint event index that the base (02 §3.3a) later factors up. Not throwaway, not
  absorbed by Step 1. Its A5 regression test is folded into the Step-0 matrix.
  Phase B (shard-tag / multi-run / manifest-as-input) and the user gate D48 are
  **deferred** — not in this build order.
- **Steps 1a–1e (orders 2–6) are the additive build in pimm-data with no pimm
  change** (master §2 Step 1). They can land in the spine order above; 01 and 03
  may land in either order (parallel). 02 must precede 04; 04 must precede 05's
  eval rewire being meaningful (though 05's collate-confirm and seam docs can land
  earlier).
- **06 (Steps 2–5) is sequential by construction** — shim before flip, flip before
  delete. Rb (pilarnet v2→v3) is a prerequisite for Step 3; Ra (two config
  migrations) for Step 4.
- The **07 gate is re-run before every advance**, not just at Step 0.

---

## 5. Cross-part contracts

These are the interfaces that span parts and must agree exactly so the parts don't
drift. Each row: the **shape/signature**, the part that **defines** it, and the
parts that **consume** it.

### 5.1 Named-label-key schema (`segment_*` / `instance_*` / `target_*`)

The decorator emits **named schema keys**, not bare `segment`/`instance`. The full
named set: `segment_pid`, `segment_interaction`, `instance_particle`,
`instance_interaction`, `instance_ancestor`, `target_vertex`, `target_energy`,
`target_contained`, plus `event_label`/`config_id`. Bare `segment`/`instance`
survive only as the back-compat single-axis alias (the JAXTPC `label_key=` path).

- **Scope tags** (the load-bearing distinction): `scope="point"` → `(N,1)`/`(N,)`
  per-point column (subset by N-changing transforms); `scope="event"` → per-event
  target, length-1 / `(D,)`, carried as `_`-prefixed metadata, **NOT** subset;
  `scope="event_broadcast"` → per-event value materialized to a `(N,1)` per-point
  column so `Collect` lifts it and the probe slices by offset.
- **Defines:** `04_label_decoration.md` §3.1 (schema) + §3.4 (per-detector default
  map).
- **Consumes:** `01_transforms.md` §3.7 (prefix-match carries `segment*`/
  `instance*`/`target*` underscore-boundary keys through N-change; excludes
  per-event `target_*` by leading-dim ≠ `n_points`); `05_collate_streams_eval.md`
  §3.2/§4.2 (`Collect(keys=[…,'event_label'])`, offset-slice recovery); the configs
  and evaluators (`panda/panseg` consume `segment_pid`/`instance_particle`).
- **Drift guard:** the prefix-match (01) and the decorator (04) must use the same
  underscore-boundary rule (`name == p or name.startswith(p + "_")`, prefixes
  `("segment","instance","target")`). A new axis family = a new `label_config`
  spec **and** a new prefix in 01 §3.2.

### 5.2 `read_meta(idx)` output

```python
reader.read_meta(idx) -> {'source_event_idx': int | None, 'n_hits': int}
```
Reads **only `evt.attrs`** (and per-file `config/` vectors) — never decodes
arrays. `source_event_idx`: per-file vector `config/source_event_idx` (preferred,
O(1)) → per-event attr → `None`. `n_hits`: per-reader attr-only count
(LUCiD sensor `n_hits`; LUCiD step `n_segments`; LUCiD hits `n_particle_hits`;
JAXTPC sensor Σ`n_pixels`; JAXTPC step/hits Σ`n_actual`; labl → `0`). Never raises.

- **Defines:** `03_readers.md` §3.0 (skeleton + precedence) and §3.1–3.8 (per
  reader).
- **Consumes:** `02_dataset_base.md` §3.4/§3.6 (manifest-cache build walks
  `read_meta` once per event, rank-0, under DDP barrier; min-points filter +
  holdout key). The base owns the D26 fallback warning when `source_event_idx is
  None`.

### 5.3 `event_identity(idx)` signature

```python
dataset.event_identity(idx) -> (config_id: int, source_event_idx: int)
```
Modality-independent, stable, pure function of file-discovered
`source_event_idx` + `config_id` (not positional `local_idx`). Public `self.split`
∈ `{'train','val','test','all'}` survives `Subset`. `self.data_list` is
`list[(source_idx, local_idx)]` where `local_idx` is a **canonical joint-index
position** (D42), not a raw shared reader index; `self.datasets` is `list[dict]` each
carrying `source_root`/`data_root`/`name`/`config_id`/`label`. The joint index
(02 §3.3a) is what makes `event_identity` truly modality-independent: every loaded
modality is translated off it, so the returned `source_event_idx` is the one shared
by all modalities for that event (Invariant 9, 02 §4).

- **Defines:** `02_dataset_base.md` §3.7 (identity) + §3.3a (joint index it keys on).
- **Consumes:** `05_collate_streams_eval.md` §3.4 (the rewired
  `_event_keys` = `{event_identity(i)}`; disjointness guard; this is the "hard
  dependency"). It is also **Seam 4** (future multi-stream alignment key).

### 5.4 Nested per-stream dataset output

```python
dataset.get_data(idx) -> {
    'name': str, 'split': str,
    'sensor': {coord, feat, value, …},      # per modality requested
    'step':   {coord, energy, segment_pid, instance_particle, …},
    'hits':   {coord, …, segment_pid, instance_particle, …},
    'labl':   {…raw labl tables…},           # top-level, never collected
    'bridges': {…},                          # per-event join artifact
    '_targets': {target_vertex:(3,), target_energy:(), …},  # _-prefixed, dropped at collate
}
```
**Never flatten in the dataset** (no bare top-level `coord`/`segment`); each stream
is self-contained. `event_label`/`config_id` are materialized **per-point** inside
the primary stream.

- **Defines:** `02_dataset_base.md` §3.8 (dispatch + `event_broadcast`
  materialization) + the existing nested form in `lucid.py`/`jaxtpc.py`
  (`05_collate_streams_eval.md` §2.4, Seam 1).
- **Consumes:** `05_collate_streams_eval.md` §3.2 (`Collect(stream=)` lifts one
  stream); `04_label_decoration.md` (writes the per-stream `segment_*`/`instance_*`
  columns).

### 5.5 `Collect(stream=)` flat output + single-stream collate batch

`Collect(stream='step', keys=(…), feat_keys=(…))` pulls keys from the nested
`data_dict['step']` and emits a **bare** flat dict (`coord`, `feat`, named labels,
`offset=tensor([n_points])`, `name`/`split` auto-passthrough). After `collate_fn`
the batch is flat: `coord (ΣN,3)`, `feat (ΣN,c)`, `offset (B,)` = `cumsum([n_b])`
via the diff/cumsum rebasing, `event_label (ΣN,1)`, `name`/`split` as python lists.
**`_`-prefixed keys are silently dropped at collate.**

- **Defines:** `05_collate_streams_eval.md` §3.2 (`Collect` contract) + §3.1/§2.1
  (collate is byte-identical REPLACE; do **NOT** overwrite pimm-data's `Collect` —
  it is ahead of the branch). `4.1` is the concrete batch shape.
- **Consumes:** the model (`Point`), the eval probe (`point.feat`/`offset`), and
  every task pipeline's terminal transform. `01_transforms.md` explicitly must NOT
  touch `Collect`.

### 5.6 The four multi-stream seams (locked now, NOT built)

D39/§9.1: lock these so the future namespaced collate + model-side primary/aux
adapter is a bounded addition. (1) **Nested dataset output** — never flatten (§5.4
above). (2) **Per-event label decoration** — each stream self-contained, cross-
stream joins resolved before batching, carried `_`-prefixed; never a join-at-
collate. (3) **Stream-aware collate structure** — the single-stream collate
operates on a *named* stream, so adding streams is a loop, not a rewrite.
(4) **Stable `event_identity`** — future batches align streams by event via this
key (§5.3 above).

- **Defines / documents:** `05_collate_streams_eval.md` §3.3 (explicitly
  NOT-BUILT). Seam 1 lives in 02, Seam 2 in 04, Seam 3 in 05, Seam 4 in 02.
- **Consumes:** the future multi-stream build only; this series guarantees the
  seams hold but builds none of the future path.

---

## 6. Decision → part traceability

Each major decision mapped to the part(s) that implement it. (Full text:
`engagement_plan_transform_dataset_placement.md` Part VIII.)

| Decision | Summary | Implemented in |
|---|---|---|
| **D1** | LUCiD fully sparse; densify/noise (Track B) out of scope, JAXTPC-only, deferred | *None of 01–07* (re-homed to `gpu_batch_transforms_plan.md`); 06 notes the sequencing |
| **D6** | New `MultiModalEventDataset` base owns selection + nested output | **02** |
| **D7** | Holdout = hash on stable `(config_id, source_event_idx)`; 3-way; stratified | **02** §3.5 (specifics from D26) |
| **D8** | min-points via cheap `n_hits`, inclusive `>=`, filter-then-hash-split | **02** §3.6 (uses **03** `read_meta`) |
| **D9** | Config-mixture native: `sources=[{root,label,weight}]`, explicit labels | **02** §3.3/§3.9 |
| **D19** | Multi-stream batch namespaced; model names primary stream | **05** §3.3 (Seam 1/3; **FUTURE**, downgraded by D35/D39) |
| **D23** | Namespaced `multistream_collate_fn`, model-side flatten; `Collect(stream=)` demoted | **05** §3.3 (**FUTURE, NOT BUILT**); 06 notes collate-as-GAIN deferral |
| **D25** | N-changing safety via `index_operator` prefix-match + per-stream `ApplyToStream` | **01** §3.2/§3.7 (prefix-match); **05** §3.2 (`ApplyToStream` wrap) |
| **D26** | Holdout specifics: blake2b, config-stratified, fallback + warning; `event_identity` in base | **02** §3.5/§3.7 |
| **D27** | Reader surfacing + manifest cache (rank-0 build under DDP barrier) | **03** (`read_meta` surfacing) + **02** §3.4/§3.6 (manifest cache) |
| **D28** | Reader emits raw FKs → dataset decorates; per-event targets per-event; named keys | **04** (decorator + named map); **03** (raw FK / `per_interaction` surfacing) |
| **D30** | Base inherits `DefaultDataset` via factored `TestModeMixin` | **02** §3.0 |
| **D35** | Single-stream-per-task near-term; collate reverts to REPLACE | **05** §3.1; framed across all parts |
| **D37** | LUCiD/JAXTPC common (base) vs different (readers/geometry/FK/sub-selectors); `volume` orthogonal | **02** §3.4 (base/subclass split, `volume` sub-selector); **04** (`fk_resolver` per detector); **03** (per-detector readers) |
| **D38** | Open/extensible decoration framework (axes = registered `label_config` entries) | **04** §3.1/§3.6; **01** §3.7 (prefix families for new axes) |
| **D39** | Multi-stream design-for-extension; lock the four seams | **05** §3.3 (seams); Seam 1/4 in **02**, Seam 2 in **04** |
| **D40** | JAXTPC in scope from the start (base + readers + decoration + `jaxtpc_seg` migration) | **02** §3.4, **03** (JAXTPC readers), **04** (JAXTPC resolver), **06** §4 (config migration) |
| **D41** | Eval reproducibility: per-run holdout+pipeline record; train≡eval transforms | **05** §3.5; **02** §3.7 (the stable identity the repro contract replays) |
| **D42** | Cross-modality desync is real and inherited; fix = a joint event index (intersect present `event_*` keys across modalities, keyed on `source_event_idx`) | **02** §3.3a (joint index; supersedes the inherited `_n_events=min`); **03** §3.0 (present-key surface); **06** §4 Phase A (A2) |
| **D43** | Phase A lands FIRST as a standalone bug-fix PR on `jaxtpc.py`; the base factors A2 up | **06** §4 (Phase A, before Step 0); **00** §4 (order 0); **02** §3.3a |
| **D44** | Intercept `min_deposits`/`min_segments` at the base on the joint index; no-op the step-reader mask; volume-aware min-points does NOT affect holdout/identity | **02** §3.6 (interception + volume-aware), §5, §6.16(c); **03** §8 (reader mask dead) |
| **D45** | Three distinct axes — `sources=`(mixture) ≠ multi-run(same physics, one label) ≠ shard-tag(sub-selector); multi-run/shard-tag is Phase-B | **02** §1 (three-axes note); Phase B deferred |
| **D46** | `_read_shard_meta` lru_cache dedups cross-reader opens under the manifest build; readers surface per-modality present-key sets | **03** §3.0/source-decisions; **02** §3.3a/§3.4 (shared scan) |
| **D47** | A4 length-mismatch warn + `strict_lengths`; manifest-as-INPUT + overflow CSV = Phase B | **02** §3.1 (`strict_lengths`), §3.3a, §5; Phase B deferred |
| **D48** | OWED BY USER (gate Phase-B scope, not Phase A): how far now; multi-run mechanism | **open** — Phase A proceeds regardless; not in this build order |

Supporting decisions also realized in this series: **D10** (surface all streams) →
03; **D11** (upstream transforms) → 01; **D29** (GridSample `{key:op}` reducers) →
01 §3.2; **D31** (negative-time correctness) → 01 §3.1/§3.3; **D34** (reversible
details with documented defaults) → every part's "Reversible defaults" section;
**D5/D17/D18/D32/D33** (de-fork boundary, rollout, deferred aggregation) → 06.

---

## 7. Open / verify-before-coding items

Consolidated from the parts' "Reversible defaults & risks" and the master plan §7.
**Resolve or confirm these before the dependent part is coded.**

- **WAND attr names — CONFIRMED (one-shard check, `config_000001`, v5).**
  `source_event_idx` present as BOTH a per-file vector `config/source_event_idx`
  `uint32 (n_events,)` (sensor + labl) and a per-event attr; `n_hits` is a per-
  event scalar attr (no `config/n_hits` vector); `per_interaction` group fields +
  CSR primaries confirmed. (Master §3.4; `03_readers.md` locked-constraints.) **No
  longer a verify item — recorded as resolved.**
- **JAXTPC `source_event_idx` is per-event attr only (no config vector).** Stamped
  by `production/save.py:344/391/625`; there is **no** `config/source_event_idx`
  vector for JAXTPC, so the base does a per-event attr walk (cheap, O(events)/file,
  manifest-cached). The base must accept vector → attr → positional. (02 §7 risk;
  03 §3.5–3.7.)
- **`target_mask` has NO producer — do not invent one.** The hmae config references
  `target_mask` but `HMAECollate` does not emit it (config↔collate drift). Part 04
  does **not** add a `target_mask` axis; surface the drift to the hmae owner.
  (Master §7; `04_label_decoration.md` §7.)
- **Named-keys decision — RESOLVED.** The decorator emits named schema keys
  (`segment_pid`/`instance_particle`/…) directly; bare `segment`/`instance` are a
  back-compat single-axis alias only. (Master §3.5; `04_label_decoration.md` §2.3.)
  Closes the "bare-vs-named gap." Confirm the prefix-match (01) and decorator (04)
  agree on the underscore-boundary rule before coding either.
- **Writer-side `n_hits` ask (deferred, do NOT block).** JAXTPC sensor `n_hits` is
  a Σ`n_pixels` attr-walk; the optional speedup is a writer-side per-event
  `evt.attrs['n_hits']` total and/or a per-file `config/n_hits` vector (LUCiD too).
  Ship the attr-walk; file the vector as a follow-up. (Master §9.2; `03_readers.md`
  §7.) Related one-line writer fix: `make_labl.py` does not stamp
  `source_event_idx` on the labl event group — source JAXTPC identity from
  sensor/step/hits, not labl. (`03_readers.md` §3.8/§7.)
- **`min_points` `>=` vs colleague's `>` — intentional parity diff.** Base uses
  inclusive `>=` (D8); a boundary event (count == threshold) is kept, differing by
  one from the colleague's `>`. Documented, not a bug. (Master §7; `02_dataset_base.md`
  §7.)
- **`testing.py` fixtures lack `source_event_idx`/`n_hits`/`per_interaction`/
  `T_reco` today.** Tests 6.1/6.2/6.6/6.7/6.9 (base) and 1–8 (readers) require the
  fixture additions; 07 owns them. Until they land, those tests exercise the
  fallback path only. (`02_dataset_base.md` §6/§7; `03_readers.md` §6.)
- **`RelativeLogNormalize` is non-idempotent + order-sensitive.** Must run once,
  before any reorder/subset of the `time` column (relies on the raw per-event min).
  Pipeline-author responsibility, documented in the class docstring. (`01_transforms.md`
  §5.2/§7.)

---

## 8. Definition of done

The build is complete when **all** of the following hold:

1. **All parts landed.** 01–05 merged additively into pimm-data (master §2 Step 1);
   06 has executed Steps 2–5 (shim → flip transforms/PILArNet → migrate JAXTPC +
   dissolve `LUCiDEventSSLDataset` → delete vendored).
2. **Test matrix green (07 / master §6).** Transform parity (incl. branch-parity
   where the branch is fetchable, golden-array regressions where not); `GridSample`
   `min_keys` back-compat byte-equal + new `max/mean/first`; `index_operator`
   prefix-match (per-point carried, per-event `target_*` not subset); holdout
   determinism (reorder / add-remove / worker count / machine / rank-identical);
   label decoration == hand FK-gather; cheap-`n_hits` == array min-points;
   migration smoke (PILArNet/panda/hmae + migrated JAXTPC seg + LUCiD SSL build +
   1-step). Re-run green before every flip.
3. **Vendored deleted (06 / master §2 Step 5, D33 gate).** `grep
   seg|resp|corr|output_mode` in `configs/` is clean; `jaxtpc_seg.py` (+ the 5cls
   child) migrated off `seg/resp/corr` → `step`; `pimm/datasets/__init__.py:9`
   stale `lucid_dataset` import fixed; the full parity suite is green; at least one
   full PILArNet run has soaked; a single `git revert` of the delete commit
   restores the vendored tree.
4. **Both detectors run a 1-step train (master §2 Step 4 gate).** A JAXTPC semseg
   config and a LUCiD SSL config each **build and run one training step** on
   pimm-data (through the shim), with the eval probe's `event_identity`-based
   disjointness guard passing and train≡eval transforms enforced (D41).

When all four hold, pimm-data owns the data layer end-to-end for single-stream
multi-task work on LUCiD and JAXTPC; multi-stream (D39) and Track B (D1) remain
designed-for but unbuilt, as intended.
