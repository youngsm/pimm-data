# pimm-data data layer — Implementation Roadmap

> **Boundary note (2026-09-13).** Where this document places the LArTPC forward
> model — `AddNoise`, `Digitize`, the noise spectrum, `add_intrinsic_noise` — in
> pimm-data, that is no longer true. Those moved to **helix** (pimm-data
> `b592600` / helix `8075a53`): pimm-data keeps `densify`, which every detector
> family needs, and gave up the physics only one family applies. The transform
> registry raises on duplicate registration, so the two packages are lockstep.
> See helix `docs/ARCHITECTURE.md` §3. The design reasoning below is kept as
> written; only the ownership changed.

**Status:** the single execution-and-sign-off document for the pimm-data data-layer
consolidation. It sequences three bodies of work into one ordered plan: a
**correctness/desync bug fix (Phase A)** that lands first, the **de-fork build
(Steps 0–5)** that moves the data layer out of the pimm fork, and a **gated
shard/event-filtering feature (Phase B/C)** that is scoped but not yet approved.

This roadmap **defers to** the canonical docs — it does not re-derive their "how"
or re-litigate their "why." Every phase points to the spec section that owns the
detail. The owner reads this to approve each big step (§6 checklist); the developer
reads each phase's template block to execute it.

**The invariant that gates every step, top to bottom:**
**existing PILArNet / panda / hmae training works.** No step lands until that holds.

---

## 0. Document map

The canonical sources this roadmap sequences. When a part spec and the master plan
disagree on a *reversible* detail, the part spec wins; when anything disagrees with
the decision log on a *structural* choice, the decision log wins (D34).

| Doc | Role | What it is |
|---|---|---|
| `README.md` | **Entry point / map** | Reading order, canonical-vs-archive classification, status, glossary. Start here. |
| `DESIGN.md` | **Authoritative design (the "why")** | Architecture, dataset/stream/label/transform/collate design, and the distilled live decisions (§8 lists all D1–D48 with live/superseded/future status). The design reference. |
| `engagement_plan_transform_dataset_placement.md` | **Decision spine (immutable record)** | Part VIII decision log **D1–D48** (append-only); Part IX de-fork inventory; Part X task→stream→label matrix. The rest is process history; the design is distilled in `DESIGN.md`. |
| `implementation_plan_pimm_data_datalayer.md` | **Master build spec** | §1 architecture, **§2 rollout runbook Steps 0–5**, §3 component specs, §4 config migration (Ra), §5 packaging (Rb), §6 test matrix, §9 D39–D41 resolutions. |
| `impl/00_index.md` | **Map / build order** | Part index, dependency DAG, cross-part contracts (§5), decision→part traceability, definition of done. Read first when navigating the impl specs. |
| `impl/01_transforms.md` | Transforms spec | `RelativeLogNormalize`, `GridSample` `{key:op}` reducers, `LogTransform.clip`, `get_view` guard, v3 vertex/`is_primary` plumbing, **`index_operator` prefix-match**. (D11, D25, D29, D31, D34, D38) |
| `impl/02_dataset_base.md` | Dataset base spec | `MultiModalEventDataset` + `TestModeMixin`: source mixture, blake2b holdout, manifest cache, min-points, `event_identity`/`split`. **Factors Phase A's joint index up (D42/D44).** (D6–D9, D26, D27, D30, D34, D36, D37, D40) |
| `impl/03_readers.md` | Readers spec | `read_meta(idx)→{source_event_idx, n_hits}` (attr-only) on all 8 readers; `read_event` surfacing (`+T_reco`, `+per_interaction`); v5 docstrings. (D10, D27, D40) |
| `impl/04_label_decoration.md` | Label decoration spec | `label_config` axis-spec schema + one generic `_decorate_from_labl`; per-detector `fk_resolver`; named keys (`segment_pid`/`instance_*`/`target_*`). (D20, D22, D28, D38) |
| `impl/05_collate_streams_eval.md` | Collate / streams / eval spec | Single-stream collate is a byte-identical REPLACE; `Collect(stream=)` contract; the 4 multi-stream seams (LOCKED, NOT BUILT); eval-hook rewire onto `event_identity`; repro contract. (D19, D23, D24, D35, D39, D41) |
| `impl/06_defork_rollout_packaging.md` | **De-fork runbook** | Steps 0–5 file-by-file actions, per-step gate + rollback, the exact `pimm/datasets/__init__.py` shim, config migration Ra, packaging, pilarnet Rb merge. (D5, D17, D18, D23-future, D32, D33) |
| `impl/07_test_matrix_fixtures.md` | **Gates / fixtures** | The Step-0 parity/determinism harness, `testing.py` fixture extensions (P07-FIX), the consolidated test matrix (TR/DS/RD/LD/CE/MG), and the gate→step mapping (§6). |
| `shard_event_filtering_handoff.md` | Phase A bug fix + Phase B/C feature | Verified filtering surface, the cross-modality desync bug (§4), the Phase A/B/C TODO, and the user-owed scope decisions (§6 → D48). |
| `gpu_batch_transforms_plan.md` / `_handoff.md` (RETIRED 2026-09-13) | **Track B (out of scope here)** | Densify/AddIntrinsicNoise, wire-TPC-only, deferred (D1/D33). Re-homed here; not part of this roadmap. |

---

## 1. The big picture

The work runs in three phases, in this order. Each line is one-sentence; the
dependency line below fixes the ordering.

1. **Phase A — correctness/desync fix (D42–D44).** Patch `src/pimm_data/jaxtpc.py`
   to build one **joint cross-modality event index**, intercept `min_deposits` at
   the base, make min-points volume-aware, warn on length mismatch, and lock it
   with a cross-modality regression test (A5). Standalone bug-fix PR on
   `jaxtpc-loader-codec-opt`. **Lands first.**
2. **De-fork Step 0 — Test matrix (gate, no code move).** Stand up the
   parity/determinism harness + `testing.py` fixture extensions; fold A5 in.
3. **De-fork Step 1 — Additive build in pimm-data (no pimm change).** Land
   transforms (incl. prefix-match + Rb pilarnet v3 merge), readers, the
   `MultiModalEventDataset` base (factoring Phase A's joint index up), label
   decoration, and the collate/eval contract — all *behind* the vendored layer.
4. **De-fork Step 2 — Re-export shim.** Drop the re-register shim into
   `pimm/datasets/__init__.py`; lowest-risk surface (byte-identical REPLACE files +
   PILArNet) flips through it.
5. **De-fork Step 3 — Flip transforms + PILArNet.** Point the shim's `TRANSFORMS` +
   PILArNet at pimm-data; prove identical first-batch tensors vs vendored.
6. **De-fork Step 4 — Migrate JAXTPC configs, flip `JAXTPCDataset`, dissolve
   `LUCiDEventSSLDataset`.** Ra config migration; JAXTPC semseg + LUCiD SSL build
   and run 1 step.
7. **De-fork Step 5 — Delete vendored.** One commit, single-`git revert`-safe;
   D33 grep gate clean; full parity suite green.
8. **Phase B/C — shard/event filtering feature (D45/D47).** Shard-tag selection,
   multi-run axis, `event_filter` registry, manifest include/exclude, CLI. **Gated
   on the two user-owed D48 decisions; scope NOT assumed.**

**Out of scope for this roadmap:** Track B (densify/noise, D1/D33 — JAXTPC-only,
re-homed to `gpu_batch_transforms_plan.md`); multi-stream-in-batch (D19/D23/D39 —
seams locked now, build deferred). See §5.

**Dependency line (strict):**
`Phase A → Step 0 → Step 1 → Step 2 → Step 3 → Step 4 → Step 5 → (Phase B/C, after D48)`.
Within Step 1, the spine is `01 transforms ∥ 03 readers → 02 base → 04 decoration →
05 collate/eval`; 01 and 03 may land in either order. Phase A is a **prerequisite
for the base's identity contract** (D42/D43): the base factors A2's joint index up,
so A must be merged before 02 lands. Phase B builds on A's joint index (D45) and so
cannot precede the base either.

---

## 2. Phase A — correctness/desync fix (lands first)

> **PR:** standalone bug-fix PR on branch `jaxtpc-loader-codec-opt`, patching the
> current `src/pimm_data/jaxtpc.py` — **the file the de-fork KEEPS** (only the
> pimm-side *vendored* copy is deleted at Step 5). Independently valuable; not
> throwaway work; not absorbed by Step 1 (D43).

**Goal.** Make every modality return the **same physics event** for a given `idx`,
so holdout/identity/label-decoration joins are meaningful — fixing the
cross-modality desync that the current `_n_events=min(...)` + one-`idx`-to-every-reader
design silently inherits (D42).

**Scope (what's in).** The five Phase-A items from the handoff §5 / D43:
- **A1 — `_read_shard_meta` lru_cache.** Module-level `@lru_cache`
  `_read_shard_meta(path) -> (n_events, n_volumes, present_event_keys,
  readout_type)`. ~20 LOC, no API change; collapses ~3× redundant per-reader file
  opens at `__init__` (~800 files at doraemon scale). Highest ROI / lowest risk.
  (D46 — adopted as an impl detail; complementary to the manifest cache, not
  competing.)
- **A2 — joint event index at the dataset level.** Intersect present `event_*` keys
  across the **loaded** modalities → one canonical map; apply event filters; hand
  per-shard index arrays to every reader via new reader kwargs `(file_list,
  events_per_file)`. Readers fall back to their own `_find_files`/`_build_index`
  when the kwargs are absent (backward compat). One index = source of truth → fixes
  the `min_deposits>0` desync (handoff §4 #1) and the gap-induced desync (#2) by
  construction.
- **A3 — volume-aware min-points + raise-on-missing-modality.** Make `min_deposits`
  volume-aware when `volume=N` is set (handoff §4 #3); raise if `min_deposits>0` (or
  `min_segments>0`) is passed without the source modality loaded (#4) instead of
  silently no-op-ing.
- **A4 — length-mismatch handling.** Warn with **concrete per-modality counts**
  instead of the silent `min(...)`; add `strict_lengths=True` to hard-error.
- **A5 — cross-modality regression test.** `modalities=('step','sensor','hits',
  'labl')` with `min_deposits>0`, assert the **same physics event** (by an
  identifying attr) is returned across every modality for every `idx`; plus a
  gap-in-one-modality variant for #2. **Both fail on HEAD today.**

**Key decisions.** **D42** (desync is real, base inherits it; fix = joint index),
**D43** (Phase A lands first, standalone, on `jaxtpc-loader-codec-opt`; base factors
A2 up; fold A5 into Step-0 matrix), **D44** (intercept `min_deposits`/`min_segments`
at the base on the joint index, deprecate/no-op the step-reader internal mask;
volume-aware min-points must NOT affect holdout/identity). Supporting: **D46**
(A1 lru_cache).

**Depends on.** Nothing upstream — patches the live `jaxtpc.py`. Builds on the
already-landed gap-tolerant indexing (`0757ee0`) and codec support (handoff §3).

**Gate (Part 07).** **A5** is the lock — fold it into the Step-0 test matrix as the
cross-modality alignment test (it is the test the matrix's DS/RD/LD joins assume).
Phase A advances to the de-fork when A5 (and its gap-in-one-modality variant) is
green and the existing `tests/test_jaxtpc*.py` + `test_jaxtpc_robustness.py` suite
still passes. No pimm-side test applies — Phase A is pimm-data-internal.

**Rollback safety.** Single PR against `jaxtpc.py` + reader kwargs + one test file.
Readers' fallback-when-kwargs-absent keeps the old per-reader path live, so a revert
of the PR restores prior behavior with zero downstream change. No pimm import path
is touched, so PILArNet/panda/hmae are untouched by construction.

**Spec pointer.** `shard_event_filtering_handoff.md` §4 (the bug), §5 Phase A
(A1–A5 with reasons), §8 (source refs: `jaxtpc.py:132-180`/`:226-269`,
`jaxtpc_step.py:84-100`). Decision text: engagement plan Part VIII D42–D44, D46.
The base's factor-up of A2: `impl/02_dataset_base.md` (joint-index step in Part 02).

---

## 3. De-fork Steps 0–5

The de-fork moves the entire data layer into pimm-data **without ever breaking
running training** (D33 mechanism: build-behind → parity → shim → flip →
delete-last). Each step is one landable commit (or small group), independently
revertable. The Part-07 gate is **re-run before every advance**, not just at Step 0.

### Step 0 — Test matrix (gate, no code move)

**Goal.** Stand up the parity/determinism harness on synthetic fixtures so every
later flip has a concrete, green gate to clear.

**Scope.** The `testing.py` fixture extensions (**P07-FIX**: stamp
`source_event_idx`, per-event/per-plane/per-volume `n_hits` counts,
`per_interaction` group + CSR primaries, `T_reco`, `format_version=5`) + the whole
consolidated test matrix scaffolded (TR/DS/RD/LD/CE/MG). **Fold Phase A's A5 in**
as the cross-modality alignment anchor (D43). No pimm change, no flip.

**Key decisions.** **D33** (test matrix is the gate), **D34** (reversible defaults
documented), **D41** (eval-repro / train≡eval assertion, CE-05), **D43** (A5 folds
in here).

**Depends on.** Phase A merged (A5 is part of this matrix; the joint index it tests
must exist).

**Gate (Part 07).** **P07-FIX merged**; the full matrix green on synthetic
fixtures: TR-01..TR-24 (transforms), DS-01..DS-17 (base), RD-01..RD-11 (readers),
LD-01..LD-10 (decoration), CE-01..CE-09 (collate/eval). Parity tests either pass
(branch importable) **or skip with their golden pair green** (§7.4 skip-can't-mask
rule). Pure gate — no pimm change.

**Rollback safety.** Nothing in pimm changed ⇒ trivially safe. PILArNet/panda/hmae
untouched (no pimm import path moved).

**Spec pointer.** `impl/07_test_matrix_fixtures.md` §3 (P07-FIX fields, both
detectors), §4 (matrix), §5 (determinism designs), §6 (gate→step), §7 (markers,
branch stub, golden strategy); master plan §6.

### Step 1 — Additive build in pimm-data (no pimm change)

**Goal.** Land the rebuilt data layer (Parts 01–05) **behind** the vendored
datasets, so pimm still imports its own layer and training is byte-for-byte
unchanged.

**Scope.** In spine order:
- **1a — Transforms (01):** the merges (`RelativeLogNormalize`, `GridSample`
  `{key:op}` reducers with `sum_keys`/`min_keys` back-compat, `LogTransform.clip`,
  `get_view` guard, **v3 vertex/`is_primary` plumbing**, `MixedScale…`) +
  **`index_operator` prefix-match**. **Do the Rb pilarnet v2→v3 merge here**
  (`06` §8) so PILArNet is flip-ready before Step 3. Do **NOT** overwrite
  pimm-data's `Collect` — it is ahead of the branch (`stream=`).
- **1b — Readers (03):** `read_meta(idx)→{source_event_idx, n_hits}` (attr-only) on
  all 8 readers + `read_event` surfacing (`+T_reco`, `+per_interaction`, raw FKs) +
  v5 docstring fix.
- **1c — Base (02):** `MultiModalEventDataset` + `TestModeMixin` (source mixture,
  blake2b holdout, manifest cache, min-points, `event_identity`/`split`). **Factors
  Phase A's A2 joint index up** as the canonical intersected index (D42).
- **1d — Label decoration (04):** `label_config` axis-spec + one generic
  `_decorate_from_labl` + per-detector `fk_resolver`; named keys.
- **1e — Collate/eval contract (05):** confirm single-stream collate is a
  byte-identical REPLACE; rewire the eval probe onto `event_identity`; document the
  4 multi-stream seams (NOT BUILT).

**Key decisions.** **D6–D9** (base owns selection), **D11/D25/D29/D31** (transforms),
**D26/D27** (holdout + manifest cache), **D28/D38** (label decoration framework),
**D30** (base inherits `DefaultDataset` via `TestModeMixin`), **D35/D39** (single-
stream now; lock the 4 seams), **D40** (JAXTPC in scope), **D41** (repro).
**D42** (base factors the joint index up). Rb merge: **D33** / `06` §8.

**Depends on.** Step 0 green. Phase A merged (the base factors A2 up). Internal
order: 01 ∥ 03, then 02, then 04, then 05's eval-rewire (05's collate-confirm and
seam-docs can land earlier).

**Gate (Part 07).** Step-0 suite green **end-to-end** now exercising real code:
TR-* (merges + prefix-match), DS-* (base + holdout determinism + cheap==array
min-points), RD-* (readers), LD-* (decoration == hand FK-gather), **MG-01**
(`TestModeMixin` extraction byte-identical; npy `DefaultDataset` path unchanged).
Rb parity: PILArNet v1/v2 `get_data` byte-identical pre/post merge; v3
`cluster_extra` width-6 assertion fires (`06` §9).

**Rollback safety.** All changes inside pimm-data; pimm still imports its vendored
layer. Revert any pimm-data commit with **zero pimm impact**. pimm `__init__.py`
still imports vendored `pilarnet`/`jaxtpc_dataset`/`lucid_dataset` — training
byte-for-byte unchanged.

**Spec pointer.** `impl/01_transforms.md`, `impl/03_readers.md`,
`impl/02_dataset_base.md`, `impl/04_label_decoration.md`,
`impl/05_collate_streams_eval.md`; Rb merge `impl/06_defork_rollout_packaging.md` §8;
master plan §2 Step 1 + §3 component specs.

### Step 2 — Re-export shim in `pimm/datasets/__init__.py`

**Goal.** Replace the 13-line vendored `__init__.py` with a shim that re-registers
pimm-data classes into **pimm's own** `DATASETS`/`TRANSFORMS` (config `type=`
strings resolve against pimm's registries — Rf), so configs reach pimm-data through
pimm's registries.

**Scope.** The exact shim (`06` §5): import the trainer-facing names from pimm-data
(`collate_fn`/`point_collate_fn`/`inseg_collate_fn`, `DefaultDataset`/
`ConcatDataset`, `build_dataset`, dataset classes, anchors) + keep-in-pimm
`MultiDatasetDataloader`; **re-register** each pimm-data dataset/transform via
`register_module(module=Cls)` behind a **membership guard** (`_reregister`) so a
double-import or a revert never raises `KeyError: already registered`. **Start with
the byte-identical REPLACE surface** (`anchors`, collate, `builder`, `defaults`) +
`PILArNetH5Dataset` — lowest-risk first. Import `DATASETS`/`TRANSFORMS`/`Compose`/
`build_dataset` **from `pimm_data` directly** (not `from .builder`) so the shim is
delete-safe without a re-edit at Step 5.

**Key decisions.** **D18** (de-fork boundary: pimm keeps `MultiDatasetDataloader` +
registries + hooks + thin shim), **D17** (de-fork replaces most of `pimm/datasets/`),
Rf (registry de-dup — never share a `Registry` object; copy class objects by name).

**Depends on.** Step 1 (the pimm-data classes the shim re-exports must exist + be
green).

**Gate (Part 07).** **MG-02** (PILArNet builds + 1 train step through the shim),
**MG-05** (`__init__` stale-import fix — no `ImportError`), **MG-06** (exports
preserved: collate fns / `DefaultDataset` / `ConcatDataset` / `build_dataset` /
dataset classes), **CE-09** (collate byte-identity REPLACE never drifts).

**Rollback safety.** Single-file change. `git checkout pimm/datasets/__init__.py`
restores the vendored wiring; the vendored `.py` files are still present (not yet
deleted), so reverting the shim is total. The membership guard makes a
revert-then-reimport harmless.

**Spec pointer.** `impl/06_defork_rollout_packaging.md` §5 (exact shim + notes),
§7.5 (registry de-dup Rf); master plan §2 Step 2.

### Step 3 — Flip transforms + PILArNet

**Goal.** Point the shim's `TRANSFORMS` re-registration at pimm-data's
`transform.py` + `detector_transforms.py` and `DATASETS` at pimm-data's
`PILArNetH5Dataset`, so PILArNet/panda/hmae/voltmae/polarmae/lejepa run on
pimm-data's transforms.

**Scope.** Uncomment the shim's transform block (`06` §5 block 5) + flip
`PILArNetH5Dataset`. The ~21 PILArNet configs consume `segment_motif`/`segment_pid`/
`instance_particle` via `Copy`, all emitted by `pilarnet.py` — they share **no
`_base_`** with `jaxtpc_seg.py`, so they flip cleanly once the Rb merge (landed
Step 1) preserves those exact emitted keys.

**Key decisions.** **D33** (flip transforms+PILArNet before JAXTPC/LUCiD; identical
first-batch gate), Rb (`06` §8 — v3 merge preserves emitted keys; v1/v2 parity
held).

**Depends on.** Step 2 (shim in place); Rb merge landed in Step 1.

**Gate (Part 07).** **MG-07** — **identical first-batch tensors vs vendored**: seed
`random`/`np.random`/`torch` identically, build the same config against vendored vs
shimmed layer, `assert_array_equal` per batch-0 key (`coord`/`feat`/`segment`/
`offset`/`instance_*`). The named Step-3 gate. Plus TR-* parity green (or
golden-paired).

**Rollback safety.** Shim-only change; revert the shim hunk. Vendored files intact
(pre-Step-5), so PILArNet falls back to the vendored layer exactly as before.

**Spec pointer.** `impl/06_defork_rollout_packaging.md` §4 Step 3 + §6.3 (PILArNet/
panda unaffected, grep proof) + §8 (Rb); master plan §2 Step 3.

### Step 4 — Migrate JAXTPC configs (Ra), flip `JAXTPCDataset`, dissolve `LUCiDEventSSLDataset`

**Goal.** Move the two JAXTPC configs off the dead `seg/resp/corr` modalities,
flip `JAXTPCDataset` to pimm-data, and dissolve `LUCiDEventSSLDataset` into the base
+ a LUCiD SSL config.

**Scope (`06` §4 Step 4 + §6).**
1. **Fix `__init__.py:9`** (Ra): shim imports `LUCiDDataset` from pimm-data + makes
   the `LUCiDEventSSLDataset`-successor registration explicit (the old `__init__`
   never imported `lucid_event_ssl`).
2. **Migrate `configs/detector/_base_/jaxtpc_seg.py`** `modalities=("seg",)` →
   `("step","labl")` + `label_key='pdg'`; wrap the per-stream geometric/voxel ops in
   `ApplyToStream(stream='step', …)`; replace `PDGToSemantic` with
   `RemapSegment(scheme='motif_5cls')`; terminal `Collect(stream='step', …)`.
   Confirm the `semseg-pt-v3m2-jaxtpc-5cls.py` child still builds (`in_channels=4`
   unchanged, no model change).
3. **Flip `JAXTPCDataset`** in the shim to pimm-data's `jaxtpc.py`.
4. **Dissolve `LUCiDEventSSLDataset`**: rewrite the LUCiD SSL config to use the
   `MultiModalEventDataset` base with a LUCiD `label_config` (base owns the
   holdout/min-points/aggregation that ran inline).

**Sequencing is load-bearing:** `ApplyToStream`/`RemapSegment` exist only in
pimm-data, registered only after Step 3's transform flip — so the JAXTPC migration
**must** be Step 4 (after Step 3).

**Key decisions.** **D40** (JAXTPC in scope: config migration), **Ra** (config
breakage on flip; migrate before flipping the dataset), **D35** (single-stream;
`Collect(stream=)`), **D41** (eval repro / train≡eval), **D3→D6** (`LUCiDEventSSLDataset`
dissolves into the base).

**Depends on.** Step 3 (transform flip must precede the migrated config, which uses
pimm-data transforms); Step 1 (the base + decoration the dissolved SSL config needs).

**Gate (Part 07).** **MG-03** (migrated JAXTPC semseg config builds + runs 1 step;
`coord`/`feat`/`segment`/`offset` present), **MG-04** (dissolved LUCiD SSL config
builds + 1 step; nested `sensor` stream, per-point `event_label`), **CE-05**
(train≡eval transform equality), **CE-03/CE-04** (eval-probe disjointness via
`event_identity`), **DS-12** (probe contract shape).

**Rollback safety.** Config edits + shim edit. Revert the config commits + shim
hunk; vendored files still present so JAXTPC/LUCiD fall back. PILArNet/panda/hmae
share no `_base_` with `jaxtpc_seg.py` (§6.3 grep proof) — untouched by these edits.

**Spec pointer.** `impl/06_defork_rollout_packaging.md` §4 Step 4 + §6 (Ra:
concrete `jaxtpc_seg.py` rewrite, `__init__.py:9` fix); master plan §2 Step 4 + §4.

### Step 5 — Delete vendored files

**Goal.** Delete the now-dead vendored data-layer files in **one commit** so a
single `git revert` of *that commit* restores the entire vendored tree.

**Scope.** Delete the `06` §3 DELETE list:
`pimm/datasets/{anchors,builder,defaults,detector_transforms,jaxtpc_dataset,
lucid_dataset,lucid_event_ssl,pilarnet,transform,utils}.py`, `pimm/datasets/readers/`,
and `pimm/datasets/preprocessing/` (if data-layer-only). The shim already points
everything at pimm-data; this commit only removes dead source. Plus the packaging
edits (`06` §7: submodule + `environment.yml` `-e ./libs/pimm-data` line +
`train.sh:228` snapshot/SHA + torch pin) land around here.

**Key decisions.** **D33** (delete vendored last; the grep gate), **D5** (de-fork
mechanical bucket), **D32** (aggregation deferred — `AggregateBySensor` built-unused).

**Depends on.** Steps 2–4 (shim fully points at pimm-data; both detectors run).

**Gate (D33 / Part 07).** All of:
- **MG-08** — `grep -rn 'seg\|resp\|corr\|output_mode' configs/` clean for
  *modality/reader* usage (the substring `seg` may survive in `semseg`/`segment` —
  gate on `modalities=`/reader usage, not the raw substring).
- `jaxtpc_seg.py` migrated (no `modalities=("seg",)`).
- `pimm/datasets/__init__.py` stale `lucid_dataset` import gone.
- Full Part-07 parity suite green (MG-02..MG-07 + the whole §4 matrix).
- **≥1 full PILArNet run soaked end-to-end** (manual gate — out of CPU-CI scope).

**Rollback safety.** **Single `git revert <delete-commit>`** restores every vendored
file. Because the shim (Steps 2–4) and the delete (Step 5) are **separate commits**
and `_reregister` is membership-guarded: reverting the delete brings the files back,
the shim still points at pimm-data, and re-registration is harmlessly skipped (for a
true rollback, also revert the shim). This is the single-revert-safe property.

**Spec pointer.** `impl/06_defork_rollout_packaging.md` §4 Step 5 + §3 (DELETE list)
+ §7 (packaging) + §10 (risks & the single-revert-safe property); master plan §2
Step 5; Part 07 §6 Step-5 gate.

---

## 4. Phase B/C — shard/event filtering (gated on D48 decisions)

> **STATUS: SCOPE OWED BY USER (D48, open).** This is a real gap and the user's
> actual ask, built on Phase A's joint index — but **how far to take it now is not
> decided.** Phase A proceeds regardless; **nothing past Phase A's bug fix starts
> here until D48's two questions are answered.** This section presents the options
> neutrally and marks them owed. Do not assume scope.

**Goal.** Let a user choose which shard files and which events are included when
constructing a dataset, uniformly across any modality combination — plus a
within-config multi-run axis and a reproducible manifest contract.

**The two owed decisions (D48 — gate the scope, NOT Phase A):**
- **G1 — how far now?** One of: Phase A only / **A + B1 shard-selection** (the
  recommended first cut; covers doraemon) / A + full B / + CLI (C).
- **G2 — multi-run mechanism?** One of: **`runs=` one dataset** (recommended; now a
  no-structural-change `sources=` extension per D45) / ConcatDataset is fine (defer
  B2) / one run at a time.

**Scope options (present, do not pre-commit) — handoff §5 Phase B/C:**
- **B1 — shard-tag selection.** `include_shards`/`exclude_shards`/`shard_filter` at
  the dataset level. Unit = shard **tag** (`'0054'`) — the only identifier invariant
  across modalities (filename is per-modality; path leaks layout). Scope per-run.
  Satisfies the doraemon shard-selection request. *(D45: shard-tag is an orthogonal
  within-run sub-selector, like `volume`.)*
- **B2 — multi-run axis.** `runs=` first-class (`['run_…', …]` or `'*'`), replacing
  the ConcatDataset workaround; extend `get_data_name` to carry run identity.
  *(Gated on G2.)* *(D45: a `run` is the multi-run/shard-union axis — same physics,
  one label, run-as-identity — distinct from `sources=` mixture; doraemon's 4
  `run_*` dirs are NOT 4 sources.)*
- **B3 — `event_filter` registry.** Registered-class form
  (`dict(type='MinDeposits', n=10)`) for config round-trip via `build_dataset`, plus
  a raw-callable escape hatch. Signature `(shard_tag, event_idx, attrs_dict) ->
  bool`; called O(N) at `__init__`, never in `__getitem__`. Built-ins:
  `MinDeposits`, `MaxDeposits`, `MinHits`. *(Registered form is the shareable one —
  callables don't survive the dict-config round trip.)*
- **B4 — manifest include/exclude.** `manifest=` (YAML/JSON, `mode: include|exclude`,
  list of `{run, shard, event_idx}`) — the reproducibility / collaborator-handoff
  contract; the right home for the overflow CSV (converted once via CLI, not
  auto-parsed). *(D47: the include/exclude curation contract is the real gap; the
  `resolved_manifest` snapshot half is superseded by D41.)*
- **C — tooling (separate, deferrable).** `pimm-data audit <root>` (per-modality
  shard counts, cross-modality gaps, overflow summary, draft manifest);
  `pimm-data manifest from-overflow <csv>`; a shared `pimm_manifest.py` schema
  vendored in both pimm-data and the producer. Keep curation/audit out of the
  runtime import path.

**Key decisions.** **D45** (three distinct axes — `sources=` ≠ multi-run/shard-union
≠ shard-tag — do NOT collapse; multi-run + shard-tag sub-selector added to the base,
Phase B, after the base lands), **D47** (manifest-as-INPUT include/exclude + overflow
CSV + CLI = Phase B/C, deferred; distinct from the internal `.npz` cache), **D48**
(OWED — the G1/G2 scope gate).

**Depends on.** The de-fork base (Step 1's `MultiModalEventDataset`, which factors
Phase A's joint index up) — the multi-run/shard-tag sub-selectors are added *to the
base*, so Phase B lands **after** the base. And on **D48 being answered.**

**Gate (Part 07).** No tests defined yet (feature unscoped). When scope is fixed,
each chosen item needs: an `__init__`-time filter test (O(N), never in
`__getitem__`); a config round-trip test for the registered `event_filter` form
(B3); a manifest include/exclude resolution test (B4); and the cross-modality
alignment invariant (A5-style) must still hold under filtering. Forbidden combos
(`('labl',)`, `('sensor','labl')`) stay FORBID; sensor/hits-only combos have no step
facts to filter on (NEEDS-NEW-API if wanted).

**Rollback safety.** Additive feature on the base; each knob defaults off (no filter
= today's behavior). Per-item PRs; reverting any one restores the base's
unfiltered selection.

**Spec pointer.** `shard_event_filtering_handoff.md` §5 Phase B/C (B1–B4, C with
reasons), §6 (the owed decisions), §7 (design rationale: shard-tag is the
cross-modality-invariant unit; raw callables break config round-trip), §8 (source
refs incl. `jaxtpc.py:220-224` `_modality_root` extension point for `runs=`).
Decision text: engagement plan Part VIII D45, D47, D48.

---

## 5. Out of scope now

These are designed-for but **not built** in this roadmap. Listed so the boundary is
explicit and nobody mistakes a locked seam for a deliverable.

- **Track B — densify / noise (D1, D33).** `Densify` / `AddIntrinsicNoise`,
  **wire-TPC-only**, LUCiD ops on CPU. Re-homed to `gpu_batch_transforms_plan.md`,
  sequenced after the base + readers land — a **separate track**, not part of the
  de-fork series. Part 06 only records that the de-fork does not block it. The base
  + readers landing here are the prerequisite Track B builds on.
- **Multi-stream-in-batch (D19, D23, D39).** Namespaced `multistream_collate_fn` +
  cross-stream joins + model-side primary/aux adapter. **Design-for-extension, NOT
  built (D39):** the four seams are **locked now** (`impl/05` §3.3) so the future
  build is a bounded addition, never a rewrite — (1) nested dataset output
  (never flatten), (2) per-event label decoration (streams self-contained, joins
  resolved before batching), (3) stream-aware collate structure (operate on a named
  stream even with one), (4) stable `event_identity` for cross-stream alignment.
  Near-term collate is a byte-identical single-stream REPLACE (D35). Flip to
  build-now only if a concrete multi-stream model is committed (D39).
- **Aggregation (`AggregateBySensor`, D32).** Built-but-unused, default off; struck
  from D16's "build now." Revisit when a PMT-merged-input task appears.
- **Persisted-to-disk filter index, auto-parsing `logs/overflow_*.csv`** (handoff
  §7). Deferred until needed / explicitly converted via CLI.

---

## 6. Owner sign-off checklist

Tick each box **in order** — each is a landable, revertable big step with a concrete
gate. The invariant **"PILArNet/panda/hmae training works"** holds at every box.

**Phase A (lands first, standalone PR on `jaxtpc-loader-codec-opt`):**

- [ ] **Phase A — joint index + min-points fix.** A1 meta-cache, A2 joint
  cross-modality index, A3 volume-aware min-points + raise-on-missing-modality, A4
  length-mismatch warn + `strict_lengths`, A5 cross-modality regression test.
  **Gate:** A5 (+ gap-in-one-modality variant) green; existing JAXTPC suite passes.
  *(D42–D44, D46)*

**De-fork (sequential; Part-07 gate re-run before every advance):**

- [ ] **Step 0 — Test matrix (no code move).** P07-FIX fixture extensions + full
  matrix scaffolded; A5 folded in. **Gate:** P07-FIX merged; TR/DS/RD/LD/CE green on
  fixtures (parity tests pass or skip with golden pair green). *(D33, D41, D43)*
- [ ] **Step 1 — Additive build in pimm-data (no pimm change).** 01 transforms (+
  prefix-match + Rb pilarnet v3 merge) ∥ 03 readers → 02 base (factors A2 up) → 04
  decoration → 05 collate/eval. **Gate:** Step-0 suite green on real code; MG-01
  byte-identical; Rb v1/v2 parity held. *(D6–D9, D11, D25–D31, D35, D37–D42)*
- [ ] **Step 2 — Re-export shim.** Membership-guarded re-register into pimm's
  registries; byte-identical surface + PILArNet first. **Gate:** MG-02 (PILArNet
  1-step through shim), MG-05, MG-06, CE-09. *(D17, D18, Rf)*
- [ ] **Step 3 — Flip transforms + PILArNet.** Shim points TRANSFORMS + PILArNet at
  pimm-data. **Gate:** MG-07 — **identical first-batch tensors vs vendored**; TR-*
  parity green/golden-paired. *(D33, Rb)*
- [ ] **Step 4 — Migrate JAXTPC configs (Ra), flip `JAXTPCDataset`, dissolve
  `LUCiDEventSSLDataset`.** `jaxtpc_seg.py` → `step`/`labl` + `ApplyToStream` +
  `RemapSegment` + terminal `Collect`; LUCiD SSL → base + config. **Gate:** MG-03 +
  MG-04 (both build + 1 step); CE-05 (train≡eval); CE-03/CE-04 (probe disjointness);
  DS-12. *(D40, D41, Ra)*
- [ ] **Step 5 — Delete vendored files.** One commit, single-`git revert`-safe;
  packaging edits (submodule + `-e` install + `train.sh` SHA snapshot + torch pin).
  **Gate (D33):** MG-08 grep clean; `jaxtpc_seg.py` migrated; stale import gone; full
  parity suite green; **≥1 soaked full PILArNet run** (manual). *(D33, D5, D32)*

**Phase B/C (do NOT start until the two D48 decisions below are answered):**

- [ ] **⚠ DECISION OWED — D48 G1: how far now?**
  Phase A only / **A + B1 shard-selection (recommended first cut, covers doraemon)** /
  A + full B / + CLI (C).
- [ ] **⚠ DECISION OWED — D48 G2: multi-run mechanism?**
  **`runs=` one dataset (recommended; no-structural-change `sources=` extension)** /
  ConcatDataset is fine (defer B2) / one run at a time.
- [ ] **Phase B/C — shard/event filtering** (scope per G1/G2 above). Built on Phase
  A's joint index, added to the de-fork base; each knob defaults off. **Gate:** per
  chosen item — `__init__`-time filter test, config round-trip for `event_filter`
  (B3), manifest resolution (B4); A5-style cross-modality alignment must still hold
  under filtering. *(D45, D47, D48)*

**Explicitly out of scope (no sign-off here):** Track B densify/noise (D1, separate
track, `gpu_batch_transforms_plan.md`); multi-stream-in-batch (D39, seams locked,
build deferred); `AggregateBySensor` (D32, built-unused).
