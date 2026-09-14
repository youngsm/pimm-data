# pimm-data data layer — documentation index

This is the single entry point to the pimm-data data-layer docs. It is
**navigational only** — it tells you what each document is and the order to read
them in. It does not re-derive any design (the "why" lives in `DESIGN.md`, the
"what/when" in `ROADMAP.md`, the "how" in the `impl/0N` specs, the immutable
record in the engagement plan's decision log).

**What is being built:** pimm-data is taking ownership of the entire data layer
(datasets, transforms, collate, readers, registries) so it can drive broad
multi-task work (SSL, semseg, instance/panoptic, vertex/energy/PID/containment)
on **both** detectors — water-Cherenkov (LUCiD/WAND) and wire-TPC (JAXTPC).
pimm keeps only trainer/DDP/model/hook code. The keystone is a
`MultiModalEventDataset` base.

---

## 1. Start here (reading order for a new developer)

1. **README** (this file) — orientation + the doc map below.
2. **`ROADMAP.md`** — the plan: phased rollout + the sign-off checklist. Read it
   to know *what lands when* and *what is gated on what*.
3. **`DESIGN.md`** — the why: authoritative design + the decisions behind it.
   Read it to understand the architecture before touching code.
4. **`impl/00_index.md`** — the build map: per-part index, dependency graph,
   build order, cross-part contracts, decision→part traceability, definition of
   done. This is the hub for the implementation specs.
5. **the relevant `impl/0N_*.md` spec** — the implementation-ready spec for the
   part you are about to build. Pick it from `impl/00_index.md` §2.

If you only need the *immutable rationale* for a specific structural choice, go
straight to the decision log: `engagement_plan_transform_dataset_placement.md`
**Part VIII** (D1–D48).

---

## 2. Canonical vs source/archive

| Doc | Class | One-liner |
|---|---|---|
| `REDESIGN.md` | **CANONICAL** | The current data-layer contract: flat-prefixed batch, per-part `_roles`, map→reduce→map, `Apply(on=)` / `Collect(parts=)`, `MultiCrop`, two row-spaces. **Supersedes the streams/collate design in `impl/05` (streams were removed).** |
| `CAMPAIGN.md` | **CANONICAL** | Cross-dataset challenge matrix (JAXTPC/LUCiD/Optical) — `dataset × task → (modalities, labels, transforms, Collect)` + config tracker. Consolidates LUCiD's `Tasks→files` and pimm's `Task→config`. |
| `DESIGN.md` | **CANONICAL** | Authoritative design + decisions (the "why"). *Being written now; if briefly absent, it is still canonical.* |
| `ROADMAP.md` | **CANONICAL** | Phased plan + sign-off checklist (the "what/when"). *Being written now; if briefly absent, still canonical.* |
| `impl/00_index.md` | **CANONICAL** | Build map: part index, dep graph, build order, cross-part contracts, DoD. Read first among the impl set. |
| `impl/01_transforms.md` | **CANONICAL** | Transform merges + `index_operator` prefix-match (D11/D25/D29/D31/D34/D38). |
| `impl/02_dataset_base.md` | **CANONICAL** | `MultiModalEventDataset` + `TestModeMixin`: selection, holdout, manifest cache, `event_identity`/`split` (D6–D9/D26/D27/D30/D36/D37/D40). |
| `impl/03_readers.md` | **CANONICAL** | `read_meta(idx)→{source_event_idx, n_hits}` + `read_event` surfacing across the 8 readers (D10/D27/D40). |
| `impl/04_label_decoration.md` | **CANONICAL** | `label_config` schema + generic `_decorate_from_labl` + per-detector `fk_resolver` (D20/D22/D28/D38). |
| `impl/05_collate_streams_eval.md` | **SUPERSEDED by REDESIGN** | Single-stream collate, `Collect(stream=)`, multi-stream seams. The streams design was replaced by the flat-prefixed `_roles` contract — see `REDESIGN.md`. Eval/repro parts still apply. |
| `impl/06_defork_rollout_packaging.md` | **CANONICAL** | De-fork rollout Steps 0–5, re-export shim, submodule/pin, config migration (D5/D17/D18/D32/D33). |
| `impl/07_test_matrix_fixtures.md` | **CANONICAL** | Step-0 parity/determinism harness + `testing.py` fixture additions; the gate re-run before every flip (D33/D34/D41). |
| `engagement_plan_transform_dataset_placement.md` **Part VIII** | **DECISION-SPINE** | The immutable, append-only decision log **D1–D48** — the record of record for every structural choice. |
| `engagement_plan_transform_dataset_placement.md` (rest of file) | PROCESS HISTORY | Round log, process re-scoping, WAND scan, de-fork inventory (Part IX), task→stream→label matrix (Part X). Historical; superseded by the canonical set where they overlap. |
| `implementation_plan_pimm_data_datalayer.md` | **SUPERSEDED** | The original master build spec (§1 architecture, §2 rollout Steps 0–5, §3 component specs, §6 test matrix). **Being superseded by DESIGN + ROADMAP + impl/0N.** Keep as a detailed back-reference (the impl specs still cite its section numbers); do not extend it — new detail goes in DESIGN/ROADMAP/impl. |
| `shard_event_filtering_handoff.md` | SOURCE/ARCHIVE | Shard/event filtering + the cross-modality desync bug (D42–D48); Phase A correctness fix + Phase B/C feature spec. Folded into the canonical set as D42–D48; this file is the detailed source. |

**De-fork boundary note:** `DESIGN`/`ROADMAP`/`impl/00–07` are the canonical set
you implement against. The decision log (Part VIII) is the immutable spine — if
a canonical doc disagrees with it on a *structural* choice, the decision log
wins (D34); on a *reversible* detail, the later code-grounded spec wins. The
three handoffs and the gpu_batch plan are historical context only; trust them
for facts they captured, but the canonical set overrides where they overlap.

---

## 3. Status

The design has **converged** — D1–D48 in the decision log are all `decided`
except the one open user-owed item (D48). The near-term structure is
**single-stream-per-task** (D35): the dataset stays multi-modal/nested but each
task selects one stream via `Collect(stream=)` into the existing single-stream
collate; multi-stream-in-batch is designed-for but **not built** (D39, four
seams locked). The first thing to land is **Phase A** — a standalone
correctness bug-fix PR for the cross-modality event desync (D42/D43), *before*
the de-fork. **Two user decisions are still owed (D48):** (G1) how far to build
now (Phase A only / +shard selection / +full filtering / +CLI) and (G2) the
multi-run mechanism (`runs=` one dataset vs ConcatDataset vs one-run-at-a-time).
Phase A proceeds regardless of those. **Nothing in this series is implemented
yet** — these are specs and plans.

---

## 4. Known cross-reference caveat

The impl specs were written before the filenames were finalized, so some
in-text **"Part NN"** labels do **not** match the filenames. This is
banner-noted at the top of the affected docs (notably `impl/02_dataset_base.md`
and `impl/04_label_decoration.md`). **Match cross-references by title, not by
number.** The canonical mapping is **`impl/00_index.md` §2** (and its drift note
right after the part-index table). Section anchors (`§3.x`) are stable; only the
part *numbers* drift. Example: `02_dataset_base.md` calls readers "Part 04" but
the real file is `03_readers.md`.

---

## 5. One-line glossary (load-bearing terms)

- **`MultiModalEventDataset`** — the new pimm-data base class that owns event
  *selection* (source mixture, holdout, min-points, identity); `LUCiDDataset`
  and `JAXTPCDataset` inherit it.
- **nested per-stream** — the dataset emits `{sensor, hits, step, labl, …}` with
  each stream self-contained; it never flattens to bare top-level
  `coord`/`feat` (Seam 1).
- **single-stream-per-task** — near-term structure (D35): one task selects one
  stream → existing single-stream collate → `Point`; multi-stream is deferred.
- **joint cross-modality index** — the canonical per-event map built by
  intersecting present `event_*` keys across loaded modalities (D42/A2); fixes
  the desync that the naive `min(len(reader))` indexing causes.
- **`source_event_idx` (identity)** — the stable, file-discovered physics-event
  id (not positional `local_idx`); the join key for the joint index and the
  holdout/`event_identity` hash.
- **`label_config`** — the axis-spec map that drives the generic decorator,
  turning reader-emitted raw FKs into named `segment_*`/`instance_*`/`target_*`
  keys; new label axes are registered entries, not code changes (D38).
- **Phase A** — the standalone correctness PR (cross-modality desync fix:
  meta-cache, joint index, volume-aware min-points + raise, length-mismatch
  handling, regression test) that lands before the de-fork (D43).
- **de-fork** — moving the data layer out of the pimm fork into pimm-data, then
  re-exporting via a shim and deleting the vendored copy (impl/06).
- **Track B** — the deferred, JAXTPC-only train-time densify + noise work
  (the retired `gpu_batch_transforms_*` plans); out of scope for the impl set (D1/D33).

---

**Superseded (2026-09-13).** The GPU/batch-transform plans (`gpu_batch_transforms_*.md`) are retired: the dense path shipped, and the LArTPC forward model they assign to pimm-data now lives in helix (see helix `docs/ARCHITECTURE.md`). pimm-data keeps `densify`. Archived under `$HELIX_ARCHIVE/retirement-backups/pimm-data-docs_2026-09-13/`.
