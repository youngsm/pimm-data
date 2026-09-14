# pimm-data restructure — the grounded plan

> **Boundary note (2026-09-13).** Where this document places the LArTPC forward
> model — `AddNoise`, `Digitize`, the noise spectrum, `add_intrinsic_noise` — in
> pimm-data, that is no longer true. Those moved to **helix** (pimm-data
> `b592600` / helix `8075a53`): pimm-data keeps `densify`, which every detector
> family needs, and gave up the physics only one family applies. The transform
> registry raises on duplicate registration, so the two packages are lockstep.
> See helix `docs/ARCHITECTURE.md` §3. The design reasoning below is kept as
> written; only the ownership changed.

**Status:** FINAL PLAN — build-ready (design + phased implementation). This is the
**single consolidated plan**; it supersedes `multimodality_plan.md`,
`gpu_batch_transforms_plan*.md` (since retired — see docs/README.md), and the multi-stream sections of `impl/05`. The older docs
are retained as historical/source (decision text in `engagement_plan` Part VIII;
code-as-built in `IMPLEMENTATION-boundary-refactor.md`).

**North star:** *pimm-data delivers, per event, namespaced multi-modality records — each
modality a sparse cloud or (post-collate, on the model's device) a dense grid, with truth
labels joined directly from `labl` via per-detector FK chains and per-event targets,
aligned across modalities, with reproducible identity-based splits — for SSL + seg +
panoptic + regression across PILArNet/JAXTPC/LUCiD. It privileges no modality and owns no
fusion or model-side targets. "Done" = adding a detector is readers + an FK resolver;
adding a task is pick modalities + write a model — never touching collate or the base.*

## 0. How we got here / the governing principle

Three rounds of consults (two external loader experts + a PM persona) + reading the
actual config tree converged on one fact: **every config trained today is
single-modality sparse; there is exactly one committed heterogeneous multi-modality
consumer (C4, wire-TPC sparse-`step` + dense-`sensor`).** So:

> **Keep the single-modality sparse path dead simple (≈ today, byte-identical). Add the
> *minimum* multi-modality + dense machinery to serve C4 — which is committed & staffed
> this cycle. Cut every platform abstraction that a real combination doesn't need.**

### The real combinations (these drive everything)
- **C1** — one sparse modality, unlabeled → SSL (PILArNet, LUCiD). *Most compute.*
- **C2** — one sparse modality + per-point labels (`segment`[, `instance`]) → semantic seg / panoptic (PILArNet, JAXTPC `step`).
- **C3** — one sparse modality + per-event targets (vertex/energy/PID/probe) → regression/probe (LUCiD, panda heads).
- **C4** — sparse `step` + dense `sensor`, multi-task (seg-on-`step` + denoise-on-`sensor`) → the one heterogeneous case (wire-TPC). **Committed.**

### Explicitly CUT (do not build)
External-first general library; event-major/streaming format; MDS/WebDataset/ArrayRecord;
own-the-sampler / resumable iterator; persisted global index; versioned-schema system;
framework-neutral/JAX core; **GPU decode** (assumed never); disaggregated decode; dataset
fingerprinting; and the **label-source *registry* plugin abstraction** (we implement the
two FK chains directly). The access layer stays **map-style HDF5 + torch `DataLoader` +
`DistributedSampler`**.

---

## 1. Contract & vocabulary

- **modality** (rename of "stream"): a real per-event point cloud or grid (`step`,
  `sensor`, `hits`). Single-modality → **bare** batch (`coord/feat/offset/...` at top
  level, byte-identical to today). Multi-modality → **namespaced** `{modality:{...}}`,
  each with its own `offset`. (Bare-when-1 / namespaced-when-many.)
- **label** (was the `labl` pseudo-modality): per-point/per-event columns produced by a
  **direct FK join** from the `labl` dimension table onto a modality. `labl` is **not** a
  modality and never appears in `modalities=` or the batch.
- **target**: a supervision tensor produced by a transform or owned by the model (e.g. the
  denoise clean grid). Not a label; built model-side where possible.
- **fusion / target-construction live in pimm**, never in the data layer (fusion-agnostic).

The data layer's job, stated minimally: *given a detector + a list of modalities + a
labels spec, deliver one event's modalities — each a sparse cloud (or, post-collate, a
dense grid) with its FK-joined labels and per-event targets — aligned across modalities,
with reproducible splits.* It privileges no modality and builds no model input.

**Representation-agnostic — no modality is intrinsically dense.** Every modality
(including `sensor`) is a **sparse point cloud by default**. Densify is an *optional*
post-collate transform you attach per-modality (`ApplyToModality('sensor', [Densify, …])`)
— not a property of the modality. A model may consume `sensor` **sparse** (a point backbone
on `wire/time`/`coord`) just as legitimately as densified (CNN on the grid). So the
C4-family is `step`(sparse) + `sensor`(**sparse OR densified — model's choice**); the data
layer forces nothing and `Densify` is itself general (configurable keys + grid geometry,
so any modality could be densified). "Phase 3 (dense)" means *the optional densify
transform*, never "make sensor dense."

---

## 2. Per-combination design (concrete)

**C1 (SSL, the hot path) — unchanged.**
`Reader(modality)` → `ApplyToModality(modality, [GridSample, aug…])` →
`Collect(modality=…, feat_keys=…)` → `collate_fn` → bare `{coord,feat,offset}` → PTv3.
No joint index (single modality), no labels, no GPU stage. **Keep allocation-light and
byte-identical.**

**C2 (seg / panoptic).** C1 + `labl` reader + **direct FK decoration** → per-point
`segment` (+ `instance` for panoptic) as **raw** values → `RemapSegment(scheme=…)` in the
per-modality pipeline → `Collect(keys=['segment','instance',…])`. Joint index runs
(`step`+`labl`). Panoptic specifics in §4.

**C3 (regression / probe).** C1/C2 + per-event target carried as a **length-B tensor**
through collate (drop the per-point `event_label` broadcast — it's not a new shape, just a
scalar riding along). Identity-based holdout is the load-bearing part.

**C4 (sparse `step` + dense `sensor`, multi-task) — the committed new work.**
Workers stay **sparse** (only sparse crosses PCIe). `Collect(modalities={'step':…,
'sensor':…})` → **namespaced** batch `{step:{coord,feat,offset,segment}, sensor:{wire,
time,value,plane_gid,offset}}`. Post-collate:
`ApplyToModality('sensor', [Densify, AddIntrinsicNoise, Digitize])` in the post-collate
bucket → `batch['sensor']['dense'] = {plane_gid:(B,W,T)}` (default device = the model's, so
GPU; placement is a choice — §3B). The joint index guarantees `step`/`sensor` are the same
physics event. The denoise **target** (pre-noise grid) and all fusion live in the pimm model.

---

## 3. The access / runtime layer — keep map-style (no new machinery)

- Map-style `Dataset` (`ShardEventDataset`, `_dataset_base.py`) + readers doing
  `searchsorted` random access over per-shard present-event indices. **Keep.**
- Shuffle/shard/DDP = torch `DistributedSampler` + `set_epoch`; `persistent_workers=True`,
  `pin_memory=True`. **No owned sampler, no resumable iterator** (epoch-granular resume is
  enough; bit-exact replay isn't required — AMP/flash are nondeterministic anyway).
- **Joint cross-modality index** (`_joint_index.py`): KEEP — it's a *correctness*
  primitive (intersect present `event_*` across modalities so one global idx → the same
  physics event; without it, an `min_deposits` filter or a production gap silently
  misaligns FK joins). Fires only when >1 modality is loaded. In-process memo
  (`_shard_meta.py`, keyed `(path,mtime,size)`) stays; **do not persist an index file.**

---

## 3A. Execution model — CPU/GPU & the pre/post-collate boundary

There is exactly **one boundary: `collate_fn`.** Every unit of work reduces to which side
it is on.

- **Pre-collate = CPU, numpy, per-sample, in DataLoader workers (forked → NO CUDA):**
  decode → FK label decoration (`get_data`) → per-modality sparse transforms
  (GridSample/voxelize, augment, RemapSegment) → `Collect` tensorizes (numpy→torch CPU) +
  namespaces/bares.
- **Boundary = `collate_fn`** (runs in the worker): ragged concat + `offset`; result
  crosses worker→main via **FD-shared IPC** — cheap *because* `Collect` already produced
  tensors.
- **Post-collate = GPU, per-batch, main process:** `move_to_device` (only sparse
  ~17MB/event crosses PCIe) → densify/noise/digitize (dense ~186MB/event **born on GPU**)
  → model.

**Rules:** workers stay CPU/numpy/fork-safe (no `torch.cuda` pre-collate — dense ops are
device-agnostic but never import `torch.cuda`); only sparse crosses PCIe *in the default*;
dense is created on the model's device (born-on-GPU is the **default** for C4, not a
requirement — placement is a choice, §3B); **tensorize at `Collect`** — a
torch tensor crosses worker→main as an O(1) shared-memory FD, numpy would pickle/copy the
whole array every batch on the bottleneck CPU; framework-neutral tensorize-at-edge is cut,
so this is a strict win (`Collect._to_tensor` is the single numpy→torch point).

**Placement table:**
| element | side | required? |
|---|---|---|
| decode, FK labels, voxelize/GridSample | CPU-pre | required |
| sparse augment, RemapSegment, per-event target carry, tensorize | CPU-pre | required (remap: either) |
| densify, intrinsic noise, digitize | **post-collate, model's device** | placement is a choice — GPU default (C4), CPU for eval/no-GPU/test (§3B) |
| user per-sample transform | CPU-pre (numpy) | — |
| user per-batch transform | GPU-post (torch) | — |
| panoptic: raw `instance` emit / contiguous renumber + thing-stuff masks | CPU-pre / **model-side** | split |
| cross-modality fusion, denoise-target construction | model-side (post) | required |
| streams/views | **vertical** — straddle; post-collate runs per-stream-per-modality | — |

**Two gaps this surfaces — FIX (Phase 1):**
- **G1 — unify seeding across the boundary.** Today post-collate noise is
  content-addressed, but **pre-collate augmentation uses global `np.random` → not per-event
  reproducible.** One `derive_seed(identity, base_seed, epoch, *, tag)` used on BOTH sides
  (tag namespaces `aug:step` / `noise` / `stream:local`); **keep `rank` (default 0)** — it
  decorrelates DDP replicas; single-node no-op (do NOT delete it — §12 C-fix2); plumb a
  per-event `np.random.Generator` into the worker
  (`data_dict['_rng']`, `_`-prefixed so collate drops it) consumed by stochastic
  transforms. Closes the CPU-aug determinism hole; identical to the §11.4 seed-`tag` seam.
- **G2 — `Collect` passes `name` through UNCONDITIONALLY** (today gated on
  `stream is not None`). `name` is the single cross-boundary identity carrier; dropping it
  silently degrades all seeding to non-reproducible batch-position seeding.

**Footguns:**
- `persistent_workers=True` does NOT re-fork per epoch → epoch must be read at
  `__getitem__` (via `set_epoch`-driven state), not captured at fork, or every epoch
  replays identical augmentation.
- `pin_memory=True` required or `move_to_device(non_blocking=True)` silently serializes.
- No `torch.cuda` in any worker/per-sample transform (fork-safety).
- No coord-mutating transform (GridSample) on the `sensor` modality pre-collate — desyncs
  the densify COO from `offset` (guarded by the densify offset-total check).
- Throughput for the sparse path = **overlap** (`num_workers≈cores`, `persistent_workers`,
  `pin_memory`, `prefetch≥2`), NOT relocating sparse work to the GPU; born-on-GPU densify
  also overlaps with next-batch CPU prefetch.

---

## 3B. Densify placement is a CHOICE — one op, one wrapper, device on the runner

Densify/noise/digitize are **device-agnostic** (`dense_ops` runs on cpu or cuda; never
imports `torch.cuda`). So **born-on-GPU is the default for C4, not a requirement** —
placement is a choice. Verified sizes: dense = **186 MB/event** (10762 wires × 4321 ticks
× 4B), sparse ~17 MB → **11×**.

Three first-class modes (all the same `dense_ops` op — chosen by *bucket* × runner
*device*, no mode-specific code):

| mode | how | IPC (worker→main) | PCIe (H2D) | densify/noise compute | best when |
|---|---|---|---|---|---|
| **A — densify+noise on GPU** (C4 default) | `post_collate`, runner `device='cuda'` | sparse (FD-shared) | **sparse ~17 MB** | GPU | **CPU-decode-bound** (usual case) — only sparse crosses |
| **B — densify+noise on CPU** | `post_collate`, runner `device='cpu'` | sparse (FD-shared) | none (dense H2D only if a GPU model follows) | CPU (main) | eval / no-GPU / CI / CPU-only model (GPU would be *wrong*) |
| **C — densify on CPU → model on GPU** | densify in `pre_collate` (workers) **or** `post_collate device='cpu'`, then model `.to('cuda')` | dense as **FD-shared tensor** (cheap — `Collect` tensorizes, *not* a 186 MB pickle) | **dense ~186 MB** | CPU (workers, overlapped via prefetch) | **GPU-bound** model + CPU workers with slack — offload densify off the GPU critical path |
| offline cache | `pre_collate` densify, deterministic noise-free grid → disk | — | — | once, CPU | precompute a fixed dense grid (no per-epoch noise) |

**The rule:** pick by workload — **A (GPU)** when CPU-decode-bound (default; only sparse
crosses PCIe); **C (CPU densify → GPU)** when the GPU is the bottleneck and workers have
slack (cost is 186 MB PCIe + worker CPU — *not* IPC); **B (CPU)** for eval/no-GPU. All three
are `dense_ops` placed via the bucket (`pre_collate`/`post_collate`) + the runner `device=`.
Implementation note: mode C (pre-collate densify) needs `collate` to stack per-plane grids
`(W,T)→(B,W,T)` — `default_collate` handles same-shape tensors, so no collate change.

**On the dense ops — two placements of ONE physics, NOT a twin to delete (corrected):**
An earlier draft said "delete the numpy twins (`Densify`/`AddNoise`/`Digitize`)" as the
"real win." **Reversed after reading the code** — that was a misread of how much is shared:
- The numpy `Densify`/`AddNoise`/`Digitize` (`detector_transforms.py`) ARE the **pre-collate
  `scope='sample'` placement** (in-worker, nested-by-label representation) — mode C / offline
  cache above. They are exercised end-to-end in `test_noise.py` and are the **oracle** for
  the torch path in `test_batch_transforms.py`.
- `dense_ops.{densify,add_intrinsic_noise,digitize}` is a **necessary device-specific port**
  (the GPU path *must* be torch; numpy fancy-indexing / a numpy Generator don't run on CUDA),
  NOT a removable duplicate. The genuinely shared math is ~2 lines (scatter) + ~3 (digitize);
  the bulk of each side is representation+validation with no cross-side equivalent.
- Deleting the numpy side would break the suite, remove the worker-CPU placement, and *add* a
  nested→COO shim — more complexity, not less. **Keep both.** The standing guarantee that the
  two placements agree bit-for-bit is `tests/test_dense_collapse.py` (single-event
  `dense_ops.densify(B=1)` == numpy `Densify`; `dense_ops.digitize` == `noise.digitize`).
- **One wrapper:** use **`ApplyToModality` in both phases** — do **not** add a separate
  `BatchApplyToModality` (`detector_transforms.py:43`).
- **Device on the runner, not the stage:** config picks the *bucket*
  (`pre_collate`/`post_collate`); `apply_batch_transforms(…, device=)` picks the *device*.

**Don't:** add a "born-on-GPU" assert (breaks CPU eval/no-GPU/tests) — assert only
**device-consistency** (inputs + `offset` share a device), never CUDA-residency. Don't
build a per-node device/phase auto-router; two buckets + a runner `device=` is enough.
Reject the "flat pipeline with a `collate` marker" model — it hides the per-sample/
per-batch + numpy/torch + process-barrier reality and invites CUDA-in-fork / silent-pickle
footguns. Keep two physical phases, one interface.

**Caveat:** with `coherent=True`, coherent noise is drawn on host numpy then copied
per-event-per-plane, so "only sparse crosses PCIe" isn't literally true in that mode (still
far cheaper than pre-collate dense). Incoherent-noise device-specificity (CPU vs CUDA RNG)
is pre-existing and orthogonal — the aug-vs-physics-target decision (§5/§10), not placement.

---

## 4. Labels — direct FK joins, raw out, panoptic-ready

Implement the **two FK chains directly** (no plugin registry):
- **JAXTPC (value-keyed):** `step`: `deposit_to_track → track_{col}` keyed by `track_ids`
  (searchsorted, per volume). `hits`: `group_id → group_to_track_v{N} → track_id →
  track_{col}`.
- **LUCiD (positional):** `hits`: `particle_idx → per_particle.{col}`. `step`: extra hop
  `track_idx → per_track.particle_idx → per_particle`.

Keep the generic `_label_decorate.py` gather (`gather_with_fill`, positional + keyed
kinds; `fill=-1`; widen dtype for bool/unsigned per F3) — that's the *mechanism*, not a
registry. Add a **`labels=` config** and **auto-load** `labl` when labels are requested +
a decoratable modality (`step`/`hits`) is present; drop `labl` from `modalities=` (the
`('labl',)`/`('sensor','labl')` rejects become structurally impossible).

**Raw out, scheme downstream:** the decorator emits raw `segment`/`instance`/`target_*`;
`RemapSegment(scheme=…)`/level-selection is a downstream transform, so changing the
chosen label needs no re-read. ⚠ Ordering footgun: `GridSample`-on-raw-then-coarsen ≠
coarsen-then-`GridSample` (majority-vote differs) — document the default; let configs
reorder.

**Panoptic (C2):**
- Co-emit `segment` (semantic) + `instance` (raw id), row-aligned, both carried through
  N-changing transforms via the `index_operator` `segment_*`/`instance_*` prefix-match
  (D25 — load-bearing; keep, and update its key list for `plane_gid`).
- **Disambiguate `-1`:** today `-1` = "unresolved FK"; panoptic also needs "stuff / no
  instance". Use distinct sentinels (unresolved → `-1`/ignore; stuff → a dedicated value).
- **Instance keying:** for cross-modality consistency (future fusion), prefer keying
  instance on the shared **track id** rather than per-plane `group_id`. (Decide at C2/C4
  build; default track-keyed.)
- **Target prep stays model-side:** contiguous renumber + thing/stuff masks =
  `InstanceParser` in pimm (per the ADR). The data layer emits raw instance ids.

---

## 5. Dense path (C4) — device-on-runner (GPU default), modality-scoped, committed

- Keep `dense_ops.py` (densify/add_intrinsic_noise/digitize, device-agnostic, never
  imports `torch.cuda`) and the runner `apply_batch_transforms` (move-to-device + content
  seeds + stages). **Rename the `densify` param `plane_id` → `plane_gid`** to match the
  batch key.
- Use **`ApplyToModality` in the post-collate phase too** (one wrapper, both phases — do
  **not** add a separate `BatchApplyToModality`). **Delete the numpy `Densify`/`AddNoise`/
  `Digitize` twins** in `detector_transforms.py`; thin stages call the single
  device-agnostic `dense_ops` on whatever-device tensors. **Device is a runner arg**
  (`apply_batch_transforms(device=)`), not baked into stages — GPU default for C4,
  `device='cpu'` for eval/no-GPU (§3B). Runner unchanged (`move_to_device` recurses; seeds
  read top-level `name`).
- **Post-collate on the model's device is the C4 default** (PCIe: only sparse crosses);
  pre-collate dense is allowed only for offline caching (§3B). Per-plane separation bounds
  VRAM; B-chunking deferred until measured.
- **Noise:** content-addressed per-event seed via the unified `derive_seed` (drop `rank`;
  §3A G1). **Default: noise is augmentation** (device-specific incoherent OK). Make
  incoherent device-independent (counter-based RNG) ONLY if input-noise must be reproducible
  across placement modes (CPU↔GPU, §3B A/B/C) or across eval. The denoise **target** (clean
  pre-noise grid) is device-independent regardless.
- **pimm side (C4 model):** a multi-modality model — `Point(batch['step'])` (sparse
  backbone) + a per-plane CNN/UNet on `batch['sensor']['dense']` (⚠ U/V/Y planes are
  ragged-width — design the dense backbone for that), fusion + multi-task heads
  (seg-on-step CE + denoise-on-sensor recon), denoise target built model-side.
  `gpu_transforms = {'sensor': dict(coherent=…, incoherent=…, n_bits=…)}` per-modality;
  `run_step` builds `BatchApplyToModality` stages; eval runs the **same stages minus the
  stochastic noise** (train≡eval).

---

## 6. Load-bearing invariants — DO NOT regress

From the real-data findings (F1–F17) and the test matrix — the restructuring must keep
these green:
- **3-tuple identity** `(config_id, file_index, source_event_idx)` for holdout (F1;
  supersedes any 2-tuple); resolution vector→`global_event_offset+event_num`→`event_num`.
  ⚠ The codec transcode (§7) **must preserve these attrs** — add a test.
- **Joint-index alignment** (D42/A5): same `source_event_idx` across loaded modalities for
  every served idx under `min_deposits>0`, gap-in-one-modality, `volume=N`.
- **Cross-modality FK consistency** (testing.py): `labl.deposit_to_track[i] ==
  hits.group_to_track[hits.deposit_to_group[i]]`, etc.
- **Gap- + dangling-shard tolerance** (F6/F17): present-key indexing; `open_event_files`
  skips dangling symlinks; contributing-but-unopenable shards still raise.
- **GridSample sum in wide accumulator** (F2); `volume=` prunes bridges (F13); cache
  fast-paths (F15/F16).
- **Collate byte-identity** for single-modality; **`Collect` last + tensor output**
  (FD-IPC for `num_workers>0`).
- **Content-addressed seeding** stable per event across batch/worker/resume.

### Reconciliations (use code-as-built)
- Real names: base = `ShardEventDataset` (`_dataset_base.py`, reader composition);
  `_joint_index.py`, `_label_decorate.py`, `_shard_meta.py`. **`MultiModalEventDataset`
  (`multimodal.py`) is a live, *separate* layer** (selection/holdout/source-mixture wrapper
  over the per-detector datasets) — keep both (§12 C-fix1).
- `DefaultDataset`/`ConcatDataset`/`TestModeMixin` (D30) are **dead** — don't carry them.
- **Single shared `TRANSFORMS` registry** (ADR §2) — not separate-registries/re-register.
- pimm-data owns readers/datasets/joint-index/decoration/transforms/collate/registry +
  the dense runner; pimm owns `Point`/packing/SSL-gens/`InstanceParser`/DDP/hooks/fusion.

---

## 7. The one real perf lever

Loading is CPU-decompression-bound; **no format change helps without GPU decode.** The
boring win: transcode shards off **gzip → blosc-lz4 / zstd** (≈4× read; `scripts/
transcode_codec.py` exists) + tune `num_workers` to cores, `persistent_workers`,
`pin_memory`, `prefetch_factor`. ⚠ **Add a test that the transcode preserves the identity
attrs** (§6) so splits don't silently move. Set a coarse throughput/GPU-util regression
check on the fixture.

---

## 8. Phased plan

| Phase | Deliverable | Repos | Ships alone? | Gate |
|---|---|---|---|---|
| **0 — Rename** | `stream→modality` (`ApplyToModality`, `Collect(modality=)`), both repos, hard, no aliases. Mechanical, full suite green. | data + pimm | yes | — |
| **1 — Labels + seeding** | `labl` out of `modalities=`; `labels=dict(key=,scheme=)` + auto-load; **direct FK joins**; raw out + `RemapSegment` downstream; per-event scalar through `Collect` (drop per-point broadcast); panoptic `segment`+`instance` co-emit + `-1` disambiguation. **+ G1** unified `derive_seed(…, tag)` (drop `rank`) + per-event worker `_rng`; **+ G2** unconditional `name` pass-through (§3A). Parity vs current decoration; migrate pimm seg/panoptic configs. | data (+ pimm cfg) | yes | 0 |
| **2 — Namespaced Collect** | `Collect(modalities={…})` (projecting, fresh-dict, namespaced); single-modality `Collect` unchanged (bare, byte-identical). collate verified unchanged. | data | yes | 0 |
| **3 — Dense per-modality** | one `ApplyToModality` used post-collate (no separate `Batch*` class); **delete the numpy densify twins** (call `dense_ops`); **device on the runner** (`device=`, GPU default / CPU for eval); **device-consistency assert** (not CUDA-residency); `dense_ops.densify` param `plane_id→plane_gid`; noise seed (`rank` removed; aug-vs-target decided). | data | yes | 2 |
| **4 — C4 model (committed)** | pimm multi-modality model: sparse `step` backbone + ragged-plane dense `sensor` backbone + fusion + multi-task heads (seg + denoise); `gpu_transforms` per-modality; `run_step` + eval-minus-noise; the real wire-TPC denoise config. | pimm | no | 1–3 |
| **5 — Perf + docs** | codec transcode + identity-preservation test; throughput regression check; consolidate docs (this file canonical; archive the rest). | data | yes | 1–4 |

Order: 0 → (1 ∥ 2) → 3 → 4 → 5. Phase 1 is the highest single-modality-impact cleanup;
Phases 2–4 deliver the committed C4.

### Implementation status (branch `restructure-data-layer`)
- **Phase 0 — DONE.** `stream→modality` rename, both repos; suite byte-identical (PF1).
- **Phase 1 — DONE (labels + G2).** `labels=` label-source on JAXTPC + LUCiD (additive,
  parity-verified); `jaxtpc_seg` config migrated; G2 unconditional `name`. *G1 deferred*
  (start non-reproducible); *legacy `labl`-in-`modalities` hard-rejection deferred* as
  cleanup.
- **Phase 2 — DONE.** Namespaced `Collect(modalities={…})`; single-modality bare byte-
  identical; collate unchanged.
- **Phase 3 (data-layer) — DONE.** Dense stages take `modality=` (born at
  `batch['sensor']['dense']` on the namespaced batch); device-consistency assert; the
  "noise drift" was a test-infra `tools` collision (fixed) — **no port drift**, parity to
  9.5e-7. *Reversed (was "deferred — delete the numpy twins"):* the numpy
  `Densify`/`AddNoise`/`Digitize` are the **pre-collate placement** (oracle + worker-CPU
  path), not removable duplication — the torch `dense_ops` is a necessary device port. Kept
  both; `tests/test_dense_collapse.py` is the standing cross-placement equivalence guard. See
  §3B. *Cosmetic-deferred:* `plane_id→plane_gid` param rename.
- **Uniform user-transform path — DONE.** Post-collate `apply_batch_transforms` accepts bare
  user callables (no `seeds=`/registration needed), matching the pre-collate `Compose`;
  `scope='sample'|'batch'` marker + `Compose` build-time fence places/guards them
  (`tests/test_user_transforms.py`). This is the concrete "user transform inside or on GPU"
  capability (§11.3). *Still owed:* per-event rng plumbing (G1, deferred) + written
  key/length contract.
- **Phase 5 (data-layer) — DONE.** Transcode identity-preservation test (F1).
- **Phase 4 — NOT STARTED (team-driven research).** The C4 fusion model + the wire-TPC
  denoise config live in pimm and require model/physics decisions (sparse-vs-dense sensor
  branch, ragged U/V/Y backbone, fusion, multi-task heads) + the real data — out of scope
  for a mechanical data-layer pass. The data layer now provides everything it needs:
  namespaced batch, sensor consumable sparse **or** densified, per-modality densify, and
  parity-verified noise. The trainer wiring (`run_step` building `modality=`-scoped stages)
  is small and follows the model.
- **Suite:** 324 passed / 6 skipped / 0 failures, no deselects.

---

## 9. Test matrix (must-stay-green + new)
- **Rename:** suite green; no `stream`/`ApplyToStream` left (grep gate).
- **Single-modality byte-identity:** C1/C2/C3 batches identical pre/post change.
- **Labels:** decoration parity vs current (step `segment` via deposit; hits
  `segment`+`instance` via group); `modalities=('step',)+labels=` decorates step; `sensor`
  undecorated; `'labl'` in `modalities=` errors; panoptic semantic+instance carry through
  GridSample/crop; `-1` vs stuff distinct.
- **Namespaced collate:** per-modality offsets; junk (`raw`/other modalities) dropped;
  `name`/`split` top-level; bare-when-1.
- **Dense (C4):** densify into `batch['sensor']['dense']`; `step` untouched;
  **device-consistency** (densify inputs + `offset` share a device; CPU result == CUDA) —
  **no CUDA-residency assert**; noise reproducible per event.
- **Invariants:** joint-index alignment (fails on HEAD without the fix); FK consistency;
  3-tuple holdout golden; dangling-shard build; transcode preserves identity attrs.

---

## 10. Decided defaults (overridable at build time)
- **Noise = augmentation** (device-specific incoherent OK). Flip incoherent to
  device-independent ONLY if input-noise must be reproducible across placement modes
  (CPU↔GPU) or eval. (§5, §3B)
- **Panoptic instance keyed on `track id`** (cross-modality consistent). Sentinels:
  unresolved-FK → `-1`/ignore; stuff/no-instance → a dedicated value (distinct from `-1`). (§4)
- **GridSample ↔ RemapSegment ordering:** default coarsen-before-voxelize when
  majority-at-coarse matters; configs may reorder. (§4)
- **Deferred to Phase 4 (pimm model design):** dense backbone for ragged U/V/Y planes
  (per-plane CNN vs shared). The only genuinely open item. (§5)

---

## 11. Future extension seams (designed-for, NOT built now)

Two ambitions to keep additive. The boring design already admits both; we preserve a
couple of cheap seams now so neither becomes a rewrite.

### 11.1 A third orthogonal axis (vocabulary)
The `stream→modality` rename frees the word **"stream"** for a new meaning. Three
orthogonal axes:
- **modality** — physical channel (`step`/`sensor`/`hits`); `modalities=`; multi → namespaced `{modality:{…}}`.
- **stream** — a named *view/branch*: its own transform sub-pipeline (+ modality subset + label/key selection) over the SAME event; multi → outer `{stream:{…}}`.
- **transform provider** — built-in (`dict(type=)`) or a user callable; orthogonal, any slot takes either.

Full future output `{stream:{modality:{keys}}}`, with degenerate collapses (1 stream →
drop the level; 1 modality → bare). **collate's Mapping-recursion already packs every
nesting depth — no collate change for either ambition.**

### 11.2 Ambition 1 — multiple streams (named transform-views over one event)
A stream = a sub-pipeline (`ApplyToModality…` + `Collect`) run over the same
`get_data(idx)`, tagged. Streams may differ in transforms, modality subset, *and*
label/key selection (e.g. A supervised, B unlabeled). Generalizes pimm's
`MultiViewGenerator` into a declarative data-layer construct — "anyone who wants different
transforms on the same data."
```
transform = dict(type='Streams', streams={
  'global': [ApplyToModality('step',[aug_heavy]), Collect(modality='step', feat_keys=…)],
  'local':  [ApplyToModality('step',[aug_light, crop]), Collect(modality='step', feat_keys=…)],
})  # → {'global':{coord,feat,offset}, 'local':{coord,feat,offset}}
```
**Seam to preserve now (cheap):** make the content-addressed seed accept an optional
**tag** — `seed(identity, base_seed, epoch, tag=stream_name)` — so per-stream augmentation
is reproducible-but-distinct. Then `Streams` is a pure additive wrapper. (Not built now;
pimm's view-generators cover SSL today.)

### 11.3 Ambition 2 — user-provided transforms per modality
A user supplies their own transform — a bare callable OR the standard `dict(type=)` —
scoped to a modality, and it's applied. ~80% already supported (a transform IS
`callable(dict)->dict`; `ApplyToModality` already composes).

**BUILT now (the uniform-callable path — the core of this ambition):**
- **Both sides accept a bare callable.** Pre-collate: `Compose` already did
  (`transform.py`). Post-collate: `apply_batch_transforms` now inspects each stage and calls
  `stage(batch)` when it doesn't take `seeds=` — so a user's GPU op `fn(batch)->batch`
  (lambda, `nn.Module`, pre-built instance) is first-class with **no** `seeds` plumbing and
  **no** registration, exactly like the pre-collate side. (`_accepts_seeds`,
  `tests/test_user_transforms.py`.)
- **`scope` is the single declaration that places a transform.** A transform may set
  `scope='sample'` (default — sees one event, pre-collate `map`) or `scope='batch'` (needs the
  collated set, post-collate only; the `Batch*` stages carry it). `Compose` **fences** a
  `scope='batch'` entry at build time (it cannot run per-event in a fork). This is the bridge
  between the *conceptual* unification (transforms differ only in scope) and the *physical*
  two phases (§3A): scope declares, the two-phase placement + fences enforce.
- Still owed: the per-event rng plumbing (G1, deferred) and the written key/length contract
  (below). The GPU bare-callable does NOT need a `BatchApplyToModality` wrapper — the runner
  tolerates it directly.

**Required ON THE USER:**
- A callable `f(sub)->sub` where `sub` is the modality's **numpy** dict (`coord (N,C)` +
  named per-point/feature arrays); mutate + return. `ApplyToModality('step',
  transforms=[f, dict(type='GridSample',…)])` — user callables + built-ins mix freely.
- **N-changing transforms** (crop/voxel/dropout) must keep ALL per-point arrays
  length-consistent — either index them all, or follow the convention (operate on `coord`,
  name new per-point keys `segment_*`/`instance_*`/`target_*`, declare `index_valid_keys`)
  so the `index_operator` prefix-match carries them. *The one real burden.*
- **Stochastic + reproducible** → consume the provided per-event rng, not global `np.random`.
- **Per-sample = numpy/CPU** (workers, fork-safe, pre-`Collect`); **per-batch = torch/GPU**
  (post-collate `BatchApplyToModality`, signature `(batch_mod, *, seeds)`). Two distinct
  extension points.
- Registration optional (register a class only for the round-trippable `dict(type=)` form).

**Required of US:**
- `Compose`/builder **accepts a bare callable** alongside `dict(type=)` (trivial).
- Document the transform **Protocol** + the per-point-key/`index_valid_keys` convention +
  dtype/shape contract (lightweight; no schema system).
- **Plumb a per-(event[,modality,stream]) rng** into stochastic transforms — the only
  nontrivial piece; also closes the current CPU-augmentation reproducibility gap (today it
  leans on global RNG).
- Optional first-call validation (keys/lengths) → clear error vs silent misalignment.
- Same for the GPU path (`BatchApplyToModality` accepts user callables).

**Caveats:** N-change misalignment is the main footgun (→ length-consistency check); the
rng-plumbing is the one real new machinery (worth it — fixes our own determinism too);
numpy(per-sample) vs torch(per-batch) must be stated or users mis-place transforms. Since
external-first is CUT, this targets *internal* researchers (custom aug without forking the
core) — valuable, no public-API-stability obligation.

### 11.4 Cheap seams to slip into the current phases (so the above stays additive)
- **Seed function takes a `tag`** (Phase 1/3 when the seed plumbing is touched) → unblocks 11.2 + 11.3-rng.
- **`Compose` accepts bare callables** (anytime; ~5 lines) → unblocks 11.3.
- Keep `Collect` outputs self-contained and collate's Mapping-recursion intact (already true).
Nothing else is needed now; both ambitions remain bounded additive changes.

---

## 12. Review findings folded in (pre-flight + corrections)

Three independent reviews (plan↔code fidelity, simplicity, completeness/risk) against this
plan + the code. The design is sound; these corrections + pre-flight are the result.
**Where they conflict with inline text in §§3A/3B/5/6/10, this section wins.**

### Pre-flight — do BEFORE Phase 0 (it's the gate, not a Phase-5 nicety)
- **PF1 — Golden snapshots on current `master`.** No test today locks the "single-modality
  byte-identity" promise (§6/§9) or Phase-1 "decoration parity." Capture golden collated
  C1/C2/C3 batches + golden decorated `segment`/`instance` from `master` first, assert
  unchanged after. You can't run old vs new API side-by-side once `labl`→`labels=` lands, so
  capture up front.
- **PF2 — The submodule.** `particle-imaging-models/libs/pimm-data` is a **git submodule**
  pinned at a SHA (one commit behind working HEAD). Every "both repos" phase (0,1,3,5) has
  THREE parts: working repo + **submodule SHA bump** + pimm configs. Add the gitlink bump to
  each phase gate, or pimm trains old code against new configs.

### Corrections to specific claims
- **C-fix1 — `MultiModalEventDataset` is ALIVE, not a planned/renamed thing.** §6 wording was
  misleading. TWO layers: `ShardEventDataset` (`_dataset_base.py`, reader-composition base)
  AND `MultiModalEventDataset` (`multimodal.py`, the live selection/holdout/source-mixture
  wrapper; owns the 3-tuple holdout, `event_identity`, the `event_label` broadcast). Keep
  both; Phase-1 C3/label changes touch `MultiModalEventDataset`.
- **C-fix2 — Keep `rank` in the seed (default 0); do NOT delete it.** `rank` decorrelates
  noise across DDP replicas (same event drawn by two ranks → identical noise without it);
  single-node makes it a harmless no-op. `derive_seed(identity, base_seed, epoch, *, tag,
  rank=0)`; coordinate with `train.py` (passes `rank=comm.get_rank()`). This supersedes the
  "drop `rank`" lines in §3A-G1 and §5.
- **C-fix3 — The numpy `Densify`/`AddNoise`/`Digitize` are the bit-exactness ORACLE** (torch
  is tested `==` numpy in `test_batch_transforms.py`; ~10 `test_noise.py` tests). "Delete the
  twins" = remove from the **runtime/import path**, but **retain numpy `densify` as a
  test-only oracle (or freeze a golden array)** — keep the correctness proof; port the live
  `test_noise.py` assertions onto `dense_ops`.
- **C-fix4 — "One `ApplyToModality` in both phases" needs seed threading.** `ApplyToStream`/
  `Compose` call `t(data_dict)` positionally; post-collate noise stages need keyword `seeds=`
  that only `apply_batch_transforms` computes. Phase 3 decides: (a) `ApplyToModality`
  computes+injects `seeds`/`rng`, or (b) stages self-seed from top-level `batch['name']`.
  Small but real new contract — not "no new machinery."
- **C-fix5 — Mode C (pre-collate densify) is NOT collate-free.** The custom `collate_fn` does
  `torch.cat` (not `stack`) on tensor leaves → B×`(W,T)` becomes `(B·W,T)`, wrong. Default
  mode C to *post-collate `device='cpu'`*; the *pre-collate* dense variant needs a small
  stack-collate path + a test before use. Modes A/B unaffected. (Supersedes the §3B
  "no collate change" note for pre-collate.)
- **C-fix6 — Denoise clean-grid availability (Phase-4 blocker).** `BatchAddIntrinsicNoise`
  mutates the grid in place → the clean pre-noise grid is gone before the model. For the
  denoise target either (a) the runner stashes the `Densify` output as a clean key before
  noising, or (b) the model re-densifies clean. Decide at Phase 4 ((a) = a second grid in
  VRAM; (b) = a recompute).
- **C-fix7 — Epoch→`__getitem__` plumbing does NOT exist** — it is real Phase-1 work, not a
  footnote. `set_epoch` is only called under DDP, on the sampler, and never reaches the
  dataset; `persistent_workers=True` means no re-fork. G1's per-event `_rng` needs an epoch
  source: build sampler→dataset epoch state read at `__getitem__`. Phase 1 owns it.

### Migration inventory (so the rename/relabel is complete)
- **Phase 0 rename** also hits: `AggregateSensorHits(stream=)` (`detector_transforms.py`);
  **`test_shim.py:26` hardcodes `"ApplyToStream"`** (grep gate won't catch it); the
  transform/recipe tests; pimm `configs/detector/_base_/jaxtpc_seg.py` + `configs/lucid/.../
  mu-e.py`; stale docstrings (`multimodal.py` 2-tuple; dense headers' `rank`).
- **Phase 1 relabel** rewrites `test_jaxtpc_task_matrix.py` (encodes the old API + asserts the
  `('labl',)`/`('sensor','labl')` rejects that become inexpressible) and
  `test_jaxtpc_semantics.py`; touches `lucid.py` decoration. C3's per-point `event_label`
  removal touches `multimodal.py` + the LUCiD probe hook (`lucid_event_probe.py`) — verify
  the probe's expected shape.
- **Sequencing:** Phase 1 and Phase 2 both edit `Collect.__call__` → **serialize them** (one
  owner). G2 (`name` top-level) is a hard dependency Phase 2 owes Phase 3/4 (the dense seeder
  reads top-level `batch['name']`).

### Test gaps to author (§9 additions)
panoptic `segment`+`instance` co-alignment through GridSample/crop; `-1`-vs-stuff sentinel
(+ its interaction with `point_collate_fn`'s `-1` instance-offset mask); namespaced-collate
per-modality offsets + junk-drop; mode-C collate stacking; pre-collate G1 `_rng`
reproducibility (across workers/epoch/resume); transcode-preserves-identity-attrs; G2
unconditional `name`. Name all three collate fns (`collate_fn`/`point_collate_fn`/
`inseg_collate_fn`) in the byte-identity invariant.

### Open questions (small, confirm at build)
- `AggregateSensorHits`: still on the live LUCiD SSL path, or orphaned (→ delete Phase 5)?
- `point_collate_fn`: does any config set `mix_prob>0`? If not, it `==` `collate_fn` → drop it.
- GridSample `min/max/mean/first` reducers: any used, or is `sum` the only one in production?

### Coherent-noise "drift" — RESOLVED (was a test-infra collision, NOT a port drift)
Investigated: there is **no** port drift. pimm-data's `coherent_noise` matches the JAXTPC
oracle to **max abs 9.5e-7** (well inside the test's `atol=1e-5`) when JAXTPC's `tools` is
imported cleanly. The 3 failures (`test_reconcile_coherent_bitexact_with_jaxtpc`,
`test_coherent_batched_matches_jaxtpc[2048/4321]`) were entirely a **`tools` namespace
collision**: `_import_jaxtpc` did `import tools.coherent_noise`, which — depending on test
order — resolved to a *foreign/stale* `tools` already cached in `sys.modules`, so the
comparison ran against the wrong module. **Fixed** by hardening both `_import_jaxtpc`
helpers (test_noise, test_batch_transforms) to evict any cached `tools[.*]` not loaded from
the JAXTPC root and verify the resolved module's `__file__`. Suite is now fully green with
**no deselects**; the "coherent bit-exact vs JAXTPC" invariant (§6) holds, so this is **not**
a Phase-3 prerequisite. (The port is faithful; nothing to reconcile.)

### Labels reclassification — as-built + a corrected estimate
Delivered (green): `labels=` opt-in on **both** `JAXTPCDataset` and `LUCiDDataset`
(`labl` is a label *source*, auto-loaded on the `labels=` signal, still joining the
cross-modality index via `_readers_named()`); the production `jaxtpc_seg` config
migrated to `modalities=("step",), labels="pdg"`; parity tests confirm byte-identical
decoration. The legacy `modalities=(...,'labl')` path is **kept working, deprecated**.

**Corrected estimate:** the full *hard rejection* (flip `_validate_modalities` to reject
`labl` in `modalities=` + migrate every legacy test) was scoped as "churny but
mechanical." On inspection it's a **structural rewrite of data-driven tests** — esp.
`test_lucid.py`'s parametrized modality-combo lists and the reject-asserting tests in
`test_jaxtpc_task_matrix.py` — touching ~10 files, for the *sole* marginal benefit of
removing the deprecated door (the additive API already delivers the clean path). So the
hard rejection is **reclassified as a dedicated follow-up cleanup**, lower priority than
the committed C4 work. Recommended order: legacy-path removal as its own pass once configs
have migrated.

### G1 (per-event seeding / reproducibility) — DEFERRED
Decision: **start non-reproducible.** Global-RNG augmentation (the current behavior) is the
norm and trains fine; none of C1–C4 require per-event-reproducible augmentation; bit-exact
replay isn't achievable anyway (AMP/flash nondeterministic, `deterministic=False`). The
reproducibility that matters — the train/val/test **split** — is already done and is
independent of G1. C4 denoise doesn't need reproducible *input* noise (the model denoises
whatever it sees; the clean target is deterministic; the existing `content_seed(name)` is
adequate). So G1's epoch→`__getitem__` plumbing + per-event `_rng` is deferred to when a
real need arrives — chiefly the §11 multi-stream/user-transform seams, which would build the
seed-`tag` *with* them. **G2 is kept** (it's just "don't drop `name`", and it helps the
existing noise seed — not reproducibility infra). Net: **Phase 1 = labels + G2 (done)**;
G1 parked.
