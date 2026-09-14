# Challenge campaign — dataset × task → config

> **Boundary note (2026-09-13).** Where this document places the LArTPC forward
> model — `AddNoise`, `Digitize`, the noise spectrum, `add_intrinsic_noise` — in
> pimm-data, that is no longer true. Those moved to **helix** (pimm-data
> `b592600` / helix `8075a53`): pimm-data keeps `densify`, which every detector
> family needs, and gave up the physics only one family applies. The transform
> registry raises on duplicate registration, so the two packages are lockstep.
> See helix `docs/ARCHITECTURE.md` §3. The design reasoning below is kept as
> written; only the ownership changed.

The single cross-dataset matrix of the ML **challenges** we care about and the
data-layer recipe each needs. It consolidates the two per-detector task tables
that already existed — LUCiD's `docs/LUCID_DATASET.md` §"Tasks → files" and
pimm's `docs/DETECTOR_DATASET.md` §"Task → config" — and extends them to the
new `OpticalDataset`. It lives here because **pimm-data is the one repo that
owns all three datasets** (`JAXTPCDataset`, `LUCiDDataset`, `OpticalDataset`) and
the modality/transform vocabulary.

**Goal:** drive a campaign that builds the runnable training config for every
row below. The configs themselves live in **`particle-imaging-models/configs/`**
(pimm owns models + training); this doc is the tracker (each row carries its
config path + status).

## How to read a row

Every challenge is `Dataset(modalities=…, labels=…) → transforms → Collect`:

- **modalities** — the tuple passed to the dataset. Vocabulary: `step` (3D truth
  deposits / segments — LUCiD calls these `edep`), `sensor` (raw readout), `hits`
  (per-particle decomposition of `sensor`; the **instance**-bearing modality),
  `labl` (label dimension tables, requested via `labels=`).
  - The LUCiD source doc's columns map as **`inst → hits`, `seg → step`**.
- **labels** — `labels='pdg'|'cluster'|'interaction'|'ancestor'` (JAXTPC) /
  `labels=True` (LUCiD) attaches `segment`/`instance` to `step`/`hits`.
  `RemapSegment(scheme='motif_5cls')` turns raw `pdg` into dense class indices.
- **Collect target** — the flat-prefixed batch the model consumes
  (`step_coord`/`step_segment`, `sensor_*`, …); SSL uses `MultiCrop`.
- **status** — `exists` (config in pimm), `planned` (to build).

---

## JAXTPC (LArTPC; wire/pixel auto-detected)

| Challenge | modalities | labels | key transforms | Collect | config | status |
|---|---|---|---|---|---|---|
| 3D semantic seg (5-class motif) | `('step',)` | `'pdg'` | `RemapSegment(motif_5cls)`, `GridSample`, `RandomRotate/Flip` | `step`: coord/grid_coord/segment, feat=coord+energy | **recipe** `configs/jaxtpc/semseg_5cls.py` (pimm training cfg: `…/semseg-pt-v3m2-jaxtpc-5cls.py`) | recipe ✓ |
| 3D instance seg | `('step',)` | `'cluster'` or `'ancestor'` | `GridSample` | `step`: coord/segment/instance | — | planned (jaxtpc supervised — deferred) |
| 3D self-supervised (SSL) | `('step',)` | — | `MultiCrop` (global/local) | `global`/`local`: coord/origin_coord/feat | `configs/jaxtpc/ssl_step.py` | recipe ✓ |
| Interaction classification/grouping | `('step',)` | `'interaction'` | `GridSample` | `step`: coord/segment | — | planned (jaxtpc supervised — deferred) |
| Raw-readout SSL | `('sensor',)` | — | `MultiCrop` (no geom aug — index space) | `global`/`local`: coord/feat | `configs/jaxtpc/ssl_sensor.py` | recipe ✓ |
| Sensor dense (densify + noise on GPU) | `('sensor',)` | — | ONE transform list, split at `Collect`: worker Collects sparse COO; post-collate tail (`ToDevice`→`Densify`→`AddNoise`→`Digitize`) = `dataset.batch_transform`, run on GPU | `sensor`: wire/time/value/plane_gid → `sensor_dense {gid:(B,W,T)}` | `configs/jaxtpc/sensor_dense_gpu.py` | recipe ✓ |
| Instance seg on hits | `('hits',)` | `'cluster'` | `GridSample` | `hits`: coord/segment/instance | — | planned (jaxtpc supervised — deferred) |
| sensor → step charge/energy recon | `('step','sensor')` | `'pdg'` (opt) | per-modality `Apply` | `sensor` in, `step` target | — | planned (jaxtpc supervised — deferred) |

## LUCiD (Water Cherenkov; PMT)

Direct from `LUCID_DATASET.md` §"Tasks → files" (`inst→hits`, `seg→step`):

| Challenge | modalities | labels | Collect | config | status |
|---|---|---|---|---|---|
| SSL on raw PMT readout | `('sensor',)` | — | `AggregateSensorHits(flatten=False)` + `MultiCrop` → `global`/`local` | `configs/lucid/ssl_sensor.py` (new-API port of the sonata pretrain) | recipe ✓ |
| SSL on per-particle decomposition | `('hits',)` | — | `MultiCrop` → `global`/`local` | `configs/lucid/ssl_hits.py` | recipe ✓ |
| SSL on 3D segments | `('step',)` | — | `MultiCrop` → `global`/`local` | `configs/lucid/ssl_step.py` | recipe ✓ |
| Per-segment Cherenkov forward sim | `('step',)` | — | `step`: coord + physics (beta/n_cherenkov) | — | planned (needs physics-key names) |
| sensor → inst denoising/deconv | `('sensor','hits')` | — | `sensor` in, `hits` target | (same shape as recon, `step`→`hits`) | planned |
| sensor → seg recon (vertex/energy/dir) | `('sensor','step')` | — | `sensor` in, `step` target | `configs/lucid/recon_sensor_to_step.py` | recipe ✓ |
| Per-PMT semantic/instance seg | `('hits',)` | `True` | `hits`: coord/grid_coord/segment/instance, feat=coord+energy+time | **recipe** `configs/lucid/perpmt_seg_hits.py` | recipe ✓ |
| 3D semantic/instance seg on segments | `('step',)` | `True` | `step`: coord/grid_coord/segment/instance | `configs/lucid/seg_step.py` | recipe ✓ |
| Event classification (μ/e, PID, π0/e, ν-flavor) | `('sensor',)` per source | `event_label` | `MultiModalEventDataset` + per-config holdout | `configs/lucid/event_{mu_vs_e,pid4,pi0_vs_e,genie_numu_vs_nue}.py` | recipe ✓ |
| Containment-filtered training | (any) | `True` | + `min_segments`/containment filter | — | planned (filter flag on any recipe) |

## Optical (PMT light; per-chunk waveforms — new)

`OpticalDataset(schema='label'|'east_west')`; each `sensor` row is a waveform
chunk, `instance` = group (interaction for `label`, side for `east_west`), the
packed `adc` is the second row-space. (Targets here are new — not in the source
docs.)

| Challenge | schema | modalities | per-chunk target | Collect | status |
|---|---|---|---|---|---|
| Interaction/operator discrimination | `label` | `('sensor',)` | `instance` (interaction) | `sensor`: pmt_id/t0_ns/length/pe/instance/adc | recipe ✓ `configs/optical/interaction_discrimination.py` |
| Per-channel PE regression | `label` | `('sensor',)` | `pe` | `sensor`: … + pe target | recipe ✓ (same loader: `configs/optical/interaction_discrimination.py`, target=pe) |
| Waveform SSL / pretraining | `label`/`east_west` | `('sensor',)` | — | `sensor`: adc (+ wave_offset) | recipe ✓ (data via the label/east-west loaders; view-gen is model-side) |
| Waveform denoising / compression | `east_west` | `('sensor',)` | clean/coeffs | `sensor`: adc | recipe ✓ `configs/optical/eastwest_readout.py` |
| Side-aware readout (east/west) | `east_west` | `('sensor',)` | — | `sensor`: … + instance(side) | recipe ✓ `configs/optical/eastwest_readout.py` |

---

## Building the configs

These configs are **data-loader specs** — the point is to set up each
challenge's data loading *exactly*, not the model. So the campaign is built in
two layers:

1. **Recipe (pimm-data `configs/<dataset>/<challenge>.py`)** — the data-loading
   half only, in the new flat-prefixed API (`Apply(on=)` + `Collect(parts=)`).
   It lives here because the new API + `OpticalDataset` exist only in pimm-data
   (not yet in pimm's pinned submodule), so a recipe is **verifiable now**:
   `tests/test_campaign_configs.py` execs each recipe against the synthetic
   fixtures and asserts the challenge's flat keys. `status = recipe ✓`.
2. **Training config (`particle-imaging-models/configs/`)** — lifts a recipe's
   `data`/`transform` block and adds the model/optimizer/hooks half, following
   the existing naming (`<task>-<backbone>-<dataset>-<variant>.py`). Needs pimm's
   `libs/pimm-data` submodule bumped to the redesign SHA first.

Built (all `recipe ✓`, verified by `tests/test_campaign_configs.py`): JAXTPC
semseg / ssl_step / ssl_sensor; LUCiD perpmt_seg_hits / ssl_sensor / ssl_hits /
ssl_step / seg_step / recon_sensor_to_step; Optical interaction_discrimination /
eastwest_readout. **Deferred:** JAXTPC supervised-label rows (per request); LUCiD
event-class/regression (needs WAND `MultiModalEventDataset` sources/holdout) and
per-segment Cherenkov (needs physics-key names); optical waveform model (TBD).

### Real data roots (verified loading)

| Dataset | `data_root` | `dataset_name` | `split` |
|---|---|---|---|
| JAXTPC (wire) | `/sdf/data/neutrino/doraemon/wire_test_00_00_02` | `sim_wire` | `run_0027575715` (type-first: `sensor/run_*/…`; 46 runs on disk as of 2026-07) |
| LUCiD (WAND) | `/sdf/data/neutrino/cjesus/DORAEMON/OLD/SK_like/config_NNNNNN` | `wc` | `''` |
| LUCiD step view | `/sdf/data/neutrino/omara/wand_sk_like_step/config_NNNNNN` | `wc` | `''` |
| Optical (label) | `/sdf/data/neutrino/doraemon/optical_test_00_00_02/wire/run_NNNNNNNNNN/sensor_optical` | `test_00_00_02_wire` | `''` |

> Optical layout changed (2026-07): the root now nests `{wire,pixel}/run_*/
> sensor_optical/` per run (paired with the TPC runs). The flat
> `test_00_00_02_pixel`-at-root layout this table previously recorded is gone;
> the toy-run validation below predates the move — re-validate the loader
> against the new files before relying on it.

`examples/toy_run.py` was run against these. Loaded + ran a toy model end-to-end:
JAXTPC ssl_step/ssl_sensor; LUCiD ssl_sensor/ssl_hits/perpmt_seg_hits (WAND
config_000001); Optical interaction (**K=4836 chunks / 72.2M packed samples /
24 interaction classes** on 2 real events). Real-data findings:
- **JAXTPC `labl` absent** at that path → supervised JAXTPC seg can't load there
  (deferred anyway). Non-supervised JAXTPC works.
- **DONE — LUCiD step↔`edep` via a symlink mirror:** the LUCiD step reader globs
  `step/wc_step_*` but WAND ships `edep/wc_edep_*`.
  `scripts/make_wand_step_mirror.sh` builds
  `/sdf/data/neutrino/omara/wand_sk_like_step/config_NNNNNN/` with
  `step/wc_step_*.h5 → cjesus …/edep/wc_edep_*.h5` (sensor/hits/labl dir-symlinked
  through). LUCiD `ssl_step` / `seg_step` / `recon_sensor_to_step` now load REAL
  WAND through this root (verified: ssl_step 4 global/12 local crops; recon
  sensor=3263 + step=20628 pts). (Alternative, not taken: an `edep` dir-name
  option on `LUCiDStepReader`.) NB: the older `/sdf/data/neutrino/omara/wand_sk_like`
  mirror is **stale** — its `edep` points at a now-deleted cjesus `seg/`; use the
  `_step` mirror above (or cjesus directly for sensor/hits).
- Real WAND `MultiModalEventDataset` index build is **slow (minutes)** — event
  recipes are gated on a synthetic two-config fixture instead.

### WAND config map + event-classification source combinations

18 single-/multi-particle WAND configs: 001 mu-, 002 pi+, 003 e-, 004 pi-,
005 pi0, 006 low-E e-, 007–012 multi-particle gun, 013 GENIE νμ, 014 GENIE νe,
015–018 pile-up. The event-class recipes pick source combinations:

| Recipe | sources (config → class) | physics |
|---|---|---|
| `event_mu_vs_e` | 001 mu- / 003 e- | single-ring lepton ID (Cherenkov PID) |
| `event_pid4` | 001/003/002/005 | 4-class single-particle PID |
| `event_pi0_vs_e` | 003 e- / 005 pi0 | NC-π0 ring-counting background |
| `event_genie_numu_vs_nue` | 013 / 014 | realistic ν-flavor (GENIE CC) |

Each uses the deterministic `(config_id, file_index, source_event_idx)` holdout
(`n_per_config=2000`) for reproducible train/val/test. Further combos to add when
wanted: energy regression within a config, single-vs-pileup (001/003 vs 015–018),
multi-particle topologies (007–012).

### Findings / follow-ups

- **DONE — LUCiD raw-sensor SSL new-API fix:** `AggregateSensorHits` gained
  `flatten=False` (keep the aggregate **nested** in the `sensor` sub-dict) so the
  new `MultiCrop(on='sensor')` + `Collect(parts=)` flow works
  (`MultiViewGenerator` → `MultiCrop` is otherwise a direct swap). Used by the
  LUCiD `ssl_sensor` / `recon_sensor_to_step` recipes.
- **Sensor `coord` is detector-index space** (2D wire / 3D pixel for JAXTPC, 3D
  PMT positions for LUCiD): JAXTPC sensor recipes skip `NormalizeCoord` and 3D
  rotations (not meaningful on index space); normalization/aug there is
  readout-dependent and left to the pimm lift.
- **`coord` comes out float64** after `NormalizeCoord` (python-float scale) —
  model-only concern (feat is float32 via `feat_keys`); not fixed (models out of
  scope for now).
- **Optical needs a waveform/sequence model** (PT-v3 is point-cloud only); the
  optical recipes are data-half + placeholder model until one exists.
- **DONE — event_label passthrough:** `Collect` now passes `event_label`/
  `config_id`/`weight` (MMED's event carriers) through like `name`/`split`, so
  event-classification batches keep their target. Minor: `event_label` currently
  collates to a *list of np.arrays* (one per sample) — cast to a tensor
  model-side, or fold a scalar-stacking step into the event role later.

Update the `config`/`status` cell as each is built so this stays the campaign's
source of truth.
