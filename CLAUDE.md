# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
pip install -e .                          # editable install (extras: [test])
pytest                                    # full suite (uses synthetic fixtures)
pytest tests/test_jaxtpc.py::test_name    # single test

# Run against real production datasets instead of synthetic fixtures:
JAXTPC_DATA_ROOT=/path/to/jaxtpc pytest
LUCID_DATA_ROOT=/path/to/wc    pytest
JAXTPC_PIXEL_DATA_ROOT=/path/to/pixel pytest tests/test_jaxtpc_pixel.py
```

There is no linter/formatter configured and no build step beyond `pip install -e .`.

## Architecture

This package is a thin data layer between sharded HDF5 detector simulation
output and PyTorch training loops. It does **not** own physics models — it
returns nested numpy dicts that transforms reshape into model-ready tensors.

### Three layers

```
readers/ (HDF5 → flat dict)  →  jaxtpc.py / lucid.py (flat → nested)  →  transform.py (nested → model batch)
```

1. **Readers** (`readers/{jaxtpc,lucid}_{step,sensor,hits,labl}.py`) — one
   per `(detector, modality)` pair. Each opens a glob of shard files,
   builds an event index, and exposes `read_event(idx) → flat dict`. They
   share no base class but follow the same protocol.
2. **Datasets** (`jaxtpc.py`, `lucid.py`, `optical.py`, `pilarnet.py`) —
   subclass `DefaultDataset` (in `defaults.py`, vendored from Pointcept). They
   wire together per-modality readers, join `labl` dimension tables onto
   `step`/`hits` point clouds, and emit a **nested** dict keyed by modality
   (`{'step': {...}, 'hits': {...}, 'labl': {...}, 'bridges': {...}}`).
   `get_data(idx)` returns raw numpy; `__getitem__` runs the transform
   pipeline. `OpticalDataset` is single-modality (`sensor`) PMT light: each
   `sensor` row is a waveform **chunk**; the raw samples are a packed **second
   row-space** (`adc`, role `('instance','sensor_wave_offset')`) — see
   "Optical" below.
3. **Transforms** (`transform.py`, `detector_transforms.py`) — Pointcept-
   style registry of point-cloud augmentations. The pipeline's terminal
   per-event transform is `Collect`, which scopes to one or more parts and
   converts numpy → torch, emitting a **flat-prefixed** batch dict
   (`step_coord`, `sensor_wire`, …) plus a `_roles` map. Anything after the
   terminal `Collect` is the **post-collate tail** (see "Dense path" below).

### Registry pattern (vendored from mmcv/Pointcept)

`_registry.py` provides `Registry`. Two top-level registries:

- `TRANSFORMS` in `transform.py` — populated via `@TRANSFORMS.register_module()`
- `DATASETS` in `builder.py` — populated via `@DATASETS.register_module()`

Configs are plain dicts with a `type:` key: `dict(type='GridSample', ...)`,
`dict(type='JAXTPCDataset', ...)`. **Import order matters** — `__init__.py`
imports modules in a specific order so the decorator side-effects run
before anything reads the registry. Adding a new transform/dataset means
both decorating it and importing the module from `__init__.py`.

### Nested-dict output and `Apply` / `Collect`

Datasets emit `{'step': {...}, 'sensor': {...}, 'hits': {...}, 'labl': {...}}`.
There is **no bare `coord` at the top level**. Transforms that hardcode
`'coord'`/`'segment'` must be scoped to a part with `Apply(on=…)` (a tuple
`on=` shares one RNG draw across parts). `Apply` is the only spelling —
`ApplyToStream`/`ApplyToModality` were aliases and are **removed**:

```python
dict(type='Apply', on='step', transforms=[
    dict(type='GridSample', grid_size=0.5, mode='train'),
])
```

`Collect(parts={'step': dict(keys=[...], feat_keys=[...])})` (or the
single-part `Collect(part='step', keys=[...])`) is the terminal per-event
transform: it pulls each named part into the flat-prefixed batch and
converts arrays to torch tensors. **`Collect` must be last** in the
per-event head — tensor output is required for `DataLoader(num_workers > 0)`
to use file-descriptor IPC instead of pickling full arrays.

`parts=`/`part=` are the only names on `Collect`; `modalities=`/`modality=`
were aliases and are **removed**, and are rejected by name rather than falling
into `**kwargs` — because `modalities=` is still live one level up on the
DATASET constructor, where it means something different (which modalities to
READ). The old `stream=` is gone.

### Dense path (post-collate tail, map → reduce → map)

A single `transform` list is **split at the terminal `Collect`** by
`ShardEventDataset._split_at_collect`: the head (`…, Collect`) is
`dataset.transform` (runs per-event in DataLoader workers, sparse — only
sparse crosses PCIe); the tail (everything after `Collect`) is
`dataset.batch_transform`, which a trainer runs **per-batch after collate**
(e.g. Lightning `on_after_batch_transfer`). `ToDevice(device=…)` opens the
tail (the device step); then `Densify`/`AddNoise`/`Digitize` (single
dispatching torch transforms — the `Batch*` variants were deleted) build
the per-plane grids **on-device**. They take `geom=` (a config-derived
plane registry via `geometry.load_plane_registry`, since real shards lack
`n_wires` attrs) and `modality='sensor'`, run `scope='sample'`, and
dispatch numpy (per-event sub-dict) vs torch (collated batch) on whether
`<part>_offset` is present. Recipe: `configs/jaxtpc/sensor_dense_gpu.py`;
bench/verify: `examples/bench_dense.py`. Coherent noise is self-contained
in pimm-data (`noise.py` numpy oracle + `dense_ops._coherent_torch`;
`coherent_numpy=True` opts back to the bit-exact oracle).

### Modality semantics and FK chain

| Modality | Role |
|---|---|
| `step`   | Per-Geant4-step truth deposits (3D) |
| `sensor` | Aggregated detector readout (raw per-channel signal) |
| `hits`   | Per-particle decomposition of `sensor` (instance-resolved) |
| `labl`   | Dimension tables (PDG, interaction, ancestry) |

`labl` has no point cloud — it joins onto `step`/`hits` to populate per-
point `segment` and `instance` columns. `('labl',)` alone and
`('sensor', 'labl')` are rejected (no instance separation to join against).

JAXTPC's FK chain (per volume, suffix `_v{N}`):
`step deposit → deposit_to_track → labl.track_ids → track_{label_key}`,
and `hits group → group_to_track → labl.track_ids → track_{label_key}`.
JAXTPC also returns a top-level `bridges` sub-dict containing
`group_to_track_v{N}`, `deposit_to_group_v{N}`, `qs_fractions_v{N}`.

### Detector-specific quirks

- **JAXTPC auto-detects readout type** (`wire` vs `pixel`) from the
  sensor/hits HDF5. Coord dimensionality and the per-volume plane filter
  (`volume=N`) depend on it — see `jaxtpc.py:_coord_keys` and the post-
  detection plane-filter block in `__init__`.
- **`label_key`** (JAXTPC) selects which `labl` column becomes `segment`:
  `'pdg'` (default), `'cluster'`, `'interaction'`, `'ancestor'`. Raw values
  flow through — use `RemapSegment(scheme='motif_5cls')` to convert to
  dense class indices.
- **LUCiD** requires `format_version >= 3` shards. `instance` is
  `particle_idx`; `segment` is `labl.particle.category`.
- **Optical** (`optical.py`, `readers/optical_sensor.py`) reads goop PMT-light
  output as per-**chunk** waveforms. Two selectable schemas (`OpticalDataset(
  schema=...)`), same per-chunk fields: `'label'` (goop `io.py` writer /
  doraemon — `event_NNN/label_K/{adc,offsets,t0_ns,pmt_id,pe_counts}`, group =
  interaction) and `'east_west'` (helix `light_output.h5` —
  `event_NNN/{east,west}/{...}` + event-level `pe_counts_{side}`, group = side,
  per-side `pmt_id`, files globbed as `*.h5`). The single reader seam is
  `_groups(evt)`. Chunks are long dense waveforms (label 20–50M, e/w ~6M
  samples/event) so per-sample point clouds are infeasible: the loader emits
  **chunks** as the `sensor` row-space (`pmt_id, t0_ns, length, pe,
  instance`=group) and packs the raw samples losslessly as the **second
  row-space** `adc` (ΣL,) tagged `('instance','sensor_wave_offset')`. No
  synthetic `coord` / no separate `plane_gid` — only the fields every goop/helix
  consumer uses; `pe` is a channel-level total within the group (shared across a
  channel's chunks), not a per-chunk PE. Canonical `Collect`:
  `offset_keys_dict=dict(offset='pmt_id', wave_offset='adc')`. TPC truth
  (`tpc_*`) is left on disk. Real data: label =
  `/sdf/data/neutrino/doraemon/optical_test_00_00_02` (`test_00_00_02_pixel`);
  e/w = goop `light_output.h5`. The second-row-space machinery (`_roles`,
  `collate`, `batch_ops.split_event`) is role-declared, not `_inst`-suffixed.

### Tests

`tests/conftest.py` provides session-scoped fixtures
(`jaxtpc_data_root`, `lucid_data_root`, `jaxtpc_pixel_data_root`) that
build synthetic v3 datasets via `pimm_data.testing` unless the matching
`*_DATA_ROOT` env var points at real shards. Tests that only make sense
on production data should use `@pytest.mark.real_data_only` — they
auto-skip when both env vars are unset (collection-time skip, see
`pytest_collection_modifyitems`).

`pimm_data.testing` regenerates a minimal but cross-modality-consistent
v3 fixture from pure numpy + h5py. Cross-modality invariants (e.g.
`labl.deposit_to_track[i] == hits.group_to_track[hits.deposit_to_group[i]]`)
are encoded there; preserve them when changing reader behaviour.
