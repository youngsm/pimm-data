"""
Synthetic v3 fixture generators for JAXTPC and LUCiD.

The readers in :mod:`pimm_data.readers` expect a specific on-disk HDF5
layout (four modalities per dataset: ``step/``, ``sensor/``, ``hits/``,
``labl/``). Real fixtures are produced by the JAXTPC and LUCiD
simulation pipelines, but those take GPU-bound physics runs and
gigabytes of edepsim input — overkill for exercising the reader /
dataset plumbing.

This module builds minimal schema-conformant fixtures from pure
numpy + h5py, with no dependency on JAX, edepsim, Geant4, or any
production code. The generated files are tiny (a few KB each) and
satisfy every cross-modality invariant the readers and dataset layer
rely on:

* **JAXTPC.** ``hits.deposit_to_group`` indexes into
  ``hits.group_to_track``; ``labl.deposit_to_track[i] ==
  hits.group_to_track[hits.deposit_to_group[i]]``; every per-deposit
  track id appears in ``labl.track_ids``; per-plane CSR entries
  decode to the declared ``n_pixels``.
* **LUCiD.** Every ``step.track_idx`` appears in
  ``labl.per_track.track_id``; every ``hits.particle_idx`` and
  ``labl.per_track.particle_idx`` is a valid index into the per-
  particle tables; every ``labl.per_track.ancestor`` is itself a
  ``track_id`` in ``per_track``.

Usage::

    from pimm_data.testing import make_jaxtpc_sample, make_lucid_sample
    make_jaxtpc_sample('/tmp/jaxtpc_synth', dataset_name='sim', n_events=2)
    make_lucid_sample('/tmp/lucid_synth',   dataset_name='wc',  n_events=2)

Both functions are idempotent given the same seed.
"""

from __future__ import annotations

import os

import h5py
import numpy as np


def _stamp_step_geometry(outdir, dataset_name, n_files, n_volumes, readout_type):
    """``config/{num_wires, volume_ranges}`` on the JAXTPC step shards.

    Real JAXTPC step shards carry both; this fixture carried neither, so any
    consumer that reads them could not be exercised without production data.
    helix's ``scripts/dump_probe_truth.py:240`` reads ``config/volume_ranges``
    and died with ``KeyError: 'volume_ranges'`` on a fixture-built corpus --
    which made the fixture useless as a full-pipeline smoke input, exactly where
    it is worth most: someone receiving the code has no data yet.

    Shapes follow a real shard: ``num_wires`` (n_vol, n_planes) int32 and
    ``volume_ranges`` (n_vol, 3, 2) float32, each volume's (min, max) along
    x/y/z, with the volumes tiling x and sharing y/z.

    JAXTPC only. LUCiD's sample builder has no volumes and does not want this.
    """
    import h5py
    import numpy as np

    n_planes = 3 if readout_type == 'wire' else 1
    num_wires = np.tile(np.array([1969, 1969, 1443][:n_planes], dtype=np.int32),
                        (n_volumes, 1))
    half = 2160.0
    ranges = np.zeros((n_volumes, 3, 2), dtype=np.float32)
    edges = np.linspace(-half, half, n_volumes + 1, dtype=np.float32)
    for v in range(n_volumes):
        ranges[v, 0] = (edges[v], edges[v + 1])
        ranges[v, 1] = (-half, half)
        ranges[v, 2] = (-half, half)

    for file_idx in range(n_files):
        path = os.path.join(outdir, 'step',
                            f'{dataset_name}_step_{file_idx:04d}.h5')
        with h5py.File(path, 'a') as f:
            cfg = f.require_group('config')
            for name, data in (('num_wires', num_wires),
                               ('volume_ranges', ranges)):
                if name in cfg:
                    del cfg[name]
                cfg.create_dataset(name, data=data)


def _stamp_source_event_idx(outdir, dataset_name, modalities, n_files,
                            n_events, start=0):
    """Add a stable per-event identity to written fixtures.

    Stamps both a per-file ``config/source_event_idx`` vector (the O(1)/file
    identity path) and a per-event ``source_event_idx`` attr, matching what
    the production writers emit. The global id is contiguous across files so
    the same physics event shares one id across all modalities — the key the
    holdout/identity layer (D26/D27) relies on.
    """
    for file_idx in range(n_files):
        sei = (np.arange(n_events, dtype=np.uint32)
               + start + file_idx * n_events)
        for mod in modalities:
            path = os.path.join(
                outdir, mod, f'{dataset_name}_{mod}_{file_idx:04d}.h5')
            if not os.path.exists(path):
                continue
            with h5py.File(path, 'r+') as f:
                cfg = f['config']
                # file_index is intrinsic in real data (both detectors) and is
                # part of the stable event identity (F1) — stamp it here too.
                cfg.attrs['file_index'] = file_idx
                if 'source_event_idx' in cfg:
                    del cfg['source_event_idx']
                cfg.create_dataset('source_event_idx', data=sei)
                for k in f.keys():
                    if not k.startswith('event_'):
                        continue
                    i = int(k.rsplit('_', 1)[1])
                    if 0 <= i < n_events:
                        f[k].attrs['source_event_idx'] = int(sei[i])


# ---------------------------------------------------------------------------
# JAXTPC
# ---------------------------------------------------------------------------

_JAXTPC_PLANES = ('U', 'V', 'Y')
_JAXTPC_PDG_POOL = np.array([13, 11, 211, 22, 2212], dtype=np.int32)
_JAXTPC_POS_STEP_MM = 0.05
_JAXTPC_POS_ORIGIN = (-2160.0, -2160.0, -2160.0)


def make_jaxtpc_sample(outdir, dataset_name='sim', n_events=2, n_files=1,
                       n_volumes=2, n_deposits=60, n_groups=6, n_tracks=6,
                       n_pixels_per_plane=40, readout_type='wire', seed=0,
                       stamp_source_event_idx=True):
    """Write a minimal schema-conformant JAXTPC v3 dataset.

    Creates ``{outdir}/{step,sensor,hits,labl}/{dataset_name}_{mod}_NNNN.h5``.

    ``readout_type`` is ``'wire'`` (three U/V/Y planes per volume) or
    ``'pixel'`` (single ``Pixel`` plane per volume; coord adds a pz axis).
    """
    assert readout_type in ('wire', 'pixel'), readout_type
    os.makedirs(outdir, exist_ok=True)
    for mod in ('step', 'sensor', 'hits', 'labl'):
        os.makedirs(os.path.join(outdir, mod), exist_ok=True)

    rng = np.random.default_rng(seed)

    for file_idx in range(n_files):
        events = [
            _build_jaxtpc_event(rng, n_volumes, n_deposits, n_groups,
                                n_tracks, n_pixels_per_plane, readout_type)
            for _ in range(n_events)
        ]
        tag = f'{dataset_name}_{{mod}}_{file_idx:04d}.h5'
        _write_jaxtpc_step(os.path.join(outdir, 'step',
                                        tag.format(mod='step')), events)
        _write_jaxtpc_sensor(os.path.join(outdir, 'sensor',
                                          tag.format(mod='sensor')), events,
                             readout_type)
        _write_jaxtpc_hits(os.path.join(outdir, 'hits',
                                        tag.format(mod='hits')), events,
                           readout_type)
        _write_jaxtpc_labl(os.path.join(outdir, 'labl',
                                        tag.format(mod='labl')), events)

    _stamp_step_geometry(outdir, dataset_name, n_files, n_volumes, readout_type)

    if stamp_source_event_idx:
        _stamp_source_event_idx(outdir, dataset_name,
                                ('step', 'sensor', 'hits', 'labl'),
                                n_files, n_events)
    return outdir


def _build_jaxtpc_event(rng, n_volumes, n_deposits, n_groups, n_tracks,
                        n_pixels_per_plane, readout_type='wire'):
    """Draw a coherent set of volumes with consistent FKs."""
    plane_names = ('Pixel',) if readout_type == 'pixel' else _JAXTPC_PLANES
    volumes = []
    for v in range(n_volumes):
        track_ids = np.sort(rng.choice(
            np.arange(1, 10 * n_tracks + 1), size=n_tracks,
            replace=False)).astype(np.int32)
        # Deterministic cycle guarantees every PDG in the pool appears when
        # n_tracks >= len(pool); downstream tests rely on PDG variety and
        # on at least one value > 20 to distinguish raw from remapped.
        track_pdg = _JAXTPC_PDG_POOL[
            np.arange(n_tracks) % len(_JAXTPC_PDG_POOL)].astype(np.int32)
        # Each group belongs to exactly one track
        group_to_track = track_ids[rng.integers(0, n_tracks, size=n_groups)
                                   ].astype(np.int32)

        # Each deposit belongs to exactly one group
        deposit_to_group = rng.integers(0, n_groups, size=n_deposits).astype(
            np.int32)
        deposit_to_track = group_to_track[deposit_to_group].astype(np.int32)

        # Deposit geometry — uint16 positions scaled by pos_step_mm
        # Keep positions well inside uint16 range and the detector fiducial.
        positions = rng.integers(1000, 60000, size=(n_deposits, 3),
                                 dtype=np.uint16)
        de = rng.uniform(0.05, 2.0, size=n_deposits).astype(np.float16)
        dx = rng.uniform(0.1, 1.0, size=n_deposits).astype(np.float16)
        theta = rng.uniform(-np.pi, np.pi, size=n_deposits).astype(np.float16)
        phi = rng.uniform(-np.pi, np.pi, size=n_deposits).astype(np.float16)
        t0_us = rng.uniform(0.0, 10.0, size=n_deposits).astype(np.float16)
        charge = rng.uniform(1e3, 5e4, size=n_deposits).astype(np.float16)
        photons = rng.uniform(1e3, 5e4, size=n_deposits).astype(np.float16)
        qs_fractions = rng.uniform(0.01, 1.0, size=n_deposits).astype(
            np.float32)

        planes = {}
        for plane in plane_names:
            planes[plane] = _build_jaxtpc_plane(
                rng, n_groups, n_pixels_per_plane, readout_type)

        volumes.append(dict(
            vol_idx=v,
            positions=positions, de=de, dx=dx, theta=theta, phi=phi,
            t0_us=t0_us, charge=charge, photons=photons,
            qs_fractions=qs_fractions,
            deposit_to_group=deposit_to_group,
            deposit_to_track=deposit_to_track,
            group_to_track=group_to_track,
            track_ids=track_ids,
            track_pdg=track_pdg,
            planes=planes,
        ))
    return volumes


def _build_jaxtpc_plane(rng, n_groups, n_pixels_per_plane, readout_type='wire'):
    """One plane: CSR-packed per-group pixel entries + delta-encoded sparse.

    For pixel readout, coord adds a second spatial axis (py/pz) in both
    the CSR centers/deltas and the sensor sparse stream.
    """
    # CSR: split n_pixels across n_groups with random but ≥1 group_size,
    # capped at uint8 max.
    base = np.ones(n_groups, dtype=np.int32)
    remainder = n_pixels_per_plane - n_groups
    if remainder < 0:
        # Fewer pixels than groups — trim groups logically by zero-sizing
        # the tail (keep n_groups rows so lengths match group_to_track).
        group_sizes = base.copy()
        group_sizes[n_pixels_per_plane:] = 0
    else:
        extra = rng.multinomial(remainder, np.ones(n_groups) / n_groups)
        group_sizes = (base + extra).astype(np.uint8)
    total = int(group_sizes.sum())

    group_ids = np.arange(n_groups, dtype=np.int32)
    center_times = rng.integers(50, 2000, size=n_groups).astype(np.int16)
    peak_charges = rng.uniform(1e3, 5e4, size=n_groups).astype(np.float32)
    delta_times = rng.integers(-5, 6, size=total).astype(np.int8)
    # Charge field is readout-specific (matches production writers):
    # wire -> charges_u16 (/65535), pixel -> charges_i16 (signed, /32767).
    # Added in the per-readout branch below.

    # Delta-encoded sparse for the sensor file. Strictly ordered so
    # cumsum reconstructs a monotone stream. Made longer than the hits
    # CSR total so sensor != hits, mirroring electronics shaping which
    # spreads each hits pixel across many sensor ticks.
    n_sparse = max(total * 3 + 1, 2)
    # delta_time >= 1 after the anchor => strictly-increasing time => unique
    # (wire, time) pairs (production sparse COO is duplicate-free; this lets the
    # densify scatter / numpy<->torch parity be exact and unambiguous).
    delta_time = np.concatenate([
        np.array([0], dtype=np.int16),
        rng.integers(1, 6, size=n_sparse - 1, dtype=np.int16),
    ])
    values = rng.integers(100, 4000, size=n_sparse, dtype=np.uint16)

    out = dict(
        group_ids=group_ids, group_sizes=group_sizes,
        center_times=center_times, peak_charges=peak_charges,
        delta_times=delta_times,
        delta_time=delta_time, values=values,
        time_start=int(rng.integers(0, 100)),
        pedestal=0,
    )

    if readout_type == 'pixel':
        # Signed, matching the production writer (pixel charges are bipolar;
        # sign is carried in charges_i16, magnitude normalized by |peak|).
        out['charges_i16'] = rng.integers(-32767, 32768, size=total, dtype=np.int16)
        out['center_py'] = rng.integers(10, 200, size=n_groups).astype(np.int16)
        out['center_pz'] = rng.integers(10, 200, size=n_groups).astype(np.int16)
        out['delta_py'] = rng.integers(-2, 3, size=total).astype(np.int8)
        out['delta_pz'] = rng.integers(-2, 3, size=total).astype(np.int8)
        out['delta_py_sparse'] = np.concatenate([
            np.array([0], dtype=np.int16),
            rng.integers(0, 3, size=n_sparse - 1, dtype=np.int16),
        ])
        out['delta_pz_sparse'] = np.concatenate([
            np.array([0], dtype=np.int16),
            rng.integers(0, 3, size=n_sparse - 1, dtype=np.int16),
        ])
        out['py_start'] = int(rng.integers(0, 100))
        out['pz_start'] = int(rng.integers(0, 100))
    else:
        out['charges_u16'] = rng.integers(1, 65535, size=total, dtype=np.uint16)
        out['center_wires'] = rng.integers(10, 500, size=n_groups).astype(np.int16)
        out['delta_wires'] = rng.integers(-3, 4, size=total).astype(np.int8)
        out['delta_wire'] = np.concatenate([
            np.array([0], dtype=np.int16),
            rng.integers(0, 4, size=n_sparse - 1, dtype=np.int16),
        ])
        out['wire_start'] = int(rng.integers(0, 100))

    # Fixed grid extent for a downstream Densify: reconstruct the same absolute
    # indices the reader will (cumsum + start) and record the covering size.
    times = np.cumsum(out['delta_time'].astype(np.int64)) + out['time_start']
    out['max_time'] = int(times.max()) if times.size else 0
    if readout_type != 'pixel':
        wires = np.cumsum(out['delta_wire'].astype(np.int64)) + out['wire_start']
        out['n_wires'] = int(wires.max()) + 1 if wires.size else 0

    return out


def _write_jaxtpc_step(path, events):
    with h5py.File(path, 'w') as f:
        cfg = f.create_group('config')
        cfg.attrs['n_events'] = len(events)
        cfg.attrs['n_volumes'] = len(events[0])
        cfg.attrs['pos_step_mm'] = _JAXTPC_POS_STEP_MM
        for i, volumes in enumerate(events):
            evt = f.create_group(f'event_{i:03d}')
            for v in volumes:
                vg = evt.create_group(f'volume_{v["vol_idx"]}')
                vg.attrs['n_actual'] = int(v['positions'].shape[0])
                vg.attrs['pos_step_mm'] = _JAXTPC_POS_STEP_MM
                vg.attrs['pos_origin_x'] = _JAXTPC_POS_ORIGIN[0]
                vg.attrs['pos_origin_y'] = _JAXTPC_POS_ORIGIN[1]
                vg.attrs['pos_origin_z'] = _JAXTPC_POS_ORIGIN[2]
                vg.create_dataset('positions', data=v['positions'])
                for k in ('de', 'dx', 'theta', 'phi', 't0_us',
                          'charge', 'photons'):
                    vg.create_dataset(k, data=v[k])


def _write_jaxtpc_sensor(path, events, readout_type='wire'):
    with h5py.File(path, 'w') as f:
        cfg = f.create_group('config')
        cfg.attrs['n_events'] = len(events)
        cfg.attrs['readout_type'] = readout_type
        # One fixed tick count per file, covering every plane's max time —
        # mirrors production (sensor files carry num_time_steps) and lets a
        # downstream Densify build a fixed (n_wires, n_ticks) grid.
        cfg.attrs['num_time_steps'] = 1 + max(
            (plane['max_time'] for volumes in events for v in volumes
             for plane in v['planes'].values()), default=0)
        # n_wires is detector geometry — CONSTANT per plane type across events
        # (not the per-event max). Use the global max so every event's grid is
        # the same fixed (n_wires, n_ticks); per-event n_wires breaks batching.
        nwires_by_type = {}
        if readout_type != 'pixel':
            for volumes in events:
                for v in volumes:
                    for pn, plane in v['planes'].items():
                        nwires_by_type[pn] = max(nwires_by_type.get(pn, 0),
                                                 int(plane['n_wires']))
        for i, volumes in enumerate(events):
            evt = f.create_group(f'event_{i:03d}')
            for v in volumes:
                vg = evt.create_group(f'volume_{v["vol_idx"]}')
                for plane_name, plane in v['planes'].items():
                    pg = vg.create_group(plane_name)
                    pg.attrs['time_start'] = plane['time_start']
                    pg.attrs['pedestal'] = plane['pedestal']
                    if readout_type != 'pixel':
                        pg.attrs['n_wires'] = int(nwires_by_type[plane_name])
                    pg.create_dataset('delta_time', data=plane['delta_time'])
                    pg.create_dataset('values', data=plane['values'])
                    if readout_type == 'pixel':
                        pg.attrs['py_start'] = plane['py_start']
                        pg.attrs['pz_start'] = plane['pz_start']
                        pg.create_dataset('delta_py',
                                          data=plane['delta_py_sparse'])
                        pg.create_dataset('delta_pz',
                                          data=plane['delta_pz_sparse'])
                    else:
                        pg.attrs['wire_start'] = plane['wire_start']
                        pg.create_dataset('delta_wire',
                                          data=plane['delta_wire'])


def _write_jaxtpc_hits(path, events, readout_type='wire'):
    shared_keys = ('group_ids', 'group_sizes', 'center_times',
                   'peak_charges', 'delta_times')
    readout_keys = (
        ('center_py', 'center_pz', 'delta_py', 'delta_pz', 'charges_i16')
        if readout_type == 'pixel'
        else ('center_wires', 'delta_wires', 'charges_u16'))
    with h5py.File(path, 'w') as f:
        cfg = f.create_group('config')
        cfg.attrs['n_events'] = len(events)
        cfg.attrs['readout_type'] = readout_type
        for i, volumes in enumerate(events):
            evt = f.create_group(f'event_{i:03d}')
            for v in volumes:
                vg = evt.create_group(f'volume_{v["vol_idx"]}')
                vg.create_dataset('group_to_track', data=v['group_to_track'])
                vg.create_dataset('deposit_to_group',
                                  data=v['deposit_to_group'])
                vg.create_dataset('qs_fractions', data=v['qs_fractions'])
                for plane_name, plane in v['planes'].items():
                    pg = vg.create_group(plane_name)
                    for key in shared_keys + readout_keys:
                        pg.create_dataset(key, data=plane[key])


def _write_jaxtpc_labl(path, events):
    with h5py.File(path, 'w') as f:
        cfg = f.create_group('config')
        cfg.attrs['n_events'] = len(events)
        cfg.attrs['n_volumes'] = len(events[0])
        for i, volumes in enumerate(events):
            evt = f.create_group(f'event_{i:03d}')
            for v in volumes:
                vg = evt.create_group(f'volume_{v["vol_idx"]}')
                vg.create_dataset('track_ids', data=v['track_ids'])
                vg.create_dataset('track_pdg', data=v['track_pdg'])
                # cluster and interaction need >1 unique value for
                # test_different_label_keys; derive from track position.
                cluster = np.arange(1, 1 + len(v['track_ids']), dtype=np.int32)
                interaction = (np.arange(len(v['track_ids']),
                                         dtype=np.int32) % 3) + 1
                vg.create_dataset('track_cluster', data=cluster)
                vg.create_dataset('track_interaction', data=interaction)
                vg.create_dataset('track_ancestor', data=v['track_ids'])
                vg.create_dataset('deposit_to_track',
                                  data=v['deposit_to_track'])


# ---------------------------------------------------------------------------
# LUCiD
# ---------------------------------------------------------------------------

_LUCID_PDG_POOL = np.array([11, 13, 22, 211, -11], dtype=np.int32)


def make_lucid_sample(outdir, dataset_name='wc', n_events=2, n_files=1,
                      n_segments=80, n_hits=120, n_hits_entries=200,
                      n_sensors=64, n_tracks=8, n_particles=3, seed=0,
                      stamp_source_event_idx=True):
    """Write a minimal schema-conformant LUCiD v3 dataset.

    Creates ``{outdir}/{step,sensor,hits,labl}/{dataset_name}_{mod}_NNNN.h5``.
    """
    os.makedirs(outdir, exist_ok=True)
    for mod in ('step', 'sensor', 'hits', 'labl'):
        os.makedirs(os.path.join(outdir, mod), exist_ok=True)

    rng = np.random.default_rng(seed)
    # Single PMT geometry reused across all events and both sensor/hits files.
    pmt_positions = rng.uniform(-500.0, 500.0,
                                size=(n_sensors, 3)).astype(np.float32)

    for file_idx in range(n_files):
        events = [
            _build_lucid_event(rng, n_segments, n_hits, n_hits_entries,
                               n_sensors, n_tracks, n_particles)
            for _ in range(n_events)
        ]
        tag = f'{dataset_name}_{{mod}}_{file_idx:04d}.h5'
        _write_lucid_step(os.path.join(outdir, 'step',
                                       tag.format(mod='step')), events)
        _write_lucid_sensor(os.path.join(outdir, 'sensor',
                                         tag.format(mod='sensor')),
                            events, pmt_positions)
        _write_lucid_hits(os.path.join(outdir, 'hits',
                                       tag.format(mod='hits')),
                          events, pmt_positions)
        _write_lucid_labl(os.path.join(outdir, 'labl',
                                       tag.format(mod='labl')), events)

    if stamp_source_event_idx:
        _stamp_source_event_idx(outdir, dataset_name,
                                ('step', 'sensor', 'hits', 'labl'),
                                n_files, n_events)
    return outdir


def _build_lucid_event(rng, n_segments, n_hits, n_hits_entries,
                       n_sensors, n_tracks, n_particles):
    # Allocate track ids — ancestor must itself be a valid track id, so we
    # pick one track per particle to be that particle's ancestor and
    # everyone else's ancestor falls back to it.
    track_ids = np.sort(rng.choice(
        np.arange(1, 10 * n_tracks + 1), size=n_tracks,
        replace=False)).astype(np.int32)
    track_particle_idx = rng.integers(0, n_particles, size=n_tracks).astype(
        np.int32)
    # One root per particle — the first track mapped to that particle.
    roots_by_particle = {}
    for tid, pidx in zip(track_ids, track_particle_idx):
        roots_by_particle.setdefault(int(pidx), int(tid))
    ancestor = np.array(
        [roots_by_particle[int(pidx)] for pidx in track_particle_idx],
        dtype=np.int32)

    # Per-particle
    category = rng.integers(0, 5, size=n_particles).astype(np.int32)
    contained = (rng.random(n_particles) > 0.3).astype(bool)
    # interaction (vertex) each particle belongs to — FK enabling the
    # one-hop instance_interaction axis (particle_idx → interaction_idx).
    n_interactions = 2
    particle_interaction_idx = rng.integers(
        0, n_interactions, size=n_particles).astype(np.int32)
    # Trivial genealogy: each particle's entry is just its own index.
    genealogy_offsets = np.arange(n_particles + 1, dtype=np.int32)
    genealogy_data = np.arange(n_particles, dtype=np.int32)
    ext_genealogy_offsets = genealogy_offsets.copy()
    ext_genealogy_data = genealogy_data.copy()

    # Per-track metadata
    pdg = rng.choice(_LUCID_PDG_POOL, size=n_tracks).astype(np.int32)
    parent_id = np.where(
        rng.random(n_tracks) < 0.5, 0, track_ids).astype(np.int32)
    interaction = rng.integers(0, 3, size=n_tracks).astype(np.int32)
    initial_energy = rng.uniform(0.1, 10.0, size=n_tracks).astype(np.float32)
    n_cherenkov_track = rng.integers(0, 100, size=n_tracks).astype(np.int32)

    # Step — track_idx is a POSITIONAL index into the per_track table
    # (row index, not the Geant4 track_id value). See
    # lucid.py::_lookup_per_track which gathers with track_idx directly.
    seg_track_idx = rng.integers(0, n_tracks, size=n_segments).astype(np.int32)
    start = rng.uniform(-500.0, 500.0, size=(n_segments, 3)).astype(np.float32)
    direction = rng.normal(0.0, 1.0, size=(n_segments, 3)).astype(np.float32)
    norms = np.linalg.norm(direction, axis=1, keepdims=True)
    direction = np.where(norms > 0, direction / norms, direction)
    end = start + direction * rng.uniform(
        0.1, 10.0, size=(n_segments, 1)).astype(np.float32)
    edep = rng.uniform(0.01, 5.0, size=n_segments).astype(np.float32)
    seg_time = rng.uniform(0.0, 100.0, size=n_segments).astype(np.float32)
    beta_start = rng.uniform(0.0, 1.0, size=n_segments).astype(np.float32)
    n_cherenkov_seg = rng.integers(0, 50, size=n_segments).astype(np.int32)
    contained_seg = (rng.random(n_segments) > 0.2).astype(bool)

    # Sensor
    sensor_sensor_idx = rng.integers(0, n_sensors, size=n_hits).astype(
        np.int32)
    sensor_pe = rng.uniform(0.1, 50.0, size=n_hits).astype(np.float32)
    sensor_t = rng.uniform(0.0, 100.0, size=n_hits).astype(np.float32)

    # Hits — particle_idx must be valid index into per_particle (< n_particles)
    hits_sensor_idx = rng.integers(0, n_sensors, size=n_hits_entries).astype(
        np.int32)
    hits_particle_idx = rng.integers(0, n_particles,
                                     size=n_hits_entries).astype(np.int32)
    hits_pe = rng.uniform(0.05, 20.0, size=n_hits_entries).astype(np.float32)
    # Real WAND (fv5) stores times as float64; keep the fixture dtype honest.
    hits_t = rng.uniform(0.0, 100.0, size=n_hits_entries).astype(np.float64)
    # fv5 columns: reconstructed (smeared) time, digitizer digit FK, and
    # emission process (0=Cherenkov, 1=scintillation, 2=dark).
    hits_t_reco = (hits_t + rng.normal(0.0, 1.0, size=n_hits_entries)).astype(
        np.float64)
    hits_digit_idx = np.arange(n_hits_entries, dtype=np.int32)
    hits_emission = rng.integers(0, 2, size=n_hits_entries).astype(np.int8)

    # Per-interaction (per-neutrino-vertex), F5. Dtypes match the v3 writer
    # (neutrino_pdg int16, source_type uint8, ragged offsets uint32). One
    # primary per interaction → CSR offsets = arange(n_interactions+1).
    int_vertex = rng.uniform(-500.0, 500.0,
                             size=(n_interactions, 3)).astype(np.float32)
    interaction_table = dict(
        contained=(rng.random(n_interactions) > 0.3).astype(bool),
        # fv5: interaction channel code (-1 for non-neutrino sources)
        interaction_channel=np.full(n_interactions, -1, dtype=np.int32),
        n_particles=rng.integers(1, n_particles + 1,
                                 size=n_interactions).astype(np.int32),
        n_primaries=np.ones(n_interactions, dtype=np.int32),
        neutrino_energy_MeV=rng.uniform(100.0, 5000.0,
                                        size=n_interactions).astype(np.float32),
        neutrino_pdg=rng.choice(np.array([12, 14, -12, -14], dtype=np.int16),
                                size=n_interactions),
        source_type=rng.integers(0, 3, size=n_interactions).astype(np.uint8),
        t0=rng.uniform(-1.0, 1.0, size=n_interactions).astype(np.float32),
        vertex_x=int_vertex[:, 0], vertex_y=int_vertex[:, 1],
        vertex_z=int_vertex[:, 2],
        primary_pdgs_data=rng.choice(_LUCID_PDG_POOL,
                                     size=n_interactions).astype(np.int16),
        primary_pdgs_offsets=np.arange(n_interactions + 1, dtype=np.uint32),
        primary_energies_data=rng.uniform(100.0, 5000.0,
                                          size=n_interactions).astype(np.float32),
        primary_energies_offsets=np.arange(n_interactions + 1, dtype=np.uint32),
        primary_track_ids_data=rng.choice(track_ids,
                                          size=n_interactions).astype(np.int32),
        primary_track_ids_offsets=np.arange(n_interactions + 1, dtype=np.uint32),
    )

    # fv5: spatial segment-group ids + per-trigger-window table (one window
    # covering all digits — matches single-vertex WAND events).
    seg_group_id = rng.integers(0, 4, size=n_segments).astype(np.int32)
    window_table = dict(
        digit_offsets=np.array([0, n_hits], dtype=np.int32),
        window_start=np.array([0.0], dtype=np.float64),
        window_end=np.array([100.0], dtype=np.float64),
    )

    return dict(
        step=dict(
            start=start, end=end, direction=direction, edep=edep,
            time=seg_time, track_idx=seg_track_idx, beta_start=beta_start,
            n_cherenkov=n_cherenkov_seg, contained=contained_seg,
            group_id=seg_group_id,
        ),
        sensor=dict(
            sensor_idx=sensor_sensor_idx, PE=sensor_pe, T=sensor_t,
        ),
        hits=dict(
            sensor_idx=hits_sensor_idx, particle_idx=hits_particle_idx,
            PE=hits_pe, T=hits_t, T_reco=hits_t_reco,
            digit_idx=hits_digit_idx, emission_process=hits_emission,
        ),
        labl=dict(
            t0=np.float32(rng.uniform(-1.0, 1.0)),
            event_contained=bool(rng.random() > 0.3),
            category=category, contained=contained,
            particle_interaction_idx=particle_interaction_idx,
            genealogy_data=genealogy_data,
            genealogy_offsets=genealogy_offsets,
            ext_genealogy_data=ext_genealogy_data,
            ext_genealogy_offsets=ext_genealogy_offsets,
            track_id=track_ids, pdg=pdg, parent_id=parent_id,
            particle_idx=track_particle_idx, ancestor=ancestor,
            interaction=interaction, initial_energy=initial_energy,
            n_cherenkov=n_cherenkov_track,
            n_particles=n_particles,
            per_interaction=interaction_table,
            per_window=window_table,
        ),
    )


def _write_lucid_step(path, events):
    with h5py.File(path, 'w') as f:
        cfg = f.create_group('config')
        cfg.attrs['n_events'] = len(events)
        cfg.attrs['format_version'] = 5   # match real WAND (readers are version-agnostic)
        # fv5 detector geometry (HK-like cylinder, meters)
        cfg.attrs['detector_half_height'] = 32.96
        cfg.attrs['detector_radius'] = 32.43
        cfg.create_dataset('detector_axis',
                           data=np.array([0.0, 0.0, 1.0], dtype=np.float32))
        for i, evt in enumerate(events):
            seg = evt['step']
            g = f.create_group(f'event_{i:03d}')
            g.attrs['n_segments'] = int(seg['start'].shape[0])
            g.create_dataset('start_x', data=seg['start'][:, 0])
            g.create_dataset('start_y', data=seg['start'][:, 1])
            g.create_dataset('start_z', data=seg['start'][:, 2])
            g.create_dataset('end_x', data=seg['end'][:, 0])
            g.create_dataset('end_y', data=seg['end'][:, 1])
            g.create_dataset('end_z', data=seg['end'][:, 2])
            g.create_dataset('edep', data=seg['edep'])
            g.create_dataset('time', data=seg['time'])
            g.create_dataset('track_idx', data=seg['track_idx'])
            g.create_dataset('dir_x', data=seg['direction'][:, 0])
            g.create_dataset('dir_y', data=seg['direction'][:, 1])
            g.create_dataset('dir_z', data=seg['direction'][:, 2])
            g.create_dataset('beta_start', data=seg['beta_start'])
            g.create_dataset('n_cherenkov', data=seg['n_cherenkov'])
            g.create_dataset('contained', data=seg['contained'])
            g.create_dataset('group_id', data=seg['group_id'])


def _write_lucid_sensor(path, events, pmt_positions):
    with h5py.File(path, 'w') as f:
        cfg = f.create_group('config')
        cfg.attrs['n_events'] = len(events)
        cfg.attrs['n_sensors'] = int(pmt_positions.shape[0])
        cfg.attrs['format_version'] = 5   # match real WAND (readers are version-agnostic)
        cfg.create_dataset('sensor_positions', data=pmt_positions)
        for i, evt in enumerate(events):
            s = evt['sensor']
            g = f.create_group(f'event_{i:03d}')
            g.create_dataset('sensor_idx', data=s['sensor_idx'])
            g.create_dataset('PE', data=s['PE'])
            g.create_dataset('T', data=s['T'])


def _write_lucid_hits(path, events, pmt_positions):
    with h5py.File(path, 'w') as f:
        cfg = f.create_group('config')
        cfg.attrs['n_events'] = len(events)
        cfg.attrs['n_sensors'] = int(pmt_positions.shape[0])
        cfg.attrs['format_version'] = 5   # match real WAND (readers are version-agnostic)
        cfg.create_dataset('sensor_positions', data=pmt_positions)
        for i, evt in enumerate(events):
            s = evt['hits']
            g = f.create_group(f'event_{i:03d}')
            g.create_dataset('sensor_idx', data=s['sensor_idx'])
            g.create_dataset('particle_idx', data=s['particle_idx'])
            g.create_dataset('PE', data=s['PE'])
            g.create_dataset('T', data=s['T'])
            g.create_dataset('T_reco', data=s['T_reco'])
            g.create_dataset('digit_idx', data=s['digit_idx'])
            g.create_dataset('emission_process', data=s['emission_process'])


def _write_lucid_labl(path, events):
    with h5py.File(path, 'w') as f:
        cfg = f.create_group('config')
        cfg.attrs['n_events'] = len(events)
        cfg.attrs['format_version'] = 5   # match real WAND (readers are version-agnostic)
        for i, evt in enumerate(events):
            l = evt['labl']
            g = f.create_group(f'event_{i:03d}')
            g.attrs['n_particles'] = int(l['n_particles'])

            pe = g.create_group('per_event')
            pe.create_dataset('t0', data=np.float32(l['t0']))
            pe.create_dataset('contained',
                              data=np.bool_(l['event_contained']))

            pint = g.create_group('per_interaction')
            for k, arr in l['per_interaction'].items():
                pint.create_dataset(k, data=arr)

            pwin = g.create_group('per_window')
            for k, arr in l['per_window'].items():
                pwin.create_dataset(k, data=arr)

            pp = g.create_group('per_particle')
            pp.create_dataset('category', data=l['category'])
            pp.create_dataset('contained', data=l['contained'])
            pp.create_dataset('interaction_idx',
                              data=l['particle_interaction_idx'])
            pp.create_dataset('genealogy_data', data=l['genealogy_data'])
            pp.create_dataset('genealogy_offsets', data=l['genealogy_offsets'])
            pp.create_dataset('ext_genealogy_data', data=l['ext_genealogy_data'])
            pp.create_dataset('ext_genealogy_offsets',
                              data=l['ext_genealogy_offsets'])

            pt = g.create_group('per_track')
            pt.create_dataset('track_id', data=l['track_id'])
            pt.create_dataset('pdg', data=l['pdg'])
            pt.create_dataset('parent_id', data=l['parent_id'])
            pt.create_dataset('particle_idx', data=l['particle_idx'])
            pt.create_dataset('ancestor', data=l['ancestor'])
            pt.create_dataset('interaction', data=l['interaction'])
            pt.create_dataset('initial_energy', data=l['initial_energy'])
            pt.create_dataset('n_cherenkov', data=l['n_cherenkov'])


# ---------------------------------------------------------------------------
# Optical (PMT light) — doraemon ``label_N`` chunk schema
# ---------------------------------------------------------------------------

def _write_optical_chunk_group(g, rng, n_pmts, max_chunks, chunk_len, pedestal):
    """Write one ``{adc,offsets,pmt_id,t0_ns}`` chunk group (no pe_counts)."""
    n_chunks = int(rng.integers(1, max_chunks + 1))
    pmt_id = rng.integers(0, n_pmts, size=n_chunks).astype(np.int32)
    lens = rng.integers(4, chunk_len + 1, size=n_chunks)
    offsets = np.concatenate([[0], np.cumsum(lens)]).astype(np.int64)
    adc = (pedestal + rng.normal(0, 5, size=int(offsets[-1]))
           ).round().clip(0, (1 << 15) - 1).astype(np.uint16)
    t0_ns = rng.uniform(0, 500, size=n_chunks).astype(np.float32)
    g.create_dataset('adc', data=adc)
    g.create_dataset('offsets', data=offsets)
    g.create_dataset('pmt_id', data=pmt_id)
    g.create_dataset('t0_ns', data=t0_ns)


def make_optical_sample(outdir, dataset_name='optical', n_events=2, n_files=1,
                        n_channels=8, n_labels=3, max_chunks_per_label=4,
                        chunk_len=16, schema='label', seed=0):
    """Write a minimal schema-conformant optical ``sensor/`` dataset.

    ``schema='label'`` writes the goop ``event_NNN/label_K/{adc,offsets,pmt_id,
    t0_ns,pe_counts}`` layout (doraemon). ``schema='east_west'`` writes
    ``event_NNN/{east,west}/{...}`` + event-level ``pe_counts_{side}`` and an
    ``n_pmts_per_side`` config (``light_output.h5``). Tiny by construction; the
    per-channel PE-truth invariant (``pe_counts[pmt_id]``) holds so the reader's
    per-chunk ``pe`` matches. TPC-truth datasets are omitted (the loader ignores
    them). Returns ``outdir``.
    """
    sensor_dir = os.path.join(outdir, 'sensor')
    os.makedirs(sensor_dir, exist_ok=True)
    rng = np.random.default_rng(seed)
    pedestal = 100.0
    n_per_side = n_channels // 2 if schema == 'east_west' else n_channels

    for file_idx in range(n_files):
        path = os.path.join(sensor_dir,
                            f'{dataset_name}_sensor_{file_idx:04d}.h5')
        with h5py.File(path, 'w') as f:
            cfg = f.create_group('config')
            cfg.attrs['n_events'] = n_events
            cfg.attrs['n_channels'] = n_channels
            cfg.attrs['pedestal'] = pedestal
            cfg.attrs['tick_ns'] = 1.0
            cfg.attrs['n_bits'] = 15
            cfg.attrs['gain'] = 1.0
            cfg.attrs['baseline_noise_std'] = 0.0
            cfg.attrs['dataset_name'] = dataset_name
            cfg.attrs['file_index'] = file_idx
            cfg.attrs['global_event_offset'] = file_idx * n_events
            if schema == 'east_west':
                cfg.attrs['n_pmts_per_side'] = n_per_side
            else:
                cfg.attrs['label_key'] = 'interaction'
            for i in range(n_events):
                evt = f.create_group(f'event_{i:03d}')
                if schema == 'east_west':
                    for side in ('east', 'west'):
                        g = evt.create_group(side)
                        _write_optical_chunk_group(
                            g, rng, n_per_side, max_chunks_per_label,
                            chunk_len, pedestal)
                        evt.create_dataset(
                            f'pe_counts_{side}',
                            data=rng.integers(0, 50, size=n_per_side
                                              ).astype(np.int32))
                else:
                    for lab in rng.choice(n_labels * 2, size=n_labels,
                                          replace=False):
                        g = evt.create_group(f'label_{int(lab)}')
                        _write_optical_chunk_group(
                            g, rng, n_channels, max_chunks_per_label,
                            chunk_len, pedestal)
                        g.create_dataset(
                            'pe_counts',
                            data=rng.integers(0, 50, size=n_channels
                                              ).astype(np.int32))
    return outdir


__all__ = ['make_jaxtpc_sample', 'make_lucid_sample', 'make_optical_sample']
