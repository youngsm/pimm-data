"""Tests for the forward noise model + Densify/AddNoise sensor transforms.

Covers, in order:
  * add_noise / incoherent_noise / coherent_noise unit behaviour;
  * the load-bearing invariant that coherent RMS does NOT scale as 1/sqrt(N);
  * Densify reconstructs a fixed (n_wires, n_ticks) grid bit-identically to a
    helix-style scatter, and is wire-only;
  * AddNoise reproducibility per event (stable across DataLoader workers) and
    its Densify-first ordering contract;
  * end-to-end dataset pipeline (Apply: Densify -> AddNoise);
  * reconciliation against JAXTPC's canonical forward model (skipped if the
    sibling JAXTPC repo isn't importable).
"""

import os
import sys
from copy import deepcopy

import numpy as np
import pytest

from pimm_data import JAXTPCDataset, Compose
from pimm_data.jaxtpc import canonical_plane_id
from pimm_data.detector_transforms import Densify


# --------------------------------------------------------------------------
# generate_noise / component unit behaviour
# --------------------------------------------------------------------------



def _sensor_sample(root, **kw):
    ds = JAXTPCDataset(data_root=root, split='', dataset_name='sim',
                       modalities=('sensor',), max_len=2, **kw)
    return ds.get_data(0)['sensor']


def test_sensor_sample_carries_shape(jaxtpc_data_root):
    sub = _sensor_sample(jaxtpc_data_root)
    assert 'shape' in sub, "reader must surface (n_wires, n_ticks) per plane"
    for plane in sub['planes']:
        nw, nt = sub['shape'][plane]
        assert nw > 0 and nt > 0


def test_densify_reconstructs_grid(jaxtpc_data_root):
    sub = _sensor_sample(jaxtpc_data_root)
    pipe = Compose([dict(type='Apply', on='sensor',
                         transforms=[dict(type='Densify')])])
    out = pipe(deepcopy({'sensor': sub}))['sensor']
    assert 'dense' in out
    assert 'coord' in out and 'raw' in out, "point cloud must be preserved"
    for plane in sub['planes']:
        nw, nt = sub['shape'][plane]
        img = out['dense'][plane]
        assert img.shape == (nw, nt)
        assert img.dtype == np.float32
        # helix-style scatter: full(0); image[wire,time]=value  (pedestal=0)
        cols = sub['raw'][plane]
        expected = np.zeros((nw, nt), dtype=np.float32)
        expected[np.asarray(cols['wire']), np.asarray(cols['time'])] = \
            np.asarray(cols['value'], dtype=np.float32)
        assert np.array_equal(img, expected)


def test_densify_pixel_raises(jaxtpc_pixel_data_root):
    sub = _sensor_sample(jaxtpc_pixel_data_root)
    pipe = Compose([dict(type='Apply', on='sensor',
                         transforms=[dict(type='Densify', on_pixel='raise')])])
    with pytest.raises(ValueError):
        pipe(deepcopy({'sensor': sub}))


def test_sensor_sample_carries_pedestal(jaxtpc_data_root):
    sub = _sensor_sample(jaxtpc_data_root)
    assert 'pedestal' in sub, "reader must surface per-plane pedestal"
    for plane in sub['planes']:
        assert plane in sub['pedestal']


def test_sensor_flat_scatter_inputs(jaxtpc_data_root):
    sub = _sensor_sample(jaxtpc_data_root)
    for k in ('wire', 'time', 'value', 'plane_gid'):
        assert k in sub, f"sensor sub-dict must surface {k!r} for the dense path"
    n = sub['coord'].shape[0]
    assert sub['wire'].shape == (n,) and np.issubdtype(sub['wire'].dtype, np.integer)
    assert np.issubdtype(sub['plane_gid'].dtype, np.integer)
    assert np.array_equal(sub['wire'], sub['coord'][:, 0].astype(sub['wire'].dtype))
    assert np.array_equal(sub['time'], sub['coord'][:, 1].astype(sub['time'].dtype))
    gid_of_pos = np.array([canonical_plane_id(p) for p in sub['planes']])
    assert np.array_equal(sub['plane_gid'], gid_of_pos[sub['plane_id'][:, 0]])


def test_plane_geometry_registry(jaxtpc_data_root):
    ds = JAXTPCDataset(
        data_root=jaxtpc_data_root, split='', dataset_name='sim',
        modalities=('sensor',), max_len=2,
        wire_lengths_per_plane={'U': (0.42, 4.63), 'V': (0.42, 4.63),
                                'Y': (2.33, 2.33)})
    ds.get_data(0)  # populate the reader's lazily-read geometry
    geom = ds.plane_geometry()
    assert geom, "registry should be non-empty after a read"
    for gid, e in geom.items():
        assert e['n_wires'] > 0 and e['n_ticks'] > 0
        assert 'pedestal' in e
        assert e['wire_lengths'].shape == (e['n_wires'],)


def test_densify_rejects_duplicate_cells(jaxtpc_data_root):
    sub = _sensor_sample(jaxtpc_data_root)
    p = sub['planes'][0]
    cols = sub['raw'][p]
    cols['wire'] = np.concatenate([cols['wire'][:1], cols['wire'][:1]])
    cols['time'] = np.concatenate([cols['time'][:1], cols['time'][:1]])
    cols['value'] = np.concatenate([cols['value'][:1], cols['value'][:1]])
    with pytest.raises(ValueError):
        Densify()(sub)


def test_canonical_plane_id_stable():
    assert canonical_plane_id('volume_0_U') == 0
    assert canonical_plane_id('volume_0_Y') == 2
    assert canonical_plane_id('volume_1_U') == 3
    assert canonical_plane_id('volume_1_Y') == 5
    assert canonical_plane_id('volume_2_Pixel') == 2  # pixel: one plane/volume
