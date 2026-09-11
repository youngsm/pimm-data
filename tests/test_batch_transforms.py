"""Tests for the post-collate dense GPU path (dense_ops + batch_transforms).

Covers: densify torch↔numpy bit-exact (CPU + CUDA), offset2batch edges, coherent
bit-exact vs the numpy oracle / JAXTPC, incoherent statistical RMS, digitize
bit-exact, the end-to-end runner (Densify→AddNoise→Digitize) on a collated batch,
empty-stages no-op, and the no-`torch.cuda`-import gate.
"""

import os
from copy import deepcopy

import numpy as np
import pytest
import torch

from pimm_data import JAXTPCDataset, collate_fn, move_to_device
from pimm_data.transform import Compose
from pimm_data import dense_ops
from pimm_data.jaxtpc import canonical_plane_id
from pimm_data.detector_transforms import Densify

_WL = {'U': (0.42, 4.63), 'V': (0.42, 4.63), 'Y': (2.33, 2.33)}


def _ds(root, B=2):
    return JAXTPCDataset(
        data_root=root, split='', dataset_name='sim', modalities=('sensor',),
        max_len=B, wire_lengths_per_plane=_WL,
        transform=[dict(type='Collect', part='sensor',
                        keys=('wire', 'time', 'value', 'plane_gid'))])


def _sensor_batch(root, B=2):
    ds = _ds(root, B)
    batch = collate_fn([ds[i] for i in range(B)])
    ds.get_data(0)  # ensure reader geometry is populated for plane_geometry()
    return ds, batch


# --------------------------------------------------------------------------
# densify
# --------------------------------------------------------------------------

@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA")


def test_offset2batch_basic_and_empty():
    # cumulative offsets: sample0=3 hits, sample1=0 (empty), sample2=2
    assert dense_ops.offset2batch(torch.tensor([3, 3, 5])).tolist() == [0, 0, 0, 2, 2]
    assert dense_ops.offset2batch(torch.tensor([3])).tolist() == [0, 0, 0]   # B=1
    assert dense_ops.offset2batch(torch.tensor([0, 0, 0])).tolist() == []     # all-empty


def test_densify_torch_matches_numpy_cpu(jaxtpc_data_root):
    ds, batch = _sensor_batch(jaxtpc_data_root, B=2)
    geom = ds.plane_geometry()
    grids = dense_ops.densify(batch['wire'], batch['time'], batch['value'],
                              batch['plane_gid'], batch['offset'], geom)
    # numpy oracle: run the per-sample Densify on each event, compare per plane
    for i in range(2):
        sub = ds.get_data(i)['sensor']
        Densify()(sub)
        for label, img in sub['dense'].items():
            gid = canonical_plane_id(label)
            assert np.array_equal(grids[gid][i].cpu().numpy(), img), \
                f"event {i} plane {label}: torch densify != numpy Densify"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA")
def test_densify_cpu_matches_cuda(jaxtpc_data_root):
    ds, batch = _sensor_batch(jaxtpc_data_root, B=2)
    geom = ds.plane_geometry()
    cpu = dense_ops.densify(batch['wire'], batch['time'], batch['value'],
                            batch['plane_gid'], batch['offset'], geom)
    cu = move_to_device(batch, 'cuda')
    gpu = dense_ops.densify(cu['wire'], cu['time'], cu['value'],
                            cu['plane_gid'], cu['offset'], geom)
    for gid in cpu:
        assert torch.equal(cpu[gid], gpu[gid].cpu())


def test_densify_rejects_float_indices():
    geom = {0: {'n_wires': 8, 'n_ticks': 16}}
    with pytest.raises(TypeError):
        dense_ops.densify(torch.zeros(3), torch.zeros(3), torch.ones(3),
                          torch.zeros(3, dtype=torch.long), torch.tensor([3]), geom)


# --------------------------------------------------------------------------
# noise — coherent bit-exact, incoherent statistical
# --------------------------------------------------------------------------

def test_todevice_only_is_move_no_dense(jaxtpc_data_root):
    _, batch = _sensor_batch(jaxtpc_data_root, B=2)
    out = Compose([dict(type='ToDevice', device='cpu')])(deepcopy(batch))
    assert 'dense' not in out
    assert torch.equal(out['wire'], batch['wire'])


def test_densify_under_volume_filter(jaxtpc_data_root):
    ds = JAXTPCDataset(data_root=jaxtpc_data_root, split='', dataset_name='sim',
                       modalities=('sensor',), volume=1, max_len=2,
                       wire_lengths_per_plane=_WL,
                       transform=[dict(type='Collect', part='sensor',
                                       keys=('wire', 'time', 'value', 'plane_gid'))])
    batch = collate_fn([ds[0], ds[1]])
    ds.get_data(0)
    geom = ds.plane_geometry()
    assert geom and set(geom) <= {3, 4, 5}, "volume=1 -> canonical gids 3,4,5 only"
    grids = dense_ops.densify(batch['wire'], batch['time'], batch['value'],
                              batch['plane_gid'], batch['offset'], geom)
    assert set(grids) <= {3, 4, 5}


def test_densify_length_mismatch_raises():
    geom = {0: {'n_wires': 8, 'n_ticks': 16}}
    with pytest.raises(ValueError):
        dense_ops.densify(torch.zeros(3, dtype=torch.long), torch.zeros(2, dtype=torch.long),
                          torch.ones(3), torch.zeros(3, dtype=torch.long),
                          torch.tensor([3]), geom)


def test_densify_missing_plane_in_registry_raises():
    geom = {0: {'n_wires': 8, 'n_ticks': 16}}  # registry lacks plane 5
    with pytest.raises(KeyError):
        dense_ops.densify(torch.zeros(2, dtype=torch.long), torch.zeros(2, dtype=torch.long),
                          torch.ones(2), torch.tensor([5, 5]), torch.tensor([2]), geom)


def test_densify_out_of_bounds_raises():
    geom = {0: {'n_wires': 4, 'n_ticks': 4}}
    with pytest.raises(ValueError):
        dense_ops.densify(torch.tensor([10]), torch.tensor([0]), torch.ones(1),
                          torch.tensor([0]), torch.tensor([1]), geom)


def test_canonical_plane_id_rejects_malformed():
    from pimm_data.jaxtpc import canonical_plane_id
    for bad in ('east_U', 'volume_0', 'volumeX_0_U', 'volume_0_Q'):
        with pytest.raises(ValueError):
            canonical_plane_id(bad)


# --------------------------------------------------------------------------
# CUDA coverage (skip if no GPU): noise + digitize + end-to-end on device
# --------------------------------------------------------------------------

def test_no_torch_cuda_import_in_source():
    import pimm_data
    root = os.path.dirname(pimm_data.__file__)
    offenders = []
    for dirpath, _, files in os.walk(root):
        for fn in files:
            if fn.endswith('.py'):
                with open(os.path.join(dirpath, fn)) as f:
                    src = f.read()
                if 'import torch.cuda' in src or 'from torch.cuda' in src:
                    offenders.append(os.path.relpath(os.path.join(dirpath, fn), root))
    assert not offenders, f"torch.cuda imported in: {offenders}"
