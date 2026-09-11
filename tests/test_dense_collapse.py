"""Densify's numpy oracle.

test_digitize_oracle_equiv left with the forward model: dense_ops.digitize is
helix's now, and the test moved there restated as the formula rather than as a
cross-boundary oracle import. What remains is the scatter, which is a data-layer
operation and stays.
Guard the dense-op twin collapse: the post-collate torch core
(:mod:`pimm_data.dense_ops`) must reproduce the pre-collate numpy transforms
bit-for-bit on a single event BEFORE the numpy twins are deleted. This witnesses
that the collapse is lossless — densify is one scope='sample' op, placed by the
harness, not two implementations.
"""
import numpy as np
import torch

from pimm_data import dense_ops
from pimm_data.detector_transforms import Densify


def _synthetic_plane(W, T, n, seed):
    """n unique (wire,time) cells in a W×T grid with float values."""
    rng = np.random.default_rng(seed)
    flat = rng.choice(W * T, size=n, replace=False)
    wire = (flat // T).astype(np.int64)
    time = (flat % T).astype(np.int64)
    value = rng.standard_normal(n).astype(np.float32)
    return wire, time, value


def test_densify_oracle_equiv_single_event():
    """numpy Densify (per-event, nested-by-label) == dense_ops.densify (flat COO,
    B=1) for the same event, bit-for-bit."""
    W, T, gid = 13, 29, 0
    wire, time, value = _synthetic_plane(W, T, 40, seed=1)

    # pre-collate numpy path (nested-by-label representation)
    sub = {'readout_type': 'wire',
           'raw': {'U': {'wire': wire, 'time': time, 'value': value}},
           'shape': {'U': (W, T)}}
    np_grid = Densify()(sub)['dense']['U']                       # (W, T)

    # post-collate torch path (flat COO, single event => B=1, offset=[N])
    grids = dense_ops.densify(
        torch.from_numpy(wire), torch.from_numpy(time), torch.from_numpy(value),
        torch.full((wire.size,), gid, dtype=torch.long),
        torch.tensor([wire.size]), {gid: {'n_wires': W, 'n_ticks': T}})
    torch_grid = grids[gid][0].numpy()                           # (W, T)

    assert torch_grid.shape == np_grid.shape == (W, T)
    assert np.array_equal(torch_grid, np_grid)
