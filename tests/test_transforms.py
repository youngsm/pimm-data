"""Tests for Compose + TRANSFORMS registry + full pipeline.

Ensures raw callables and registry dicts both work in Compose, that
build_dataset resolves by type string, and that transforms + collate
produce the expected batched output keys.
"""

import numpy as np
import pytest
import torch

from pimm_data import (
    Compose, TRANSFORMS, DATASETS, build_dataset,
    JAXTPCDataset, collate_fn,
)


def test_compose_raw_callable():
    c = Compose([lambda d: {**d, 'a': 1}, lambda d: {**d, 'b': 2}])
    out = c({})
    assert out == {'a': 1, 'b': 2}


def test_compose_registry_dict():
    # ToTensor converts numpy arrays in data to tensors
    c = Compose([dict(type='ToTensor')])
    d = {'coord': np.zeros((3, 3), dtype=np.float32)}
    out = c(d)
    assert torch.is_tensor(out['coord'])


def test_compose_mixed():
    marker = {'seen': False}
    def mark(d):
        marker['seen'] = True
        return d
    c = Compose([mark, dict(type='ToTensor')])
    d = {'coord': np.zeros((1, 3), dtype=np.float32)}
    c(d)
    assert marker['seen']


def test_compose_rejects_non_callable_non_dict():
    with pytest.raises(TypeError):
        Compose([42])


def test_transforms_registered_count():
    # 31 from transform.py + PDGToSemantic/RemapSegment/Apply/
    # AggregateSensorHits from detector_transforms.py = 35, after the
    # boundary-refactor drops (PR-A) and the SSL move to pimm (PR-C).
    assert len(TRANSFORMS) >= 33
    # A few anchor cases
    for name in ('ToTensor', 'GridSample', 'Collect', 'NormalizeCoord',
                 'RandomRotate', 'PDGToSemantic', 'Apply',
                 'RemapSegment'):
        assert TRANSFORMS.get(name) is not None, f"{name} missing"


def test_build_dataset_resolves_by_type(jaxtpc_data_root):
    cfg = dict(type='JAXTPCDataset', data_root=jaxtpc_data_root,
               split='', dataset_name='sim', modalities=('step',),
               max_len=2)
    ds = build_dataset(cfg)
    assert isinstance(ds, JAXTPCDataset)
    assert len(ds) == 2


def test_end_to_end_transform_collate(jaxtpc_data_root):
    """Pipeline: Apply scopes per-cloud transforms, Collect flattens."""
    transform = [
        dict(type='Apply', on='step', transforms=[
            dict(type='NormalizeCoord', center=[0, 0, 0], scale=4000.0),
            dict(type='GridSample', grid_size=0.001, hash_type='fnv',
                 mode='train', return_grid_coord=True),
        ]),
        dict(type='ToTensor'),
        dict(type='Collect', part='step',
             keys=('coord', 'grid_coord', 'segment'),
             feat_keys=('coord', 'energy')),
    ]
    ds = JAXTPCDataset(data_root=jaxtpc_data_root, split='',
                       dataset_name='sim',
                       modalities=('step', 'labl'), label_key='pdg',
                       min_deposits=50, max_len=4, transform=transform)
    batch = collate_fn([ds[0], ds[1]])
    assert batch['coord'].shape[1] == 3
    assert 'offset' in batch
    assert len(batch['offset']) == 2
    assert 'feat' in batch
    assert 'segment' in batch


def test_dataset_getitem_uses_transforms(jaxtpc_data_root):
    """Single-sample path: transforms applied on ds[idx].

    ToTensor recurses into nested sub-dicts, so seg['coord'] becomes a
    tensor in place.
    """
    transform = [dict(type='ToTensor')]
    ds = JAXTPCDataset(data_root=jaxtpc_data_root, split='',
                       dataset_name='sim', modalities=('step',),
                       max_len=1, transform=transform)
    s = ds[0]
    assert torch.is_tensor(s['step']['coord'])


# --- PR-B regression tests for two previously-silent transform bugs -------

def test_clip_gaussian_jitter_zero_mean():
    """ClipGaussianJitter must be zero-mean.

    Was ``self.mean = np.mean(3)`` -> scalar 3.0, adding a ~+3 offset to every
    coordinate. With scalar=0 the jitter must be exactly 0, and the mean must
    be a length-3 zero vector.
    """
    from pimm_data.transform import ClipGaussianJitter
    t = ClipGaussianJitter(scalar=1.0)
    # Deterministic catch: mean must be a length-3 zero vector, not scalar 3.0.
    assert np.asarray(t.mean).shape == (3,)
    assert np.all(np.asarray(t.mean) == 0)
    # Statistical catch: with the bug the jitter centres on ~+1 (mean 3 -> clip
    # -> 1) * scalar; zero-mean keeps the applied offset near 0.
    np.random.seed(0)
    out = t({'coord': np.zeros((20000, 3), dtype=np.float64)})['coord']
    assert np.abs(out.mean()) < 0.1


def test_random_drop_actually_writes(monkeypatch):
    """RandomDrop must actually overwrite the dropped rows.

    Was ``data_dict[key][idx][:] = value`` -> wrote into a fancy-index copy
    (silent no-op). Force apply and a deterministic choice of rows 0,1.
    """
    from pimm_data.transform import RandomDrop
    monkeypatch.setattr(np.random, 'rand', lambda *a, **k: 0.0)
    monkeypatch.setattr(np.random, 'choice',
                        lambda n, k, replace=False: np.array([0, 1]))
    t = RandomDrop(key='energy', p_apply=1.0, p_drop=0.5, value=0.0)
    energy = np.ones((4, 1), dtype=np.float64)
    out = t({'energy': energy})['energy']
    assert out[0, 0] == 0.0 and out[1, 0] == 0.0   # dropped rows written
    assert out[2, 0] == 1.0 and out[3, 0] == 1.0   # untouched rows preserved
