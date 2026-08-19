"""Which pimm-data transforms work inside Apply on each modality?

For every 'typical' transform and every modality (3D seg, 2D inst, 2D sensor),
build a tiny pipeline with Apply + the transform, run it, and check
the output shape is sensible. This documents — and enforces — the supported
recipes for using transforms on nested sub-dicts (see README §Using with
transforms).
"""

import numpy as np
import pytest
import torch

from pimm_data import JAXTPCDataset, Compose, collate_fn
from pimm_data.transform import TRANSFORMS


# --- fixtures: one dataset per modality config ------------------------------

def _ds(root, modalities, **kw):
    defaults = dict(data_root=root, split='', dataset_name='sim',
                    modalities=modalities, label_key='pdg', min_deposits=0,
                    max_len=2)
    defaults.update(kw)
    return JAXTPCDataset(**defaults)


@pytest.fixture(scope='module')
def step_sample(jaxtpc_data_root):
    """A 3D step sub-dict with labels (step.coord shape (N,3))."""
    ds = _ds(jaxtpc_data_root, ('step', 'labl'))
    return ds.get_data(0)


@pytest.fixture(scope='module')
def hits_sample(jaxtpc_data_root):
    """A 2D hits sub-dict with labels (hits.coord shape (E,2))."""
    ds = _ds(jaxtpc_data_root, ('hits', 'labl'))
    return ds.get_data(0)


@pytest.fixture(scope='module')
def sensor_sample(jaxtpc_data_root):
    """A 2D sensor sub-dict, no labels (sensor.coord shape (M,2))."""
    ds = _ds(jaxtpc_data_root, ('sensor',))
    return ds.get_data(0)


def _run(sample, modality, transforms):
    """Run transforms inside an Apply on a fresh copy and return the
    post-transform sub-dict."""
    from copy import deepcopy
    data = deepcopy(sample)
    pipe = Compose([
        dict(type='Apply', on=modality, transforms=transforms),
    ])
    data = pipe(data)
    return data[modality]


# --- spatial transforms on 3D seg -----------------------------------------

def test_step_3d_normalize_coord(step_sample):
    out = _run(step_sample, 'step', [
        dict(type='NormalizeCoord', center=[0, 0, 0], scale=4000.0)])
    assert np.max(np.linalg.norm(out['coord'], axis=1)) < 2.0, \
        "after scale=4000 mm, radius should be roughly bounded"


def test_step_3d_random_rotate(step_sample):
    out = _run(step_sample, 'step', [
        dict(type='RandomRotate', angle=[-1, 1], axis='z',
             center=[0, 0, 0], p=1.0)])
    assert out['coord'].shape == step_sample['step']['coord'].shape


def test_step_3d_random_flip(step_sample):
    out = _run(step_sample, 'step', [dict(type='RandomFlip', p=1.0)])
    assert out['coord'].shape == step_sample['step']['coord'].shape


def test_step_3d_random_scale(step_sample):
    out = _run(step_sample, 'step', [
        dict(type='RandomScale', scale=[0.9, 1.1])])
    assert out['coord'].shape == step_sample['step']['coord'].shape


def test_step_3d_random_jitter(step_sample):
    out = _run(step_sample, 'step', [
        dict(type='RandomJitter', sigma=0.01, clip=0.05)])
    assert out['coord'].shape == step_sample['step']['coord'].shape


def test_step_3d_grid_sample(step_sample):
    out = _run(step_sample, 'step', [
        dict(type='GridSample', grid_size=10.0, hash_type='fnv',
             mode='train', return_grid_coord=True)])
    n_before = step_sample['step']['coord'].shape[0]
    n_after = out['coord'].shape[0]
    assert n_after <= n_before
    assert 'grid_coord' in out


def test_step_3d_log_transform_energy(step_sample):
    out = _run(step_sample, 'step', [
        dict(type='LogTransform', min_val=0.01, max_val=20.0,
             keys=('energy',))])
    assert out['energy'].shape == step_sample['step']['energy'].shape


def test_step_3d_remap_segment(step_sample):
    out = _run(step_sample, 'step', [
        dict(type='RemapSegment', scheme='motif_5cls')])
    # motif_5cls has classes 0..4
    unique = np.unique(out['segment'])
    assert unique.max() <= 4
    assert unique.min() >= -1  # -1 sentinel preserved


def test_step_3d_shuffle_point(step_sample):
    out = _run(step_sample, 'step', [dict(type='ShufflePoint')])
    assert out['coord'].shape == step_sample['step']['coord'].shape
    # per-point arrays must still line up
    assert out['segment'].shape[0] == out['coord'].shape[0]


def test_step_3d_random_dropout(step_sample):
    out = _run(step_sample, 'step', [
        dict(type='RandomDropout', dropout_ratio=0.2, dropout_application_ratio=1.0)])
    n_before = step_sample['step']['coord'].shape[0]
    n_after = out['coord'].shape[0]
    assert n_after < n_before


def test_step_3d_positive_shift(step_sample):
    out = _run(step_sample, 'step', [dict(type='PositiveShift')])
    assert (out['coord'] >= 0).all()


def test_step_3d_copy(step_sample):
    """Copy transform duplicates keys within a modality sub-dict."""
    out = _run(step_sample, 'step', [
        dict(type='Copy', keys_dict={'coord': 'origin_coord'})])
    assert 'origin_coord' in out
    assert out['origin_coord'].shape == out['coord'].shape


# --- spatial transforms on 2D hits -----------------------------------------

def test_hits_2d_grid_sample(hits_sample):
    out = _run(hits_sample, 'hits', [
        dict(type='GridSample', grid_size=1.0, hash_type='fnv',
             mode='train', return_grid_coord=True)])
    assert out['coord'].shape[1] == 2
    assert 'grid_coord' in out


def test_hits_2d_random_flip(hits_sample):
    out = _run(hits_sample, 'hits', [
        dict(type='RandomFlip', p=0.5, axes=('x', 'y'))])
    assert out['coord'].shape == hits_sample['hits']['coord'].shape


def test_hits_2d_random_scale(hits_sample):
    out = _run(hits_sample, 'hits', [
        dict(type='RandomScale', scale=[0.9, 1.1])])
    assert out['coord'].shape == hits_sample['hits']['coord'].shape


def test_hits_2d_random_jitter(hits_sample):
    out = _run(hits_sample, 'hits', [
        dict(type='RandomJitter', sigma=0.01, clip=0.05)])
    assert out['coord'].shape == hits_sample['hits']['coord'].shape


def test_hits_2d_remap_segment(hits_sample):
    out = _run(hits_sample, 'hits', [
        dict(type='RemapSegment', scheme='motif_5cls')])
    unique = np.unique(out['segment'])
    assert unique.max() <= 4 and unique.min() >= -1


def test_hits_2d_shuffle_point(hits_sample):
    out = _run(hits_sample, 'hits', [dict(type='ShufflePoint')])
    assert out['segment'].shape[0] == out['coord'].shape[0]


def test_hits_2d_random_dropout(hits_sample):
    out = _run(hits_sample, 'hits', [
        dict(type='RandomDropout', dropout_ratio=0.2,
             dropout_application_ratio=1.0)])
    assert out['coord'].shape[0] < hits_sample['hits']['coord'].shape[0]


def test_hits_2d_positive_shift(hits_sample):
    out = _run(hits_sample, 'hits', [dict(type='PositiveShift')])
    assert (out['coord'] >= 0).all()


# --- sensor modality (no labels) --------------------------------------------

def test_sensor_2d_grid_sample(sensor_sample):
    out = _run(sensor_sample, 'sensor', [
        dict(type='GridSample', grid_size=1.0, mode='train',
             return_grid_coord=True)])
    assert out['coord'].shape[1] == 2


def test_sensor_2d_random_flip(sensor_sample):
    out = _run(sensor_sample, 'sensor', [
        dict(type='RandomFlip', p=0.5, axes=('x', 'y'))])
    assert out['coord'].shape == sensor_sample['sensor']['coord'].shape


def test_sensor_2d_log_transform_energy(sensor_sample):
    out = _run(sensor_sample, 'sensor', [
        dict(type='LogTransform', keys=('energy',))])
    assert out['energy'].shape == sensor_sample['sensor']['energy'].shape


# --- transforms that should be avoided on 2D streams ----------------------

def test_normalize_coord_on_2d_is_unsafe(hits_sample):
    """NormalizeCoord handles 2D coords OK (no hardcoded 3D). Smoke check."""
    out = _run(hits_sample, 'hits', [
        dict(type='NormalizeCoord', center=[0, 0], scale=1000.0)])
    assert out['coord'].shape[1] == 2


def test_random_rotate_on_2d_fails(hits_sample):
    """RandomRotate builds a 3x3 matrix and multiplies by coord; 2D coord
    has shape (N, 2), incompatible. Documents the limitation."""
    with pytest.raises(ValueError):
        _run(hits_sample, 'hits', [
            dict(type='RandomRotate', angle=[-1, 1], axis='z',
                 center=[0, 0], p=1.0)])


# --- full-pipeline integration for each typical recipe --------------------

def test_recipe_3d_supervised_step(jaxtpc_data_root):
    """Canonical 3D semantic seg pipeline — the one in
    configs/detector/_base_/jaxtpc_step.py."""
    ds = _ds(jaxtpc_data_root, ('step', 'labl'),
             transform=[
                 dict(type='Apply', on='step', transforms=[
                     dict(type='RemapSegment', scheme='motif_5cls'),
                     dict(type='NormalizeCoord', center=[0, 0, 0], scale=4000.0),
                     dict(type='LogTransform', min_val=0.01, max_val=20.0),
                     dict(type='GridSample', grid_size=0.001, hash_type='fnv',
                          mode='train', return_grid_coord=True),
                     dict(type='RandomRotate', angle=[-1, 1], axis='z',
                          center=[0, 0, 0], p=1.0),
                     dict(type='RandomFlip', p=0.5),
                 ]),
                 dict(type='ToTensor'),
                 dict(type='Collect', part='step',
                      keys=('coord', 'grid_coord', 'segment'),
                      feat_keys=('coord', 'energy')),
             ])
    batch = collate_fn([ds[0], ds[1]])
    assert batch['coord'].shape[1] == 3
    assert 'feat' in batch
    assert len(batch['offset']) == 2


def test_recipe_2d_supervised_hits(jaxtpc_data_root):
    """2D supervised-on-hits pipeline (hits+labl combo)."""
    ds = _ds(jaxtpc_data_root, ('hits', 'labl'),
             transform=[
                 dict(type='Apply', on='hits', transforms=[
                     dict(type='RemapSegment', scheme='motif_5cls'),
                     dict(type='GridSample', grid_size=1.0, hash_type='fnv',
                          mode='train', return_grid_coord=True),
                     dict(type='RandomFlip', p=0.5, axes=('x', 'y')),
                 ]),
                 dict(type='ToTensor'),
                 dict(type='Collect', part='hits',
                      keys=('coord', 'grid_coord', 'segment', 'instance'),
                      feat_keys=('coord', 'energy')),
             ])
    batch = collate_fn([ds[0], ds[1]])
    assert batch['coord'].shape[1] == 2


def test_recipe_ssl_raw_sensor(jaxtpc_data_root):
    """SSL on raw sensor — no labels flow."""
    ds = _ds(jaxtpc_data_root, ('sensor',),
             transform=[
                 dict(type='Apply', on='sensor', transforms=[
                     dict(type='GridSample', grid_size=1.0, mode='train',
                          return_grid_coord=True),
                     dict(type='ShufflePoint'),
                 ]),
                 dict(type='ToTensor'),
                 dict(type='Collect', part='sensor',
                      keys=('coord', 'grid_coord'),
                      feat_keys=('coord', 'energy')),
             ])
    batch = collate_fn([ds[0], ds[1]])
    assert batch['coord'].shape[1] == 2
    assert 'segment' not in batch


def test_recipe_denoising_sensor_plus_hits(jaxtpc_data_root):
    """Two Apply blocks, one per cloud. Both streams transform
    independently; Collect pulls whichever becomes the model input."""
    ds = _ds(jaxtpc_data_root, ('sensor', 'hits'),
             transform=[
                 dict(type='Apply', on='sensor', transforms=[
                     dict(type='GridSample', grid_size=1.0, mode='train',
                          return_grid_coord=True),
                 ]),
                 dict(type='Apply', on='hits', transforms=[
                     dict(type='GridSample', grid_size=1.0, mode='train',
                          return_grid_coord=True),
                 ]),
                 dict(type='ToTensor'),
                 dict(type='Collect', part='sensor',
                      keys=('coord', 'grid_coord'),
                      feat_keys=('coord', 'energy')),
             ])
    # Dataset transform runs per-sample; confirm each sample has both
    # sub-dicts transformed in-place before Collect picks one.
    sample = ds[0]
    assert 'coord' in sample and sample['coord'].shape[1] == 2
    assert 'grid_coord' in sample
