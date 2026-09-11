"""Campaign data-loader recipes load and yield the expected challenge batch.

Each recipe in ``configs/<dataset>/<challenge>.py`` is the DATA-LOADING half of a
campaign challenge (CAMPAIGN.md). This gate execs each recipe, points its train
dataset at the synthetic fixture, runs collate, and asserts the flat-prefixed
keys the challenge needs — keeping the recipes honest as the campaign fans out.
"""
import copy
import os
import runpy

import pytest

from pimm_data import build_dataset, collate_fn

_CONFIGS = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'configs')

# Multi-crop SSL recipes emit packed global/local view parts.
_SSL = {'global_coord', 'global_offset', 'global_feat',
        'local_coord', 'local_offset', 'local_feat'}

# (recipe, train-spec overrides for the tiny fixture, required flat keys)
_JAXTPC = [
    ('jaxtpc/semseg_5cls.py', dict(min_deposits=0),
     {'step_coord', 'step_grid_coord', 'step_segment', 'step_feat', 'step_offset'}),
    ('jaxtpc/ssl_step.py', dict(min_deposits=0), _SSL),
    ('jaxtpc/ssl_sensor.py', dict(), _SSL),
]
_LUCID = [
    ('lucid/perpmt_seg_hits.py', dict(),
     {'hits_coord', 'hits_segment', 'hits_instance', 'hits_feat', 'hits_offset'}),
    ('lucid/ssl_sensor.py', dict(), _SSL),
    ('lucid/ssl_hits.py', dict(), _SSL),
    ('lucid/ssl_step.py', dict(), _SSL),
    ('lucid/seg_step.py', dict(),
     {'step_coord', 'step_segment', 'step_instance', 'step_feat', 'step_offset'}),
    ('lucid/recon_sensor_to_step.py', dict(),
     {'sensor_coord', 'sensor_feat', 'sensor_offset',
      'step_coord', 'step_feat', 'step_offset'}),
]


def _load_train(rel, **override):
    spec = dict(copy.deepcopy(runpy.run_path(os.path.join(_CONFIGS, rel))['data']['train']))
    spec.update(override)
    return build_dataset(spec)


def _collate(ds):
    b = collate_fn([ds[0], ds[1]])
    return {k for k in b if k != '_roles'}, b


@pytest.mark.parametrize('rel,override,required', _JAXTPC,
                         ids=[r[0] for r in _JAXTPC])
def test_jaxtpc_recipes(jaxtpc_data_root, rel, override, required):
    ds = _load_train(rel, data_root=jaxtpc_data_root, split='', **override)
    keys, _ = _collate(ds)
    assert required <= keys, f"{rel}: missing {required - keys}"


@pytest.mark.parametrize('rel,override,required', _LUCID,
                         ids=[r[0] for r in _LUCID])
def test_lucid_recipes(lucid_data_root, rel, override, required):
    ds = _load_train(rel, data_root=lucid_data_root, split='', **override)
    keys, _ = _collate(ds)
    assert required <= keys, f"{rel}: missing {required - keys}"


def test_optical_label_recipe(optical_data_root):
    ds = _load_train('optical/interaction_discrimination.py',
                     data_root=optical_data_root, dataset_name='optical', split='')
    keys, b = _collate(ds)
    assert {'sensor_pmt_id', 'sensor_adc', 'sensor_instance', 'sensor_offset',
            'sensor_wave_offset'} <= keys
    assert b['_roles']['sensor_adc'] == ('instance', 'sensor_wave_offset')
    assert int(b['sensor_offset'][-1]) != int(b['sensor_wave_offset'][-1])


_EVENT = ['lucid/event_mu_vs_e.py', 'lucid/event_pid4.py',
          'lucid/event_pi0_vs_e.py']  # genie configs absent from the synth fixture


@pytest.mark.parametrize('rel', _EVENT, ids=_EVENT)
def test_lucid_event_recipes(wand_synth_data_root, rel):
    """MultiModalEventDataset event-class recipes: holdout partitions, event_label
    carried to the batch (the new Collect passthrough)."""
    spec = dict(copy.deepcopy(runpy.run_path(os.path.join(_CONFIGS, rel))['data']['train']))
    spec.update(data_root=wand_synth_data_root, split='train',
                holdout=dict(seed=0, n_per_config=1), min_points=0)
    ds = build_dataset(spec)
    keys, b = _collate(ds)
    assert {'sensor_coord', 'sensor_feat', 'sensor_offset', 'event_label'} <= keys


def test_optical_eastwest_recipe(optical_eastwest_data_root):
    ds = _load_train('optical/eastwest_readout.py',
                     data_root=optical_eastwest_data_root,
                     dataset_name='light', split='')
    keys, b = _collate(ds)
    assert {'sensor_pmt_id', 'sensor_adc', 'sensor_instance', 'sensor_wave_offset'} <= keys
    assert b['_roles']['sensor_adc'] == ('instance', 'sensor_wave_offset')
    # instance = side (0 east / 1 west)
    import numpy as np
    assert set(np.unique(b['sensor_instance'].numpy()).tolist()) <= {0, 1}


