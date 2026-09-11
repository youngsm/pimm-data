"""Phase 1 — labels (labl -> labels=), G1 seeding, G2 name pass-through."""
import hashlib

import numpy as np

from pimm_data import JAXTPCDataset
from pimm_data.transform import Collect


def _seg_fp(a):
    a = np.ascontiguousarray(a)
    return (tuple(a.shape), str(a.dtype), hashlib.sha1(a.tobytes()).hexdigest()[:16])


def _jds(root, modalities, **kw):
    return JAXTPCDataset(data_root=root, split='', dataset_name='sim',
                         modalities=modalities, min_deposits=0, max_len=2, **kw)


def test_labels_param_matches_legacy_labl_modality(jaxtpc_data_root):
    """modalities=('step',), labels='pdg' decorates identically to the legacy
    modalities=('step','labl'), label_key='pdg' — labl is now a label source."""
    new = _jds(jaxtpc_data_root, ('step',), labels='pdg').get_data(0)['step']
    old = _jds(jaxtpc_data_root, ('step', 'labl'), label_key='pdg').get_data(0)['step']
    assert _seg_fp(new['segment']) == _seg_fp(old['segment'])
    assert _seg_fp(new['instance']) == _seg_fp(old['instance'])


def test_labels_requires_decoratable_modality(jaxtpc_data_root):
    import pytest
    with pytest.raises(ValueError, match="decoratable"):
        _jds(jaxtpc_data_root, ('sensor',), labels='pdg')


def test_labels_warns_on_undecorated_modality(jaxtpc_data_root):
    """labels= + a non-decoratable modality (sensor) is legal but WARNS — the
    silent 'sensor isn't labelled' coupling made loud."""
    import pytest
    with pytest.warns(UserWarning, match="not label-decoratable"):
        _jds(jaxtpc_data_root, ('step', 'sensor'), labels='pdg')


def test_labels_no_warn_when_all_decoratable(jaxtpc_data_root):
    """No warning when every requested modality is decoratable (step/hits)."""
    import warnings as _w
    with _w.catch_warnings():
        _w.simplefilter('error')                 # any warning -> failure
        _jds(jaxtpc_data_root, ('step',), labels='pdg').get_data(0)


def test_lucid_labels_param_matches_legacy(lucid_data_root):
    """LUCiD: modalities=('hits',), labels=True decorates identically to the
    legacy modalities=('hits','labl')."""
    from pimm_data import LUCiDDataset

    def ld(mods, **kw):
        return LUCiDDataset(data_root=lucid_data_root, split='', dataset_name='wc',
                            modalities=mods, max_len=2, **kw)

    new = ld(('hits',), labels=True).get_data(0)['hits']
    old = ld(('hits', 'labl')).get_data(0)['hits']
    assert set(new) == set(old)
    for k in ('segment', 'instance'):
        if k in new:
            assert _seg_fp(new[k]) == _seg_fp(old[k])


# --- Phase 2: namespaced multi-modality Collect ---------------------------

def test_namespaced_multimodality_collect(jaxtpc_data_root):
    """REDESIGN: Collect(parts={...}) -> FLAT underscore-prefixed keys
    (step_coord, sensor_wire, …) + _roles, each part its OWN offset, junk dropped."""
    from pimm_data import collate_fn
    ds = JAXTPCDataset(
        data_root=jaxtpc_data_root, split='', dataset_name='sim',
        modalities=('step', 'sensor'), labels='pdg', min_deposits=0, max_len=2,
        transform=[dict(type='Collect', parts={
            'step':   dict(keys=('coord', 'segment'), feat_keys=('coord', 'energy')),
            'sensor': dict(keys=('wire', 'time', 'value', 'plane_gid')),
        })])
    batch = collate_fn([ds[0], ds[1]])

    assert {'step_coord', 'step_segment', 'step_feat', 'step_offset',
            'sensor_wire', 'sensor_time', 'sensor_value', 'sensor_plane_gid',
            'sensor_offset', 'name', 'split', '_roles'} <= set(batch)
    assert 'sensor_coord' not in batch and 'sensor_raw' not in batch   # junk gone
    # each part has its OWN cumulative offset (B,), no leading 0
    assert batch['step_offset'].shape == (2,) and batch['sensor_offset'].shape == (2,)
    assert int(batch['step_offset'][-1]) == batch['step_coord'].shape[0]
    assert int(batch['sensor_offset'][-1]) == batch['sensor_wire'].shape[0]
    assert batch['step_coord'].shape[1] == 3 and batch['sensor_wire'].ndim == 1


def test_collect_rejects_both_forms():
    import pytest
    from pimm_data.transform import Collect
    with pytest.raises(AssertionError):
        Collect(keys=['coord'], parts={'step': dict(keys=['coord'])})


def test_sensor_consumed_sparse_no_densify(jaxtpc_data_root):
    """No dense assumption: sensor is a sparse modality. A model may collect it
    as a 2D point cloud (coord/feat) with NO densify step — densify is opt-in."""
    from pimm_data import collate_fn
    ds = JAXTPCDataset(
        data_root=jaxtpc_data_root, split='', dataset_name='sim',
        modalities=('step', 'sensor'), labels='pdg', min_deposits=0, max_len=2,
        transform=[dict(type='Collect', parts={
            'step':   dict(keys=('coord', 'segment'), feat_keys=('coord', 'energy')),
            'sensor': dict(keys=('coord',), feat_keys=('coord', 'energy')),
        })])
    batch = collate_fn([ds[0], ds[1]])
    assert batch['sensor_coord'].shape[1] == 2             # 2D wire×time point cloud
    assert 'sensor_feat' in batch and 'sensor_offset' in batch
    assert 'sensor_dense' not in batch                     # nothing forced dense


# --- Phase 3: optional densify scoped to a namespaced modality ------------

def test_transcode_preserves_identity_attrs(jaxtpc_data_root, tmp_path):
    """Re-codec'ing a shard must preserve the 3-tuple identity inputs
    (source_event_idx / global_event_offset / file_index) — otherwise the
    identity-based holdout split silently moves (real-data finding F1)."""
    import glob
    import importlib.util
    import os

    from pimm_data._shard_meta import read_shard_meta

    tc_path = os.path.join(os.path.dirname(__file__), '..', 'scripts',
                           'transcode_codec.py')
    spec = importlib.util.spec_from_file_location('transcode_codec', tc_path)
    tc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tc)

    shards = (glob.glob(os.path.join(jaxtpc_data_root, 'step', '*.h5'))
              or glob.glob(os.path.join(jaxtpc_data_root, '**', '*step*.h5'),
                           recursive=True))
    assert shards, "no step shard in the fixture"
    src = shards[0]
    before = read_shard_meta(src)
    dst = str(tmp_path / 'transcoded.h5')
    tc.transcode_file(src, dst, tc.codec_kwargs('lzf'))   # built-in codec, no plugin
    after = read_shard_meta(dst)

    assert after['global_event_offset'] == before['global_event_offset']
    assert after['file_index'] == before['file_index']
    sb, sa = before['source_event_idx'], after['source_event_idx']
    assert (sa is None) == (sb is None)
    if sb is not None:
        assert np.array_equal(sa, sb)


def test_g2_bare_collect_passes_name_split():
    # modality=None (bare) Collect must still carry the identity keys.
    data = {'coord': np.zeros((3, 3), np.float32), 'name': 'evt0', 'split': 'train'}
    out = Collect(keys=['coord'])(data)
    assert out['name'] == 'evt0'
    assert out['split'] == 'train'
    assert tuple(out['offset'].tolist()) == (3,)


def test_g2_modality_collect_passes_name_split():
    data = {'step': {'coord': np.zeros((3, 3), np.float32)},
            'name': 'evt0', 'split': 'train'}
    out = Collect(part='step', keys=['coord'])(data)
    assert out['name'] == 'evt0' and out['split'] == 'train'
