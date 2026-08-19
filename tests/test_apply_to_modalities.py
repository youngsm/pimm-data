"""Apply(on=) — scope a sub-pipeline to one or more parts.

A multi-part Apply(on=tuple) is IMPLICITLY shared (one RNG draw, co-registered);
independent augmentation = separate Apply blocks. No `shared` flag.
"""
import numpy as np
import pytest

from pimm_data.transform import TRANSFORMS


def _two_identical():
    base = np.array([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]], np.float32)
    return {'step': {'coord': base.copy()}, 'sensor': {'coord': base.copy()}}


def _rot(on):
    return TRANSFORMS.build(dict(
        type='Apply', on=on,
        transforms=[dict(type='RandomRotate', angle=[-1, 1], axis='z', p=1.0)]))


def test_multipart_is_implicitly_shared():
    """on=tuple -> SAME rotation on both parts (identical inputs stay equal)."""
    np.random.seed(1)
    out = _rot(['step', 'sensor'])(_two_identical())
    assert np.allclose(out['step']['coord'], out['sensor']['coord'])


def test_separate_blocks_are_independent():
    """Independent augmentation = separate Apply blocks -> the parts diverge."""
    np.random.seed(1)
    d = _two_identical()
    _rot('step')(d)
    _rot('sensor')(d)
    assert not np.allclose(d['step']['coord'], d['sensor']['coord'])


def test_shared_still_actually_transforms():
    np.random.seed(2)
    d = _two_identical()
    before = d['step']['coord'].copy()
    out = _rot(['step', 'sensor'])(d)
    assert not np.allclose(out['step']['coord'], before)


def test_missing_part_skipped_unless_required():
    np.random.seed(0)
    d = {'step': {'coord': np.ones((2, 3), np.float32)}}      # no 'sensor'
    out = _rot(['step', 'sensor'])(d)                          # sensor absent -> skipped
    assert 'sensor' not in out and 'step' in out
    with pytest.raises(KeyError, match="missing"):
        TRANSFORMS.build(dict(type='Apply', on=['step', 'sensor'], required=True,
                              transforms=[dict(type='RandomFlip')]))(d)


def test_single_part_apply_back_compat():
    """Apply is a back-compat single-part Apply."""
    np.random.seed(0)
    d = {'step': {'coord': np.ones((2, 3), np.float32) * 3}}
    out = TRANSFORMS.build(dict(type='Apply', on='step',
        transforms=[dict(type='NormalizeCoord')]))(d)
    assert 'step' in out


def test_rng_advances_after_shared_block():
    np.random.seed(3)
    pre = np.random.get_state()[1][0]
    _rot(['step', 'sensor'])(_two_identical())
    post = np.random.get_state()[1][0]
    assert pre != post
