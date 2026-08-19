"""Collect build-time validation (REDESIGN §9) — fail at construction, not epoch 1."""
import pytest

from pimm_data.transform import Collect


def test_namespaced_spec_requires_keys():
    with pytest.raises(ValueError, match="needs 'keys='"):
        Collect(parts={'step': dict(feat_keys=('coord',))})   # no keys=


def test_roles_for_non_collected_key_rejected():
    with pytest.raises(ValueError, match="non-collected keys"):
        Collect(parts={'step': dict(keys=('coord',),
                                          roles={'edge_index': ('edge', 'self')})})  # not in keys


def test_valid_namespaced_spec_ok():
    Collect(parts={'step': dict(keys=('coord', 'edge_index'),
                                     roles={'edge_index': ('edge', 'self')})})


def test_rejects_both_forms():
    with pytest.raises(AssertionError):
        Collect(keys=['coord'], parts={'step': dict(keys=['coord'])})


def test_the_removed_aliases_are_rejected_by_name():
    """`modality=`/`modalities=` on Collect are gone, and must not fall into **kwargs.

    They are worth rejecting BY NAME rather than just deleting: `modalities=` is
    still a live argument one config level up, on the dataset constructor, where
    it means something different (which modalities to READ). The same word named
    two things, and Collect's `**kwargs` would have swallowed a misplaced one
    without a word — silently collecting nothing.
    """
    import pytest
    for dead, use in (("modality", "part"), ("modalities", "parts")):
        with pytest.raises(TypeError) as ei:
            Collect(**{dead: 'step'})
        assert f"use {use}=" in str(ei.value), str(ei.value)
        assert "DATASET is unrelated" in str(ei.value)
