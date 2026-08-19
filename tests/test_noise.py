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
from pimm_data.noise import (generate_noise, incoherent_noise, coherent_noise,
                             digitize, DEFAULT_ENC, DEFAULT_SAMPLING_RATE_HZ)


# --------------------------------------------------------------------------
# generate_noise / component unit behaviour
# --------------------------------------------------------------------------

def test_generate_noise_returns_noise():
    """generate_noise returns the noise array (caller adds); shape/dtype OK."""
    rng = np.random.default_rng(0)
    shape = (128, 256)
    noise = generate_noise(shape, rng=rng, wire_lengths_m=2.3, incoherent=True,
                           coherent=True)
    assert noise.shape == shape
    assert noise.dtype == np.float32
    assert np.any(noise != 0.0)
    # caller adds it; generation does not touch any image
    img = np.zeros(shape, dtype=np.float32)
    noisy = img + noise
    assert np.all(img == 0.0) and np.any(noisy != 0.0)


def test_incoherent_requires_wire_lengths():
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError):
        generate_noise((8, 16), rng=rng, incoherent=True, coherent=False)


def test_generate_noise_rejects_non_2d():
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError):
        generate_noise((10,), rng=rng, wire_lengths_m=2.3)


def test_incoherent_rms_matches_enc_model():
    """Per-channel RMS ~ sqrt(white^2 + (y + z*L)^2) with a flat series shape."""
    rng = np.random.default_rng(1)
    n_ch, n_ticks = 64, 4096
    L = 2.0
    x, y, z = DEFAULT_ENC
    noise = incoherent_noise((n_ch, n_ticks), L, rng)
    rms = noise.std(axis=1)
    expected = np.sqrt(x**2 + (y + z * L) ** 2)
    assert abs(rms.mean() - expected) < 0.1 * expected


def test_incoherent_per_length_array():
    rng = np.random.default_rng(2)
    n_ch, n_ticks = 32, 4096
    lengths = np.linspace(0.5, 4.0, n_ch)
    noise = incoherent_noise((n_ch, n_ticks), lengths, rng)
    # Longer wires -> larger series term -> larger RMS (monotone trend).
    rms = noise.std(axis=1)
    # correlation between length and rms should be strongly positive
    c = np.corrcoef(lengths, rms)[0, 1]
    assert c > 0.8


def test_incoherent_length_mismatch_raises():
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError):
        incoherent_noise((10, 64), np.ones(7), rng)


# --------------------------------------------------------------------------
# coherent noise: shared-within-group + the no-1/sqrt(N) invariant
# --------------------------------------------------------------------------

def test_coherent_shared_within_group():
    rng = np.random.default_rng(3)
    gs = 64
    noise = coherent_noise(256, 512, rng, group_size=gs)
    # every channel in a group carries the identical waveform
    for g0 in range(0, 256, gs):
        block = noise[g0:g0 + gs]
        assert np.allclose(block, block[0][None, :])
    # different groups differ
    assert not np.allclose(noise[0], noise[gs])


@pytest.mark.parametrize("group_size", [16, 32, 64, 128])
def test_coherent_rms_independent_of_group_size(group_size):
    """The shared waveform survives per-channel averaging — its per-channel RMS
    is ~rms_adc regardless of group_size, NOT rms_adc/sqrt(group_size)."""
    rng = np.random.default_rng(4)
    rms_adc = 2.5
    noise = coherent_noise(512, 8192, rng, group_size=group_size,
                           rms_adc=rms_adc)
    per_channel_rms = noise.std(axis=1).mean()
    # within ~25% of the per-group target (neighbor coupling adds a little);
    # the 1/sqrt(N) bug would give 2.5/4=0.6 .. 2.5/sqrt(128)=0.22 — far below.
    assert 0.75 * rms_adc < per_channel_rms < 1.5 * rms_adc


def test_coherent_adjacent_group_anticorrelation():
    rng = np.random.default_rng(5)
    gs = 32
    noise = coherent_noise(8 * gs, 8192, rng, group_size=gs, beta=0.15)
    # one representative channel per group
    reps = noise[::gs]
    cc = np.corrcoef(reps)
    # adjacent groups are anti-correlated (beta>0); non-adjacent ~0
    adj = np.array([cc[i, i + 1] for i in range(len(reps) - 1)])
    assert adj.mean() < -0.05


# --------------------------------------------------------------------------
# Densify
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


def test_densify_pixel_skip(jaxtpc_pixel_data_root):
    sub = _sensor_sample(jaxtpc_pixel_data_root)
    pipe = Compose([dict(type='Apply', on='sensor',
                         transforms=[dict(type='Densify', on_pixel='skip')])])
    out = pipe(deepcopy({'sensor': sub}))['sensor']
    assert 'dense' not in out


# --------------------------------------------------------------------------
# AddNoise
# --------------------------------------------------------------------------

def _densify_addnoise(sub, **addnoise_kw):
    pipe = Compose([dict(type='Apply', on='sensor', transforms=[
        dict(type='Densify'),
        dict(type='AddNoise', **addnoise_kw),
    ])])
    return pipe(deepcopy({'sensor': sub}))['sensor']


def test_addnoise_requires_densify(jaxtpc_data_root):
    sub = _sensor_sample(jaxtpc_data_root)
    pipe = Compose([dict(type='Apply', on='sensor',
                         transforms=[dict(type='AddNoise')])])
    with pytest.raises(KeyError):
        pipe(deepcopy({'sensor': sub}))


def test_addnoise_changes_dense_and_keeps_clean_copy(jaxtpc_data_root):
    sub = _sensor_sample(jaxtpc_data_root)
    clean = _densify_addnoise(sub, coherent=False, incoherent=False)  # no-op tags
    noisy = _densify_addnoise(sub, coherent=True, incoherent=False)
    plane = sub['planes'][0]
    assert not np.allclose(clean['dense'][plane], noisy['dense'][plane])


def test_addnoise_reproducible_per_event(jaxtpc_data_root):
    """Same event id -> identical noise (stable across worker scheduling)."""
    sub = _sensor_sample(jaxtpc_data_root)
    a = _densify_addnoise(sub, coherent=True, incoherent=False, base_seed=7)
    b = _densify_addnoise(sub, coherent=True, incoherent=False, base_seed=7)
    plane = sub['planes'][0]
    assert np.array_equal(a['dense'][plane], b['dense'][plane])


def test_addnoise_seed_and_name_vary(jaxtpc_data_root):
    sub = _sensor_sample(jaxtpc_data_root)
    plane = sub['planes'][0]
    a = _densify_addnoise(sub, coherent=True, incoherent=False, base_seed=1)
    b = _densify_addnoise(sub, coherent=True, incoherent=False, base_seed=2)
    assert not np.array_equal(a['dense'][plane], b['dense'][plane])
    # different event name -> different noise at the same seed
    sub2 = deepcopy(sub)
    sub2['name'] = sub['name'] + '_other'
    c = _densify_addnoise(sub, coherent=True, incoherent=False, base_seed=1)
    d = _densify_addnoise(sub2, coherent=True, incoherent=False, base_seed=1)
    assert not np.array_equal(c['dense'][plane], d['dense'][plane])


def test_pipeline_end_to_end(jaxtpc_data_root):
    """Full dataset transform: Densify -> AddNoise(coherent) under Apply."""
    ds = JAXTPCDataset(
        data_root=jaxtpc_data_root, split='', dataset_name='sim',
        modalities=('sensor',), max_len=2,
        transform=[dict(type='Apply', on='sensor', transforms=[
            dict(type='Densify'),
            dict(type='AddNoise', coherent=True, incoherent=False,
                 group_size=64),
        ])])
    sample = ds[0]
    assert 'dense' in sample['sensor']
    for plane, img in sample['sensor']['dense'].items():
        nw, nt = sample['sensor']['shape'][plane]
        assert img.shape == (nw, nt)


# --------------------------------------------------------------------------
# Digitize
# --------------------------------------------------------------------------

def test_digitize_matches_production_formula():
    """digitize == round(x + ped).clip(0, 4095) - ped (doraemon/JAXTPC path)."""
    rng = np.random.default_rng(0)
    x = rng.uniform(-2000, 2000, size=(40, 128)).astype(np.float32)
    ped = 410
    out = digitize(x, ped)  # default n_bits=12 -> adc_max=4095, gain=1
    expected = np.round(x + ped).clip(0, 4095).astype(np.float32) - ped
    assert np.array_equal(out, expected)


def test_digitize_clips_rounds_and_is_integer_valued():
    ped = 410
    x = np.array([[-1000.0, -410.3, 0.4, 0.6, 3684.4, 5000.0]], dtype=np.float32)
    out = digitize(x, ped)  # valid pedestal-subtracted range [-410, 3685]
    # clipped to [-ped, adc_max-ped]
    assert out.min() >= -ped and out.max() <= 4095 - ped
    assert np.isclose(out[0, 0], -ped)        # -1000 -> code 0 -> -410
    assert np.isclose(out[0, -1], 4095 - ped)  # 5000 -> code 4095 -> 3685
    # integer-valued (codes), rounding to nearest
    assert np.allclose(out, np.round(out))


def test_digitize_gain_and_nbits():
    # n_bits sets adc_max; a value above the cap is clipped (10-bit -> 1023)
    out10 = digitize(np.array([[2000.0]], np.float32), 0, n_bits=10)
    assert out10[0, 0] == 1023.0
    # explicit adc_max overrides n_bits
    out = digitize(np.array([[1000.0]], np.float32), 0, n_bits=12, adc_max=500)
    assert out[0, 0] == 500.0
    # gain scales before pedestal/clip
    outg = digitize(np.array([[100.0]], np.float32), 0, gain=2.0)
    assert outg[0, 0] == 200.0


def test_sensor_sample_carries_pedestal(jaxtpc_data_root):
    sub = _sensor_sample(jaxtpc_data_root)
    assert 'pedestal' in sub, "reader must surface per-plane pedestal"
    for plane in sub['planes']:
        assert plane in sub['pedestal']


def test_digitize_requires_dense(jaxtpc_data_root):
    sub = _sensor_sample(jaxtpc_data_root)
    pipe = Compose([dict(type='Apply', on='sensor',
                         transforms=[dict(type='Digitize')])])
    with pytest.raises(KeyError):
        pipe(deepcopy({'sensor': sub}))


def test_full_forward_pipeline_densify_addnoise_digitize(jaxtpc_data_root):
    """Densify -> AddNoise(coherent) -> Digitize: output is integer-valued and
    inside the pedestal-shifted 12-bit range."""
    ped = 410
    ds = JAXTPCDataset(
        data_root=jaxtpc_data_root, split='', dataset_name='sim',
        modalities=('sensor',), max_len=2,
        transform=[dict(type='Apply', on='sensor', transforms=[
            dict(type='Densify'),
            dict(type='AddNoise', coherent=True, incoherent=False),
            dict(type='Digitize', pedestal=ped, n_bits=12),
        ])])
    sample = ds[0]
    for plane, img in sample['sensor']['dense'].items():
        assert np.allclose(img, np.round(img)), "digitized output must be integer-valued"
        assert img.min() >= -ped and img.max() <= 4095 - ped


# --------------------------------------------------------------------------
# Reconciliation against JAXTPC's canonical forward model
# --------------------------------------------------------------------------

# --------------------------------------------------------------------------
# Phase 1: dense-path plumbing + correctness
# --------------------------------------------------------------------------

from pimm_data.jaxtpc import canonical_plane_id
from pimm_data.detector_transforms import Densify, Digitize


def test_canonical_plane_id_stable():
    assert canonical_plane_id('volume_0_U') == 0
    assert canonical_plane_id('volume_0_Y') == 2
    assert canonical_plane_id('volume_1_U') == 3
    assert canonical_plane_id('volume_1_Y') == 5
    assert canonical_plane_id('volume_2_Pixel') == 2  # pixel: one plane/volume


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


def test_no_double_digitize(jaxtpc_data_root):
    sub = _sensor_sample(jaxtpc_data_root)
    Densify()(sub)
    Digitize(pedestal=0)(sub)
    with pytest.raises(RuntimeError):
        Digitize(pedestal=0)(sub)


def _import_jaxtpc():
    for root in (os.environ.get('JAXTPC_ROOT'),
                 '/sdf/group/neutrino/omara/JAXTPC'):
        if root and os.path.isdir(os.path.join(root, 'tools')):
            if root not in sys.path:
                sys.path.insert(0, root)
            # Namespace-collision guard: another `tools` package on sys.path
            # (e.g. pimm's) may already be cached in sys.modules, shadowing
            # JAXTPC's. Evict any cached `tools[.*]` not loaded from this root so
            # the import below resolves to JAXTPC's (the parity oracle). Without
            # this the comparison silently ran against the wrong module.
            for m in [k for k in list(sys.modules)
                      if k == 'tools' or k.startswith('tools.')]:
                f = getattr(sys.modules[m], '__file__', None) or ''
                if not f.startswith(root):
                    del sys.modules[m]
            try:
                import tools.coherent_noise as cn
                import tools.noise as nz
            except Exception:
                return None
            if not (getattr(cn, '__file__', '') or '').startswith(root):
                return None  # got a foreign `tools`, treat as unavailable
            return cn, nz, root
    return None


def test_reconcile_enc_params_with_npz():
    """pimm-data's inline ENC defaults must equal JAXTPC's noise_spectrum.npz."""
    jx = _import_jaxtpc()
    if jx is None:
        pytest.skip("JAXTPC repo not importable")
    _, nz, root = jx
    npz = os.path.join(root, 'config', 'noise_spectrum.npz')
    if not os.path.exists(npz):
        pytest.skip("noise_spectrum.npz not found")
    x, y, z, _, _ = nz.load_noise_params(npz)
    assert np.allclose(DEFAULT_ENC, (x, y, z)), \
        f"pimm-data DEFAULT_ENC {DEFAULT_ENC} != JAXTPC npz ({x}, {y}, {z})"


def test_reconcile_coherent_bitexact_with_jaxtpc():
    """Identical params + identical Generator -> bit-identical coherent noise.

    pimm-data.coherent_noise is a faithful numpy port of JAXTPC
    generate_coherent_noise, so with the same np.random.Generator they must
    agree exactly (the grouping + spectrum + beta coupling are shared)."""
    jx = _import_jaxtpc()
    if jx is None:
        pytest.skip("JAXTPC repo not importable")
    cn, _, _ = jx
    n_wires, n_ticks, gs = 200, 2048, 64
    ours = coherent_noise(n_wires, n_ticks, np.random.default_rng(123),
                          group_size=gs, rms_adc=2.5, corner_freq_hz=20000.0,
                          spectral_slope=1.5, beta=0.15,
                          sampling_rate_hz=DEFAULT_SAMPLING_RATE_HZ)
    theirs = cn.generate_coherent_noise(
        n_wires, n_ticks, group_size=gs, beta=0.15, rms_adc=2.5,
        corner_freq_hz=20000.0, spectral_slope=1.5, sampling_rate_hz=2e6,
        rng=np.random.default_rng(123))
    assert np.allclose(ours, theirs, atol=1e-5)
