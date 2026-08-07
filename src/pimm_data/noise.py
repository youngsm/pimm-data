"""
Forward noise model for LArTPC wire planes — a tagged numpy function.

This is pimm-data's lightweight, load-time noise applicator. It mirrors the
forward noise model that lives in JAXTPC (``tools/noise.py`` incoherent +
``tools/coherent_noise.py`` coherent) but is implemented in plain numpy with no
JAX / torch dependency, so it runs inside DataLoader workers.

Two additive components, selected by tags:

* **incoherent** — per-channel *independent* noise. ENC = ``sqrt(white^2 +
  (y + z*L)^2)`` (MicroBooNE model, arXiv:1705.07341). A shaped "series"
  component (renormalised to ``series_rms`` per channel, exactly as JAXTPC
  ``_noise_core`` does) plus a flat white component.
* **coherent** — a per-GROUP *shared* waveform: ``group_size`` channels share
  one realization. RMS is per-group and is **NOT** divided by ``sqrt(N)`` — the
  shared waveform must survive per-channel averaging, which is exactly what the
  inverse (``helix.tpc.remove_coherent``'s per-group median) is built to kill.

Parameters are **inline defaults** documented to match JAXTPC's
``config/noise_spectrum.npz`` + the YAML coherent block; ``tests/test_noise.py``
pins them with a reconciliation test. The grouping convention
(``arange(n) // group_size``) is identical to JAXTPC ``broadcast_to_wires`` and
helix ``broadcast_groups`` — that single integer is the whole "don't drift"
contract between forward injection and inverse removal.
"""

from __future__ import annotations

import numpy as np

# ── Inline defaults — MUST match JAXTPC config/noise_spectrum.npz + the YAML
#    simulation.coherent_noise block (tests/test_noise.py asserts this). ──
# (white_x [ADC], series_y [ADC], series_z [ADC/m]); wire length L is in METERS.
DEFAULT_ENC = (0.90, 0.79, 0.22)
DEFAULT_SAMPLING_RATE_HZ = 2.0e6  # 2 MHz (0.5 us / tick)
# coherent component (JAXTPC tools/coherent_noise.py defaults)
DEFAULT_COH_RMS_ADC = 2.5
DEFAULT_COH_CORNER_FREQ_HZ = 20_000.0
DEFAULT_COH_SLOPE = 1.5
DEFAULT_COH_BETA = 0.15
DEFAULT_GROUP_SIZE = 64

#: Sentinel for the MEASURED series spectrum shipped in ``data/``. It is the
#: default because white was the wrong answer and was silently easy to get: the
#: research pipeline called ``AddNoise(coherent=True, incoherent=True)`` without
#: naming a spectrum, got flat noise, and trained a foundation model on it. The
#: measured spectrum concentrates power in 10-490 kHz and has NONE above 500 kHz,
#: so white misplaces noise across every wavelet band (see helix's
#: MULTI_EVENT_BATCHING.md sibling note in the corpus design docs).
#:
#: Pass ``series_spectrum=None`` for white — still available, but now an explicit
#: choice rather than what you get by saying nothing.
DEFAULT_SERIES_SPECTRUM = "microboone"

_SPECTRA = {"microboone": "noise_spectrum.npz"}
_spectrum_cache = {}


def load_series_spectrum(name=DEFAULT_SERIES_SPECTRUM):
    """Named spectrum -> ``(freqs_hz, amps)`` from the packaged ``data/`` file."""
    if name not in _SPECTRA:
        raise KeyError(f"unknown series spectrum {name!r}; known: {sorted(_SPECTRA)}")
    if name not in _spectrum_cache:
        import os
        path = os.path.join(os.path.dirname(__file__), "data", _SPECTRA[name])
        with np.load(path) as d:
            _spectrum_cache[name] = (d["spectrum_freqs_hz"].copy(),
                                     d["spectrum_shape"].copy())
    return _spectrum_cache[name]


def resolve_series_spectrum(spec):
    """``None`` -> white; a name -> the packaged measurement; a tuple -> itself."""
    if spec is None or isinstance(spec, tuple) and len(spec) == 2 and not isinstance(spec[0], str):
        return spec
    if isinstance(spec, str):
        return load_series_spectrum(spec)
    return spec


def _series_spectrum_shape(n_ticks, series_spectrum, sampling_rate_hz):
    """Interpolated series amplitude spectrum (shape only).

    The absolute scale is irrelevant: ``incoherent_noise`` renormalises the
    series component to ``series_rms`` per channel (mirroring JAXTPC
    ``_noise_core``), so only the spectral *shape* matters. ``series_spectrum``
    is ``(freqs_hz, amps)`` (as in noise_spectrum.npz) or ``None`` → flat/white.
    """
    series_spectrum = resolve_series_spectrum(series_spectrum)
    if series_spectrum is None:
        return None
    freqs_emp, amps_emp = series_spectrum
    freqs = np.fft.rfftfreq(n_ticks, d=1.0 / sampling_rate_hz)
    return np.interp(freqs, np.asarray(freqs_emp, dtype=np.float64),
                     np.asarray(amps_emp, dtype=np.float64))


def incoherent_noise(shape, wire_lengths_m, rng, *, enc=DEFAULT_ENC,
                     series_spectrum=DEFAULT_SERIES_SPECTRUM,
                     sampling_rate_hz=DEFAULT_SAMPLING_RATE_HZ):
    """Per-channel independent noise, shape ``(n_channels, n_ticks)`` [ADC].

    Mirrors JAXTPC ``tools.noise._noise_core``: a frequency-shaped series
    component renormalised to ``series_rms = y + z*L`` per channel, plus a flat
    white component with RMS ``x``. ``wire_lengths_m`` is a scalar (uniform) or
    a ``(n_channels,)`` array, in METERS.
    """
    n_ch, n_ticks = int(shape[0]), int(shape[1])
    white_x, series_y, series_z = enc

    L = np.asarray(wire_lengths_m, dtype=np.float64).reshape(-1)
    if L.shape[0] == 1:
        L = np.full(n_ch, L[0])
    if L.shape[0] != n_ch:
        raise ValueError(
            f"wire_lengths_m has {L.shape[0]} entries but image has {n_ch} "
            f"channels (pass a scalar for uniform length or one value/channel)")
    series_rms = (series_y + series_z * L).astype(np.float64)  # (n_ch,)

    n_freq = n_ticks // 2 + 1
    spec = _series_spectrum_shape(n_ticks, series_spectrum, sampling_rate_hz)

    real = rng.standard_normal((n_ch, n_freq))
    imag = rng.standard_normal((n_ch, n_freq))
    if spec is not None:
        real = real * spec[None, :]
        imag = imag * spec[None, :]
    cpx = real + 1j * imag
    cpx[:, 0] = cpx[:, 0].real           # DC must be real
    if n_ticks % 2 == 0:
        cpx[:, -1] = cpx[:, -1].real     # Nyquist must be real (even n)

    shaped = np.fft.irfft(cpx, n=n_ticks, axis=1)
    cur = np.maximum(np.std(shaped, axis=1, keepdims=True), 1e-10)
    shaped = shaped / cur * series_rms[:, None]

    white = rng.standard_normal((n_ch, n_ticks)) * white_x
    return (shaped + white).astype(np.float32)


def _coherent_spectrum(n_ticks, corner_freq_hz, spectral_slope, sampling_rate_hz):
    """A(f) = 1 / (1 + f/f_corner)^(slope/2), with A(0) = 0 (no DC)."""
    freqs = np.fft.rfftfreq(n_ticks, d=1.0 / sampling_rate_hz)
    spec = 1.0 / (1.0 + freqs / corner_freq_hz) ** (spectral_slope / 2.0)
    spec[0] = 0.0
    return spec.astype(np.float64)


def coherent_noise(n_channels, n_ticks, rng, *, group_size=DEFAULT_GROUP_SIZE,
                   rms_adc=DEFAULT_COH_RMS_ADC,
                   corner_freq_hz=DEFAULT_COH_CORNER_FREQ_HZ,
                   spectral_slope=DEFAULT_COH_SLOPE, beta=DEFAULT_COH_BETA,
                   sampling_rate_hz=DEFAULT_SAMPLING_RATE_HZ):
    """Per-group shared waveform broadcast to channels, ``(n_channels, n_ticks)``.

    Faithful numpy port of JAXTPC ``tools.coherent_noise.generate_group_waveforms``
    + ``broadcast_to_wires``: one waveform per group with adjacent-group
    anti-correlation (``w'(g) = w(g) - beta*(w(g-1)+w(g+1))``), then per-group RMS
    renormalised to ``rms_adc`` **after** the coupling by the *measured* realized RMS
    (JAXTPC 6998d81 — the coupling inflates variance, so analytic/Parseval norm of the
    pre-coupling spectrum runs ~2% high at the default beta).
    """
    n_groups = (n_channels + group_size - 1) // group_size
    spec = _coherent_spectrum(n_ticks, corner_freq_hz, spectral_slope,
                              sampling_rate_hz)
    n_freq = len(spec)

    base = np.empty((n_groups, n_ticks), dtype=np.float32)
    for g in range(n_groups):
        real = rng.standard_normal(n_freq) * spec
        imag = rng.standard_normal(n_freq) * spec
        cpx = real + 1j * imag
        cpx[0] = cpx[0].real
        if n_ticks % 2 == 0:
            cpx[-1] = cpx[-1].real
        base[g] = np.fft.irfft(cpx, n=n_ticks)

    z = np.zeros((1, n_ticks), dtype=np.float32)
    left = np.concatenate([z, base[:-1]], axis=0)
    right = np.concatenate([base[1:], z], axis=0)
    waveforms = base - beta * (left + right)

    # normalize AFTER coupling, by the measured realized RMS (matches JAXTPC).
    realized = float(np.sqrt(np.mean(waveforms.astype(np.float64) ** 2)))
    if realized > 0:
        waveforms = waveforms * (rms_adc / realized)

    wire_to_group = np.arange(n_channels) // group_size
    return waveforms[wire_to_group].astype(np.float32)


def generate_noise(shape, *, rng, wire_lengths_m=None, incoherent=True,
                   coherent=False, enc=DEFAULT_ENC,
                   series_spectrum=DEFAULT_SERIES_SPECTRUM,
                   sampling_rate_hz=DEFAULT_SAMPLING_RATE_HZ,
                   group_size=DEFAULT_GROUP_SIZE, coh_rms=DEFAULT_COH_RMS_ADC,
                   coh_corner_freq_hz=DEFAULT_COH_CORNER_FREQ_HZ,
                   coh_spectral_slope=DEFAULT_COH_SLOPE, beta=DEFAULT_COH_BETA):
    """Return a dense ``(n_channels, n_ticks)`` noise array [ADC].

    Plain numpy (NOT jitted). Returns the **noise** — the caller adds it to the
    signal::

        noisy = image + generate_noise(image.shape, rng=rng, coherent=True)

    Returning the noise (rather than the noisy image) keeps generation decoupled
    from application: it is easy to inspect/reuse/save the realization and robust
    to the caller's dtype / in-place choices. The ``incoherent`` / ``coherent``
    tags select the additive components; ``rng`` is an explicit
    ``np.random.Generator`` (the caller owns reproducibility).

    JAXTPC production output already carries incoherent noise, so the typical
    load-time use is ``incoherent=False, coherent=True`` (add the component
    JAXTPC omits by default); generate the full model only for noise-free input.
    ``wire_lengths_m`` (scalar or per-channel, METERS) is required only when
    ``incoherent=True``.
    """
    if len(shape) != 2:
        raise ValueError(f"generate_noise expects a 2-D (n_channels, n_ticks) "
                         f"shape, got {tuple(shape)}")
    n_ch, n_ticks = int(shape[0]), int(shape[1])
    noise = np.zeros((n_ch, n_ticks), dtype=np.float32)
    if incoherent:
        if wire_lengths_m is None:
            raise ValueError("incoherent=True requires wire_lengths_m "
                             "(scalar or per-channel, in meters)")
        noise += incoherent_noise((n_ch, n_ticks), wire_lengths_m, rng, enc=enc,
                                  series_spectrum=series_spectrum,
                                  sampling_rate_hz=sampling_rate_hz)
    if coherent:
        noise += coherent_noise(n_ch, n_ticks, rng,
                                group_size=group_size, rms_adc=coh_rms,
                                corner_freq_hz=coh_corner_freq_hz,
                                spectral_slope=coh_spectral_slope, beta=beta,
                                sampling_rate_hz=sampling_rate_hz)
    return noise


def digitize(signal, pedestal, *, n_bits=12, adc_max=None, gain=1.0):
    """Quantize a pedestal-subtracted ADC image to integer codes.

    The production digitization step (mirrors JAXTPC ``electronics._digitize_signal``
    and the doraemon ``make_noisy`` path): add the pedestal back, optionally
    scale by ``gain``, round to the nearest integer, clip to the valid code range
    ``[0, adc_max]``, then subtract the pedestal again so the result stays
    pedestal-subtracted::

        out = round(signal*gain + pedestal).clip(0, adc_max) - pedestal

    ``adc_max`` defaults to ``(1 << n_bits) - 1`` (12-bit → 4095). Apply *after*
    noise (``image + generate_noise(...)``), as the last forward step. Returns a
    ``float32`` array, integer-valued, in ``[-pedestal, adc_max - pedestal]``.
    """
    x = np.asarray(signal, dtype=np.float32)
    amax = float(adc_max) if adc_max is not None else float((1 << int(n_bits)) - 1)
    raw = np.round(x * gain + pedestal)
    return np.clip(raw, 0.0, amax).astype(np.float32) - pedestal
