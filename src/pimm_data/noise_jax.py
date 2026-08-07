"""JAX/GPU noise generation — the jitted twin of :mod:`pimm_data.noise`.

Same forward model, same parameters, same estimators; only the RNG stream
differs (JAX counter-based keys vs numpy Generator), so output is *statistically*
identical rather than bit-identical.

Provenance:
* the **incoherent** (intrinsic) component mirrors JAXTPC
  ``tools.noise._noise_core`` — which is itself already JAX and is the
  authoritative forward model. The numpy :func:`pimm_data.noise.incoherent_noise`
  mirrors the same function, so all three agree by construction.
* the **coherent** component ports :func:`pimm_data.noise.coherent_noise`
  (itself a port of JAXTPC ``tools.coherent_noise``), including the two details
  that are easy to get wrong: the adjacent-group anti-correlation is applied
  BEFORE normalisation, and the RMS renormalisation uses the **pooled-global**
  measured RMS over all groups (not per-group).

**Everything numeric runs inside ``jax.jit``.** Left as plain ``jnp`` calls these
run one kernel at a time, where JAX's per-op dispatch is ~5x torch's and nothing
fuses: measured 11.2 ms (coherent) and 10.0 ms (incoherent) per 1969x4321 plane.
Jitted they are 0.42 ms and 2.98 ms — *faster* than the torch equivalents in
``dense_ops`` (1.00 / 3.48 ms). Only shapes and scalars are static, so a detector
compiles each core once.

jax is imported lazily at call time, so importing pimm-data never pulls it in.
"""

import numpy as np

from .noise import (DEFAULT_SERIES_SPECTRUM, resolve_series_spectrum,
                    
    DEFAULT_ENC, DEFAULT_SAMPLING_RATE_HZ, DEFAULT_GROUP_SIZE,
    DEFAULT_COH_RMS_ADC, DEFAULT_COH_CORNER_FREQ_HZ, DEFAULT_COH_SLOPE,
    DEFAULT_COH_BETA, _series_spectrum_shape, _coherent_spectrum,
)

__all__ = ["incoherent_noise_jax", "coherent_noise_jax", "generate_noise_jax"]

_CORES = None


def _jnp():
    import jax.numpy as jnp
    return jnp


def _cores():
    """Build (and cache) the jitted incoherent/coherent cores."""
    global _CORES
    if _CORES is None:
        import functools
        import jax
        import jax.numpy as jnp

        def _rfft_normal(key, shape, spec, n_ticks):
            """N(0,1) real/imag shaped by ``spec``, DC (and Nyquist for even
            ``n_ticks``) forced real, then irfft."""
            k_re, k_im = jax.random.split(key, 2)
            real = jax.random.normal(k_re, shape) * spec
            imag = jax.random.normal(k_im, shape) * spec
            cpx = real + 1j * imag
            cpx = cpx.at[..., 0].set(cpx[..., 0].real)
            if n_ticks % 2 == 0:
                cpx = cpx.at[..., -1].set(cpx[..., -1].real)
            return jnp.fft.irfft(cpx, n=n_ticks, axis=-1)

        @functools.partial(jax.jit, static_argnames=("n_ch", "n_ticks"))
        def _inc(key, spec, series_rms, white_x, n_ch, n_ticks):
            k_s, k_w = jax.random.split(key, 2)
            shaped = _rfft_normal(k_s, (n_ch, spec.shape[0]), spec[None, :], n_ticks)
            cur = jnp.maximum(jnp.std(shaped, axis=1, keepdims=True), 1e-10)
            shaped = shaped / cur * series_rms[:, None]
            white = jax.random.normal(k_w, (n_ch, n_ticks)) * white_x
            return (shaped + white).astype(jnp.float32)

        @functools.partial(jax.jit, static_argnames=("n_ch", "n_ticks", "gs"))
        def _coh(key, spec, rms_adc, beta, n_ch, n_ticks, gs):
            n_groups = (n_ch + gs - 1) // gs
            base = _rfft_normal(key, (n_groups, spec.shape[0]), spec[None, :], n_ticks)
            z = jnp.zeros((1, n_ticks), base.dtype)
            left = jnp.concatenate([z, base[:-1]], axis=0)
            right = jnp.concatenate([base[1:], z], axis=0)
            wav = base - beta * (left + right)
            realized = jnp.sqrt(jnp.mean(wav.astype(jnp.float32) ** 2))
            wav = jnp.where(realized > 0,
                            wav * (rms_adc / jnp.maximum(realized, 1e-30)), wav)
            return wav[jnp.arange(n_ch) // gs].astype(jnp.float32)

        _CORES = (_inc, _coh)
    return _CORES


def incoherent_noise_jax(key, n_channels, n_ticks, wire_lengths_m, *,
                         enc=DEFAULT_ENC,
                         series_spectrum=DEFAULT_SERIES_SPECTRUM,
                         sampling_rate_hz=DEFAULT_SAMPLING_RATE_HZ):
    """Per-channel independent noise ``(n_channels, n_ticks)`` [ADC], on device.

    Frequency-shaped series component renormalised per channel to
    ``series_rms = y + z*L``, plus a flat white component with RMS ``x``.
    ``series_spectrum=(freqs_hz, amps)`` gives the COLORED spectrum; ``None`` is
    white (the old FM-corpus default, and the bug that motivated the rebuild).
    """
    jnp = _jnp()
    white_x, series_y, series_z = enc
    L = np.asarray(wire_lengths_m, dtype=np.float64).reshape(-1)
    if L.shape[0] == 1:
        L = np.full(n_channels, L[0])
    if L.shape[0] != n_channels:
        raise ValueError(
            f"wire_lengths_m has {L.shape[0]} entries but {n_channels} channels")
    series_rms = jnp.asarray((series_y + series_z * L).astype(np.float32))
    spec_np = _series_spectrum_shape(n_ticks, series_spectrum, sampling_rate_hz)
    spec = (jnp.ones(n_ticks // 2 + 1, jnp.float32) if spec_np is None
            else jnp.asarray(np.asarray(spec_np, np.float32)))
    return _cores()[0](key, spec, series_rms, float(white_x),
                       int(n_channels), int(n_ticks))


def coherent_noise_jax(key, n_channels, n_ticks, *, group_size=DEFAULT_GROUP_SIZE,
                       rms_adc=DEFAULT_COH_RMS_ADC,
                       corner_freq_hz=DEFAULT_COH_CORNER_FREQ_HZ,
                       spectral_slope=DEFAULT_COH_SLOPE, beta=DEFAULT_COH_BETA,
                       sampling_rate_hz=DEFAULT_SAMPLING_RATE_HZ):
    """Per-group shared waveform broadcast to channels ``(n_channels, n_ticks)``.

    Adjacent-group anti-correlation ``w'(g) = w(g) - beta*(w(g-1) + w(g+1))``,
    then renormalised to ``rms_adc`` by the POOLED-GLOBAL measured RMS **after**
    the coupling (matching :func:`pimm_data.noise.coherent_noise` / JAXTPC).
    """
    jnp = _jnp()
    spec = jnp.asarray(np.asarray(
        _coherent_spectrum(n_ticks, corner_freq_hz, spectral_slope, sampling_rate_hz),
        np.float32))
    return _cores()[1](key, spec, float(rms_adc), float(beta),
                       int(n_channels), int(n_ticks), int(group_size))


def generate_noise_jax(key, shape, *, wire_lengths_m=None, incoherent=True,
                       coherent=False, enc=DEFAULT_ENC,
                       series_spectrum=DEFAULT_SERIES_SPECTRUM,
                       sampling_rate_hz=DEFAULT_SAMPLING_RATE_HZ,
                       group_size=DEFAULT_GROUP_SIZE, coh_rms=DEFAULT_COH_RMS_ADC,
                       coh_corner_freq_hz=DEFAULT_COH_CORNER_FREQ_HZ,
                       coh_spectral_slope=DEFAULT_COH_SLOPE, beta=DEFAULT_COH_BETA):
    """Dense ``(n_channels, n_ticks)`` noise [ADC] on device — the jax twin of
    :func:`pimm_data.noise.generate_noise`. Returns the NOISE; the caller adds it.
    """
    import jax
    jnp = _jnp()
    n_ch, n_ticks = int(shape[0]), int(shape[1])
    out = jnp.zeros((n_ch, n_ticks), jnp.float32)
    k_inc, k_coh = jax.random.split(key, 2)
    if incoherent:
        if wire_lengths_m is None:
            raise ValueError("incoherent=True requires wire_lengths_m (meters)")
        out = out + incoherent_noise_jax(
            k_inc, n_ch, n_ticks, wire_lengths_m, enc=enc,
            series_spectrum=series_spectrum, sampling_rate_hz=sampling_rate_hz)
    if coherent:
        out = out + coherent_noise_jax(
            k_coh, n_ch, n_ticks, group_size=group_size, rms_adc=coh_rms,
            corner_freq_hz=coh_corner_freq_hz, spectral_slope=coh_spectral_slope,
            beta=beta, sampling_rate_hz=sampling_rate_hz)
    return out
