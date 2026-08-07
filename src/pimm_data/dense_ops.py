"""
Device-agnostic torch ops for the post-collate dense sensor path.

These run on whatever device the input tensors live on (CPU or CUDA) — the SAME
code is the CPU and GPU implementation; "CPU vs GPU" is purely where the batch
sits relative to ``.to(device)``. This module imports ``torch`` but **never**
``torch.cuda`` — device follows the batch (born-on-GPU: only the sparse hits cross
PCIe; the dense grids are created here, on-device).

Three ops, used post-collate by :mod:`pimm_data.batch_transforms`:

* ``densify`` — scatter sparse hits into per-plane dense grids ``{plane_id: (B, W_p, T)}``
  via ``index_add_`` (== last-wins assignment on the unique COO this assumes; the
  numpy reference :class:`pimm_data.detector_transforms.Densify` asserts uniqueness).
* ``add_intrinsic_noise`` — add fresh forward noise. **Both components are torch-FFT
  ports running on-device** (coherent: :func:`_coherent_torch`; incoherent:
  :func:`_incoherent_torch`) — statistical parity to the numpy forward model only
  (torch RNG ≠ numpy, and torch CPU vs CUDA streams differ → realizations are
  **device-specific**, not bit-exact to JAXTPC). ``coherent_numpy=True`` opts back into
  the bit-exact / device-independent numpy coherent oracle (slower: per-group Python
  loop + H2D copy). Uses ``std(unbiased=False)`` to match the numpy ddof=0 renorm.
* ``digitize`` — quantize to ADC codes (bit-exact to :func:`pimm_data.noise.digitize`).
"""

from __future__ import annotations

import numpy as np
import torch

from .noise import (coherent_noise as _coherent_noise_np, _series_spectrum_shape,
                    DEFAULT_SERIES_SPECTRUM, resolve_series_spectrum,
                    DEFAULT_ENC, DEFAULT_SAMPLING_RATE_HZ, DEFAULT_GROUP_SIZE,
                    DEFAULT_COH_RMS_ADC, DEFAULT_COH_CORNER_FREQ_HZ,
                    DEFAULT_COH_SLOPE, DEFAULT_COH_BETA)

# decorrelate the per-component torch RNG streams from each other (and so a
# numpy-coherent run and a torch-coherent run never share a draw). Masked to 63
# bits so they don't fight the 63-bit seed mask in manual_seed.
_INCOH_SEED_SALT = 0x9E3779B97F4A7C15 & ((1 << 63) - 1)
_COH_SEED_SALT = 0xD1B54A32D192ED03 & ((1 << 63) - 1)


def offset2batch(offset):
    """Cumulative ``offset`` (B,) -> per-point batch index (ΣN,).

    Inverse of collate's cumsum. Empty events (zero-length runs) contribute no
    points and are handled by ``repeat_interleave`` on the diffed counts.
    """
    counts = torch.diff(offset, prepend=offset.new_zeros(1))
    return torch.repeat_interleave(
        torch.arange(offset.numel(), device=offset.device), counts)


def densify(wire, time, value, plane_id, offset, geom):
    """Scatter sparse hits into per-plane dense grids.

    Parameters
    ----------
    wire, time : (ΣN,) integer tensors — absolute grid indices (raw COO).
    value : (ΣN,) tensor — the per-hit ADC value.
    plane_id : (ΣN,) integer tensor — the CANONICAL plane id per hit.
    offset : (B,) integer tensor — cumulative per-sample hit counts (from collate).
    geom : ``{plane_id: {'n_wires': W, 'n_ticks': T}}``.

    Returns ``{plane_id: (B, W, T) float32}`` on ``wire.device``. Uses
    ``index_add_`` — on the unique COO this assumes, identical to last-wins
    assignment (so it matches the numpy reference); on the GPU it is collision-free
    for unique indices, hence deterministic.
    """
    if torch.is_floating_point(wire) or torch.is_floating_point(time):
        raise TypeError("densify: wire/time must be integer grid indices")
    # device-consistency (NOT CUDA-residency): the dense path is device-agnostic
    # and runs wherever its inputs live, but they must all live on ONE device.
    devs = {t.device for t in (wire, time, value, plane_id, offset)}
    if len(devs) > 1:
        raise ValueError(
            f"densify: inputs span multiple devices {devs}; move the whole "
            "batch to a single device before densify.")
    wire = wire.reshape(-1).long()
    time = time.reshape(-1).long()
    value = value.reshape(-1).to(torch.float32)
    plane_id = plane_id.reshape(-1)
    n = wire.numel()
    # length consistency: a coord-mutating/subsampling transform (e.g. GridSample)
    # on the sensor modality before densify desyncs the flat scatter inputs from
    # `offset` (derived from coord) — catch it loudly rather than silently
    # scatter a corrupted grid.
    if not (time.numel() == n and value.numel() == n and plane_id.numel() == n):
        raise ValueError(
            f"densify: wire/time/value/plane_id length mismatch "
            f"({n}/{time.numel()}/{value.numel()}/{plane_id.numel()}).")
    B = int(offset.numel())
    if B and int(offset[-1]) != n:
        raise ValueError(
            f"densify: offset total ({int(offset[-1])}) != n hits ({n}) — a "
            "coord-mutating transform likely ran on the sensor modality before "
            "densify; densify needs the immutable raw COO.")
    # every plane present in the batch must be in the geometry registry, else its
    # hits would be silently dropped (registry built from too few events / config).
    if n:
        present = set(int(g) for g in torch.unique(plane_id).tolist())
        missing = present - set(int(g) for g in geom)
        if missing:
            raise KeyError(
                f"densify: plane id(s) {sorted(missing)} not in the geometry "
                "registry — it must cover every plane present in the batch.")
    batch = offset2batch(offset)
    grids = {}
    for gid, e in geom.items():
        W, T = int(e['n_wires']), int(e['n_ticks'])
        m = plane_id == int(gid)
        buf = value.new_zeros(B * W * T)
        if bool(m.any()):
            b, w, t, v = batch[m], wire[m], time[m], value[m]
            # bounds check: a wrong/stale registry geometry would otherwise scatter
            # out of range (CUDA illegal access / silent corruption).
            if int(w.max()) >= W or int(w.min()) < 0 or int(t.max()) >= T or int(t.min()) < 0:
                raise ValueError(
                    f"densify: plane {gid} has wire/time outside the registry grid "
                    f"({W}x{T}) — geometry mismatch (config vs data).")
            flat = (b * W + w) * T + t
            buf.index_add_(0, flat, v)
        grids[int(gid)] = buf.view(B, W, T)
    return grids


def _series_spectrum_torch(n_ticks, series_spectrum, sampling_rate_hz, device):
    series_spectrum = resolve_series_spectrum(series_spectrum)
    spec = _series_spectrum_shape(n_ticks, series_spectrum, sampling_rate_hz)
    if spec is None:
        return None
    return torch.as_tensor(spec, dtype=torch.float32, device=device)


def _incoherent_torch(shape, wire_lengths_m, *, gen, enc, series_spectrum,
                      sampling_rate_hz, device):
    """Torch port of :func:`pimm_data.noise.incoherent_noise` (statistical parity).

    Mirrors the numpy path: shaped series renormalised to ``series_rms = y + z·L``
    per channel + flat white. ``std(unbiased=False)`` matches numpy ddof=0.
    """
    n_ch, n_ticks = int(shape[0]), int(shape[1])
    white_x, series_y, series_z = enc
    L = torch.as_tensor(wire_lengths_m, dtype=torch.float32, device=device).reshape(-1)
    if L.numel() == 1:
        L = L.expand(n_ch)
    if L.numel() != n_ch:
        raise ValueError(f"wire_lengths has {L.numel()} entries, expected {n_ch}")
    series_rms = (series_y + series_z * L)[:, None]
    n_freq = n_ticks // 2 + 1

    real = torch.randn(n_ch, n_freq, generator=gen, device=device)
    imag = torch.randn(n_ch, n_freq, generator=gen, device=device)
    spec = _series_spectrum_torch(n_ticks, series_spectrum, sampling_rate_hz, device)
    if spec is not None:
        real = real * spec
        imag = imag * spec
    cpx = torch.complex(real, imag)
    cpx[:, 0] = cpx[:, 0].real
    if n_ticks % 2 == 0:
        cpx[:, -1] = cpx[:, -1].real
    shaped = torch.fft.irfft(cpx, n=n_ticks, dim=1)
    cur = shaped.std(dim=1, unbiased=False, keepdim=True).clamp_min(1e-10)
    shaped = shaped / cur * series_rms

    white = torch.randn(n_ch, n_ticks, generator=gen, device=device) * white_x
    return (shaped + white).to(torch.float32)


def _coherent_spectrum_torch(n_ticks, corner_freq_hz, spectral_slope,
                             sampling_rate_hz, device):
    """A(f) = 1/(1 + f/f_corner)^(slope/2), A(0)=0 — torch twin of `_coherent_spectrum`."""
    freqs = torch.fft.rfftfreq(n_ticks, d=1.0 / sampling_rate_hz, device=device)
    spec = 1.0 / (1.0 + freqs / corner_freq_hz) ** (spectral_slope / 2.0)
    spec[0] = 0.0
    return spec.to(torch.float32)


def _coherent_torch(n_channels, n_ticks, *, gen, group_size, rms_adc,
                    corner_freq_hz, spectral_slope, beta, sampling_rate_hz, device):
    """On-device coherent noise (n_channels, n_ticks) — torch port of
    :func:`pimm_data.noise.coherent_noise`.

    Vectorised over groups (no Python per-group loop / no H2D copy): draws all
    ``n_groups`` shaped waveforms at once, applies adjacent-group anti-correlation
    ``w - beta*(w_{g-1}+w_{g+1})``, renormalises per-group RMS to ``rms_adc`` by the
    **measured realized RMS AFTER coupling** (matching the numpy oracle, JAXTPC
    6998d81 — NOT the analytic/Parseval pre-coupling norm, which runs ~2% high at
    the default beta), then broadcasts by ``wire//group_size``. Statistical parity
    to numpy only (torch RNG != numpy; CUDA vs CPU streams differ) — the realisation
    is device-specific, not bit-exact to JAXTPC, exactly as the incoherent port.
    """
    n_groups = (n_channels + group_size - 1) // group_size
    spec = _coherent_spectrum_torch(n_ticks, corner_freq_hz, spectral_slope,
                                    sampling_rate_hz, device)            # (n_freq,)
    real = torch.randn(n_groups, spec.numel(), generator=gen, device=device) * spec
    imag = torch.randn(n_groups, spec.numel(), generator=gen, device=device) * spec
    imag[:, 0] = 0.0                                  # DC real (zero imag, alias-free)
    if n_ticks % 2 == 0:
        imag[:, -1] = 0.0                            # Nyquist real (even n)
    base = torch.fft.irfft(torch.complex(real, imag), n=n_ticks, dim=1)  # (n_groups, n_ticks)

    z = base.new_zeros(1, n_ticks)
    left = torch.cat([z, base[:-1]], dim=0)
    right = torch.cat([base[1:], z], dim=0)
    wav = base - beta * (left + right)

    # normalize AFTER coupling by the measured realized RMS (matches the numpy
    # oracle's post-coupling renorm; the coupling inflates variance vs Parseval).
    realized = wav.pow(2).mean().sqrt()
    if float(realized) > 0:
        wav = wav * (rms_adc / realized)

    wire_to_group = torch.arange(n_channels, device=device) // group_size
    return wav[wire_to_group].to(torch.float32)


def add_intrinsic_noise(grids, geom, *, seeds, enc=DEFAULT_ENC,
                        coherent=True, incoherent=False,
                        sampling_rate_hz=DEFAULT_SAMPLING_RATE_HZ,
                        group_size=DEFAULT_GROUP_SIZE, coh_rms=DEFAULT_COH_RMS_ADC,
                        coh_corner_freq_hz=DEFAULT_COH_CORNER_FREQ_HZ,
                        coh_spectral_slope=DEFAULT_COH_SLOPE, beta=DEFAULT_COH_BETA,
                        series_spectrum=DEFAULT_SERIES_SPECTRUM,
                        coherent_numpy=False):
    """Add fresh forward noise to per-plane dense grids in place; returns ``grids``.

    ``grids`` : ``{plane_id: (B, W, T)}`` on-device. ``geom`` : registry with
    ``n_wires``/``n_ticks`` (+ ``wire_lengths`` per plane when ``incoherent``).
    ``seeds`` : ``(B,)`` per-event content-addressed seeds.

    Both components run **on-device** via per-event torch Generators (one per event,
    advanced across planes in canonical-id order). Statistical parity to the numpy
    forward model, not bit-exact — torch RNG differs from numpy and CUDA vs CPU
    streams differ, so realisations are device-specific. Set ``coherent_numpy=True``
    to recover the bit-exact / device-independent numpy coherent oracle (slower:
    per-group Python loop + H2D copy) for provenance / cross-checks.
    """
    if not grids:
        return grids
    dev = next(iter(grids.values())).device
    B = next(iter(grids.values())).shape[0]
    # sorted so the per-event cross-plane draw order is a deterministic function
    # of the canonical plane id (not dict insertion / HDF5 enumeration order).
    gids = sorted(grids.keys())

    if coherent and coherent_numpy:                  # bit-exact numpy oracle (opt-in)
        for b in range(B):
            rng = np.random.default_rng(int(seeds[b]))
            for gid in gids:
                _, W, T = grids[gid].shape
                coh = _coherent_noise_np(
                    W, T, rng, group_size=group_size, rms_adc=coh_rms,
                    corner_freq_hz=coh_corner_freq_hz,
                    spectral_slope=coh_spectral_slope, beta=beta,
                    sampling_rate_hz=sampling_rate_hz)
                grids[gid][b] += torch.as_tensor(coh, dtype=torch.float32, device=dev)
    elif coherent:                                   # on-device torch (default)
        for b in range(B):
            gen = torch.Generator(device=dev)
            gen.manual_seed((int(seeds[b]) ^ _COH_SEED_SALT) & ((1 << 63) - 1))
            for gid in gids:
                _, W, T = grids[gid].shape
                grids[gid][b] += _coherent_torch(
                    W, T, gen=gen, group_size=group_size, rms_adc=coh_rms,
                    corner_freq_hz=coh_corner_freq_hz, spectral_slope=coh_spectral_slope,
                    beta=beta, sampling_rate_hz=sampling_rate_hz, device=dev)

    if incoherent:
        for b in range(B):
            gen = torch.Generator(device=dev)
            gen.manual_seed((int(seeds[b]) ^ _INCOH_SEED_SALT) & ((1 << 63) - 1))
            for gid in gids:
                _, W, T = grids[gid].shape
                wl = geom[gid].get('wire_lengths')
                if wl is None:
                    raise ValueError(
                        f"incoherent noise needs 'wire_lengths' for plane {gid} "
                        "(configure wire_lengths_per_plane / the geometry registry)")
                grids[gid][b] += _incoherent_torch(
                    (W, T), wl, gen=gen, enc=enc, series_spectrum=series_spectrum,
                    sampling_rate_hz=sampling_rate_hz, device=dev)
    return grids


def digitize(grids, pedestal, n_bits=12, adc_max=None, gain=1.0):
    """Quantize per-plane grids to ADC codes — bit-exact to ``pimm_data.noise.digitize``.

    ``round(g*gain + ped).clip(0, adc_max) - ped``. ``pedestal`` is a scalar or a
    ``{plane_id: ped}`` dict. ``adc_max`` defaults to ``(1 << n_bits) - 1``.
    """
    amax = float(adc_max) if adc_max is not None else float((1 << int(n_bits)) - 1)
    out = {}
    for gid, g in grids.items():
        ped = float(pedestal[gid]) if isinstance(pedestal, dict) else float(pedestal)
        out[int(gid)] = torch.round(g * gain + ped).clamp(0.0, amax) - ped
    return out
