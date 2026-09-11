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

# The forward-model imports are gone with it: this module is now the generic
# scatter only -- going to dense is a data-layer operation, the physics is not.

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
