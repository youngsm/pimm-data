"""Deterministic dataset splitting keyed on event IDENTITY, not position.

Lifted out of :mod:`pimm_data.multimodal`, which has used this since D7/D26, so
that every reader-backed dataset can split the same way instead of each one
inventing its own.

**Why identity and not a positional slice.** A slice of the joint index is a
split exactly once. The index is built over a *glob* of shards, so adding a
shard, removing one, or re-globbing in a different order renumbers every event
after the change — and a probe holdout defined as ``(17700, 18000)`` silently
becomes a set of trained-on events, with nothing to report it. The coefficient
corpus is explicitly built to grow (one run now, eight planned), so this is not
a hypothetical.

Hashing an intrinsic identity instead gives a split that is stable under shard
add / remove / reorder, reproducible across machines and processes (unlike
``hash()``, which is salted per interpreter, or ``np.random.permutation``, which
depends on call order), and — for the coeff corpus, whose identity is the
simulation event — stable even across a corpus REBUILD with a different basis or
noise model.

The counterpart to this module is a written manifest: hashing makes the split
reproducible, but a resolved list of identities next to the data is what makes it
auditable and citable. See ``pimm_data.coeff.CoeffTPCDataset.holdout_manifest``.
"""

import hashlib
import struct

__all__ = ["holdout_uniform", "assign_split", "normalise_fractions"]


def holdout_uniform(seed, identity):
    """Deterministic uniform draw in [0, 1) for one event.

    ``identity`` is a tuple of ints — for the generic case
    ``(config_id, file_index, source_event_idx)``; folding ``file_index`` in
    keeps it unique when ``source_event_idx`` is shard-local. blake2b → uint64 →
    [0, 1).

    Byte-compatible with the original in ``multimodal.py``: same ``'<qqqq'``
    packing and same digest size, so existing LUCiD splits do not move.
    """
    payload = struct.pack("<q" + "q" * len(identity), int(seed),
                          *(int(x) for x in identity))
    return struct.unpack("<Q", hashlib.blake2b(payload, digest_size=8).digest())[0] \
        / float(1 << 64)


def normalise_fractions(fractions):
    """Validate a ``{name: fraction}`` mapping and return cumulative bounds.

    Returns ``[(name, lo, hi), ...]`` covering [0, 1) with no gap, so every event
    lands in exactly one split — an event belonging to nothing is a silent way to
    lose data, which a positional scheme did by leaving 999 events unassigned.
    """
    if not fractions:
        raise ValueError("holdout fractions is empty")
    bad = {k: v for k, v in fractions.items() if not (0.0 <= float(v) <= 1.0)}
    if bad:
        raise ValueError(f"holdout fractions must be in [0, 1]: {bad}")
    total = sum(float(v) for v in fractions.values())
    if abs(total - 1.0) > 1e-9:
        raise ValueError(
            f"holdout fractions must sum to 1.0, got {total:.6f} from "
            f"{fractions} — an event in no split is data silently dropped")
    out, lo = [], 0.0
    for name in sorted(fractions):                 # sorted: order-independent
        hi = lo + float(fractions[name])
        out.append((name, lo, hi))
        lo = hi
    out[-1] = (out[-1][0], out[-1][1], 1.0 + 1e-12)   # close the top edge
    return out


def assign_split(seed, identity, bounds):
    """Which split ``identity`` belongs to, given ``normalise_fractions`` output."""
    u = holdout_uniform(seed, identity)
    for name, lo, hi in bounds:
        if lo <= u < hi:
            return name
    return bounds[-1][0]                            # unreachable; float guard
