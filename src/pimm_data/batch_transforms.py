"""
Post-collate, on-device dense transforms (REDESIGN).

There is **no batch-transform concept and no runner**: the dense ops are ordinary
``scope='sample'`` transforms placed *after* a ``ToDevice`` step and run by the plain
:class:`pimm_data.transform.Compose` (or any loop). The dense grids are born
on-device; pass ``ToDevice('cpu')`` to run the same transforms on CPU.

``ToDevice`` is the device step. The dense ops — ``Densify`` / ``AddNoise`` /
``Digitize`` (in :mod:`pimm_data.detector_transforms`, ``scope='sample'``) — are
ONE class each that DISPATCH on input: a per-event sensor sub-dict (numpy,
pre-collate) OR the flat collated batch (``<m>_wire/.../offset`` -> ``<m>_dense=
{gid:(B,W,T)}``, torch, post-collate; born on the inputs' device, GPU after
``ToDevice``). There is no separate ``Batch*`` version and no runner — noise
self-seeds from ``batch['name']`` (+ optional ``batch['_epoch']``/``_rank``).

Each is ``fn(batch) -> batch`` (flat-prefixed keys; ``modality=`` sets the prefix,
``None`` = bare). :func:`sensor_dense_cfg` returns the standard chain as config;
:func:`build_sensor_gpu_stages` wraps it in a ``Compose`` you call on the batch.

Seeds are content-addressed per event (``blake2b(name|base_seed|epoch|rank)``), so
the same event gets the same noise across batch position / worker / resume. The
**incoherent** component uses a torch ``Generator`` whose CPU/CUDA streams differ, so
incoherent realizations are device-specific (coherent uses numpy, device-independent).
Geometry is a registry baked into the transform, not carried per-sample.
"""

from __future__ import annotations

import hashlib
import warnings

import torch

from . import dense_ops
from .transform import TRANSFORMS


# ── device move (idempotent) ──────────────────────────────────────────────
def move_to_device(batch, device, non_blocking=True):
    """Recursively move tensors to ``device``; non-tensors untouched.

    Idempotent — a tensor already on ``device`` is returned by ``.to`` without a
    copy. Use ``DataLoader(pin_memory=True)`` for ``non_blocking`` to be truly async.
    """
    if isinstance(batch, torch.Tensor):
        return batch.to(device, non_blocking=non_blocking)
    if isinstance(batch, dict):
        return {k: move_to_device(v, device, non_blocking) for k, v in batch.items()}
    if isinstance(batch, (list, tuple)):
        return type(batch)(move_to_device(v, device, non_blocking) for v in batch)
    return batch


# ── content-addressed seeding ─────────────────────────────────────────────
def content_seed(event_name, base_seed=0, epoch=0, rank=0):
    """Stable per-event seed: same event -> same noise across batch/worker/resume."""
    payload = f"{event_name}|{int(base_seed)}|{int(epoch)}|{int(rank)}".encode()
    h = hashlib.blake2b(payload, digest_size=8).digest()
    return int.from_bytes(h, 'little') & ((1 << 63) - 1)


def _seeds_for(batch, base_seed):
    """Per-event content-addressed seeds from ``batch['name']`` (self-contained — no
    runner injects them). Epoch/rank are read from optional ``batch['_epoch']`` /
    ``batch['_rank']`` the trainer may stamp (default 0 = stable within a run)."""
    names = batch.get('name')
    if isinstance(names, str):
        names = [names]
    epoch = int(batch.get('_epoch', 0)) if not torch.is_tensor(batch.get('_epoch')) else int(batch['_epoch'])
    rank = int(batch.get('_rank', 0)) if not torch.is_tensor(batch.get('_rank')) else int(batch['_rank'])
    if not names:
        warnings.warn(
            "AddNoise: batch has no 'name' -> seeding falls back to batch position, "
            "NOT reproducible across shuffles/epochs/runs. Ensure Collect passes "
            "'name' through.", RuntimeWarning, stacklevel=2)
        # length B from the dense grid is unknown here; caller passes count via name
        return None
    return [content_seed(nm, base_seed, epoch, rank) for nm in names]


# ── batch transforms (post-collate; registered, scope='batch') ───────────────
@TRANSFORMS.register_module()
class ToDevice:
    """Move the whole batch to a device — the device move expressed as a transform.

    So device selection composes uniformly with the other batch transforms instead
    of being a special ``apply_batch_transforms(device=)`` argument: put
    ``dict(type='ToDevice', device='cuda')`` first in the list and everything after
    runs on-device. Idempotent (a tensor already on ``device`` is returned without a
    copy). ``scope='batch'`` so the pre-collate ``Compose`` fence keeps it post-collate.
    """

    scope = 'sample'

    def __init__(self, device, *, non_blocking=True):
        self.device = device
        self.non_blocking = non_blocking

    def __call__(self, batch):
        return move_to_device(batch, self.device, non_blocking=self.non_blocking)


class BatchTransformMixin:
    """Mixin for a LightningModule: run the post-collate transforms in
    ``on_after_batch_transfer``.

    Set ``self.batch_stages`` to a ``Compose`` (e.g. from
    :func:`build_sensor_gpu_stages`). Lightning calls this with the batch on-device;
    we stamp ``_epoch``/``_rank`` so the self-seeding noise is per-epoch/rank, then
    run the transforms. (For a custom loop, just call your ``Compose`` on the batch.)
    """

    batch_stages = None

    def on_after_batch_transfer(self, batch, dataloader_idx=0):
        if not self.batch_stages:
            return batch
        trainer = getattr(self, 'trainer', None)
        batch['_epoch'] = getattr(trainer, 'current_epoch', 0) or 0
        batch['_rank'] = getattr(trainer, 'global_rank', 0) or 0
        return self.batch_stages(batch)
