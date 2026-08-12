"""Shared base for the multimodal shard datasets (consolidation).

:class:`~pimm_data.jaxtpc.JAXTPCDataset` and
:class:`~pimm_data.lucid.LUCiDDataset` both compose per-modality shard readers
(step / sensor / hits / labl), build one joint cross-modality event index
(Phase A / D42), and expose the same minimal ``torch.utils.data.Dataset``
surface (``__getitem__`` → transform pipeline, ``__len__``). That wiring lives
here once; each subclass supplies only its reader construction (``__init__``)
and its per-modality ``_build_*`` cloud builders / label model.

This base is self-contained (it inherits ``torch.utils.data.Dataset``
directly): the detector datasets never used a generic npy loader, cache, or
test-time augmentation path.

A subclass must, in ``__init__`` (before ``super().__init__``): set
``_modalities`` / ``_max_len`` / ``_strict_lengths`` / ``_source_data_root``,
create the ``{step,sensor,hits,labl}_reader`` attrs (None when absent), set
``_canonical_reader`` (the reader whose file names label events), and call
``self._build_joint_index(source_label, filter_label)``. It must also define
``get_data(idx)`` returning the nested per-modality sample dict.
"""

import glob
import os
import logging

from torch.utils.data import Dataset

from .transform import Compose
from ._joint_index import build_joint_index

log = logging.getLogger(__name__)


class ShardEventDataset(Dataset):
    """Reader composition + joint index + a minimal Dataset surface."""

    #: Modalities this dataset family understands. Also the reader-iteration
    #: order for the joint index (subclasses may pick a different
    #: ``_canonical_reader`` precedence; close order is irrelevant).
    VALID_MODALITIES = ('step', 'sensor', 'hits', 'labl')

    def __init__(self, *, split, data_root, transform=None,
                 ignore_index=-1, loop=1):
        super().__init__()
        self.data_root = data_root
        self.split = split
        # map -> reduce -> map: one transform list, split at the terminal Collect.
        # The head (... Collect) runs per-event in workers (self.transform). The
        # tail (post-Collect: ToDevice/Densify/AddNoise/Digitize) runs per-BATCH
        # after collate (self.batch_transform) — a trainer calls it post-collate
        # (e.g. on_after_batch_transfer). No separate cpu/gpu transform argument;
        # ToDevice in the tail is the device step. None when there's no tail.
        head, tail = self._split_at_collect(transform)
        self.transform = Compose(head)
        self.batch_transform = Compose(tail) if tail else None
        self.ignore_index = ignore_index
        self.loop = loop
        # Detector datasets have no test-time-augmentation path; kept as a
        # plain attr only so any generic consumer that probes it sees False.
        self.test_mode = False
        self.data_list = self.get_data_list()
        log.info(
            "Totally %d x %d samples in %s %s set.",
            len(self.data_list), self.loop,
            os.path.basename(self.data_root), split)

    @staticmethod
    def _split_at_collect(transform):
        """Split the pipeline at the terminal ``Collect`` (the reduce boundary):
        head = ``[…, Collect]`` (per-event), tail = ``[post-Collect]`` (per-batch).

        Accepts ``Collect`` as a ``dict(type='Collect', …)`` or an instance. No
        ``Collect`` (or none given) → everything is the per-event head.
        """
        if not transform:
            return transform, None
        from .transform import Collect
        specs = list(transform)

        def _is_collect(t):
            if isinstance(t, dict):
                return t.get('type') == 'Collect'
            return isinstance(t, Collect)

        idxs = [i for i, t in enumerate(specs) if _is_collect(t)]
        if not idxs:
            return specs, None
        cut = idxs[-1] + 1               # terminal Collect ends the per-event head
        return specs[:cut], specs[cut:]

    # -- Dataset surface ---------------------------------------------------

    def prepare_train_data(self, idx):
        return self.transform(self.get_data(idx))

    def __getitem__(self, idx):
        return self.prepare_train_data(idx)

    def __len__(self):
        return len(self.data_list) * self.loop

    # -- modality wiring ----------------------------------------------------

    def _validate_modalities(self, modalities):
        mods = set(modalities)
        if not mods:
            raise ValueError("modalities is empty; must load at least one")
        unknown = mods - set(self.VALID_MODALITIES)
        if unknown:
            raise ValueError(
                f"Unknown modalities {unknown}; valid: {self.VALID_MODALITIES}")
        if mods == {'labl'}:
            raise ValueError(
                "Invalid modality combination ('labl',): labl is a dimension "
                "table and requires an instance-bearing modality ('step' or "
                "'hits') to join against.")
        if mods == {'sensor', 'labl'}:
            raise ValueError(
                "Invalid modality combination ('sensor', 'labl'): sensor has "
                "no instance/particle separation, so labl can't be attached. "
                "Add 'hits' or 'step' to the modalities tuple.")

    def _modality_root(self, modality):
        mod_dir = os.path.join(self._source_data_root, modality)
        return mod_dir if os.path.isdir(mod_dir) else self._source_data_root

    def _expand_runs(self, runs, modalities):
        """Resolve a ``runs=`` argument to an explicit list of run names.

        A string is a glob pattern (``'run_*'`` / ``'*'``) expanded ONCE
        (sorted) against the first loaded modality root that yields run
        directories, then the same explicit list is handed to every reader —
        per-modality expansion could silently diverge and positionally
        desync the joint index. A list/tuple is taken verbatim
        (caller-chosen order)."""
        if isinstance(runs, str):
            tried = []
            for m in modalities:
                root = self._modality_root(m)
                expanded = sorted(
                    os.path.basename(p.rstrip(os.sep))
                    for p in glob.glob(os.path.join(root, runs))
                    if os.path.isdir(p))
                if expanded:
                    return expanded
                tried.append(root)
            raise FileNotFoundError(
                f"runs={runs!r} matched no run directories under any "
                f"modality root: {tried}")
        runs = list(runs)
        if not runs:
            raise ValueError("runs= must name at least one run")
        return runs

    def _readers_named(self):
        """``(modality, reader)`` for each loaded modality, in
        ``VALID_MODALITIES`` order (the order the joint index expects)."""
        return [(m, getattr(self, f'{m}_reader'))
                for m in self.VALID_MODALITIES
                if getattr(self, f'{m}_reader', None) is not None]

    def _build_joint_index(self, source_label, filter_label=''):
        """Build one joint cross-modality event index, injected into every
        reader so a global idx maps to the SAME physics event in all
        modalities (Phase A / D42 — see :mod:`pimm_data._joint_index`)."""
        self._n_events = build_joint_index(
            self._readers_named(), strict_lengths=self._strict_lengths,
            source_label=source_label, filter_label=filter_label)

    def get_data_list(self):
        """Event ids for this split.

        The VALUES are event ids, not positions — subclasses must dereference
        ``self.data_list[i]`` rather than passing ``i`` to a reader. That was
        only ever true by coincidence (``list(range(n))`` makes index == value),
        and the first non-identity list silently read the wrong events.

        ``event_range=(lo, hi)`` selects ``[lo, hi)``; ``exclude_range=(lo, hi)``
        removes that band; ``max_len`` truncates whatever is left. Lives here
        rather than on one dataset so every reader-backed dataset can express a
        split, and so the index contract is stated in one place.

        For a split that is stable under shard reordering or corpus growth, see
        :class:`~pimm_data.multimodal.MultiModalEventDataset`, which partitions
        on a hash of the event identity instead of a positional slice.
        """
        n = getattr(self, '_n_events', 0)
        rng = getattr(self, '_event_range', None)
        ids = list(range(n)) if rng is None else \
            list(range(max(0, int(rng[0])), min(n, int(rng[1]))))
        exc = getattr(self, '_exclude_range', None)
        if exc is not None:
            lo, hi = int(exc[0]), int(exc[1])
            ids = [i for i in ids if not (lo <= i < hi)]
        max_len = getattr(self, '_max_len', -1)
        if max_len > 0:
            ids = ids[:max_len]
        if not ids and (rng is not None or exc is not None):
            raise ValueError(
                f"event_range={rng} with exclude_range={exc} selects NO events "
                f"out of {n} — an empty split fails later and far from here.")
        return ids

    def get_data_name(self, idx):
        reader = self._canonical_reader
        file_idx, event_num = reader.locate(idx)
        fname = os.path.basename(reader.h5_files[file_idx])
        # Multi-run (B2 runs=): qualify with the run so names — and
        # everything keyed on them (content-addressed noise seeds, holdout
        # hashing, cache identity) — stay unique across runs. Single-split
        # legacy names are byte-identical to before (run_of() -> '').
        run = reader.run_of(file_idx) if hasattr(reader, 'run_of') else ''
        prefix = f"{run}/" if run else ""
        return f"{prefix}{fname}_evt{event_num:03d}"

    def __del__(self):
        for m in self.VALID_MODALITIES:
            reader = getattr(self, f'{m}_reader', None)
            if reader is not None:
                try:
                    reader.close()
                except Exception:
                    pass


#: Back-compat alias for the pre-consolidation name (internal; only
#: jaxtpc.py / lucid.py imported it).
_MultiModalShardDataset = ShardEventDataset
