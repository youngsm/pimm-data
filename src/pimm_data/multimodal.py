"""MultiModalEventDataset — detector-agnostic event-selection layer.

Wraps one single-source dataset (``LUCiDDataset`` / ``JAXTPCDataset`` / any
nested-output dataset registered in :data:`DATASETS`) per **source**, and adds
the selection capabilities that are common to every detector (D6/D37):

* **source mixture** (D9) — combine several roots, each with its own integer
  ``label`` / ``config_id`` and a sampling ``weight``;
* **deterministic holdout** (D7/D26) — a 3-way train/val/test partition keyed
  on a stable ``(config_id, source_event_idx)`` hash (blake2b), so the split
  is reproducible across machines and invariant to shard add/remove/reorder
  (unlike a positional ``np.random.permutation``);
* **stable event identity** (``event_identity``/``split``) for the eval probe;
* per-source ``event_label`` / ``config_id`` attached to each sample.

This *composes* existing single-source datasets rather than re-implementing
their readers/builders — those already own the per-modality clouds, label
decoration, and (Phase A) the joint cross-modality index. The base only owns
"which events exist and which split they belong to".

Cross-modality alignment within a source is guaranteed by the wrapped
dataset's Phase-A joint index (see :mod:`pimm_data._joint_index`).
"""

import os
import struct
import hashlib
import logging
from copy import deepcopy

import numpy as np
from torch.utils.data import Dataset

from .builder import DATASETS, build_dataset
from .transform import Compose
from ._shard_meta import read_shard_meta

log = logging.getLogger(__name__)

_HOLDOUT_ROLES = ('train', 'val', 'test', 'all')


def _holdout_uniform(seed, identity):
    """Deterministic uniform draw in [0,1) for one event.

    ``identity = (config_id, file_index, source_event_idx)`` (see
    ``MultiModalEventDataset.event_identity``). blake2b → uint64 → [0,1).
    Folding ``config_id`` in makes the split config-stratified for free;
    folding ``file_index`` in makes the identity UNIQUE even when
    ``source_event_idx`` is shard-local (doraemon). Reproducible across
    machines/processes (unlike ``hash()`` or ``np.random``) and stable under
    shard reorder/add-remove (file_index/source_event_idx are intrinsic).
    """
    config_id, file_index, sei = identity
    payload = struct.pack('<qqqq', int(seed), int(config_id),
                          int(file_index), int(sei))
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    return struct.unpack('<Q', digest)[0] / float(1 << 64)


@DATASETS.register_module()
class MultiModalEventDataset(Dataset):
    """Mixture of single-source datasets with deterministic holdout.

    Parameters
    ----------
    source_dataset : dict
        Config template for one source's dataset, e.g.
        ``dict(type='LUCiDDataset', modalities=('sensor',), dataset_name='wc')``.
        ``data_root`` / ``split`` / ``transform`` are filled per source by this
        wrapper (any given here are overridden).
    sources : list
        One entry per source. A ``str`` is a root (or a ``name`` resolved under
        ``data_root``); a ``dict`` may set ``root`` (or ``name``), ``label``,
        ``config_id``, ``weight``, ``split``.
    data_root : str, optional
        Base directory; a source given by ``name`` resolves to
        ``data_root/name``.
    split : str
        Holdout role to expose: ``'train'`` / ``'val'`` / ``'test'`` / ``'all'``.
    holdout : dict, optional
        ``{seed, fractions=(tr,va,te)}`` (hash buckets) or
        ``{seed, n_per_config=k}`` (k held-out events per source, by smallest
        hash). ``None`` → everything is ``train`` (no holdout).
    max_events_per_source : int
        Cap per source after holdout (``-1`` = no cap).
    mixture : dict, optional
        ``{weights: 'natural'|'balanced'|list[float]}``. ``natural`` (default)
        keeps each source's event count; ``balanced`` equalizes via integer
        replication; a list gives explicit per-source replication weights.
    transform, loop, test_mode, test_cfg, ignore_index : standard.
    """

    def __init__(
        self,
        source_dataset,
        sources,
        *,
        data_root=None,
        split='train',
        holdout=None,
        max_events_per_source=-1,
        min_points=0,
        mixture=None,
        transform=None,
        loop=1,
        test_mode=False,
        test_cfg=None,
        ignore_index=-1,
    ):
        super().__init__()
        if not sources:
            raise ValueError("sources must contain at least one entry")
        if split not in _HOLDOUT_ROLES:
            raise ValueError(f"split must be one of {_HOLDOUT_ROLES}, got "
                             f"{split!r}")
        if test_mode:
            raise NotImplementedError(
                "MultiModalEventDataset is a train/val selection layer; "
                "test-time fragmented inference is not supported here.")

        self._source_dataset_cfg = dict(source_dataset)
        self._base_data_root = data_root
        self.split = split
        self._holdout = holdout
        self._max_events_per_source = int(max_events_per_source)
        self._min_points = int(min_points)
        self._mixture = mixture or {}
        self.transform = Compose(transform)
        self.loop = int(loop)
        self.ignore_index = ignore_index
        self.test_mode = False
        self.test_cfg = test_cfg

        self._specs = [self._normalize_source(s, i)
                       for i, s in enumerate(sources)]
        # `datasets` mirrors the eval-probe contract: a list of per-source
        # dicts carrying the sub-dataset + identity (see lucid_event_probe).
        self.datasets = self._build_sources()
        self.data_list = self._build_data_list()

        if len(self.data_list) == 0:
            raise ValueError(
                f"MultiModalEventDataset yielded 0 events for split={split!r} "
                f"(holdout={holdout}, max_events_per_source="
                f"{max_events_per_source}).")
        log.info("MultiModalEventDataset: %d events x%d loop, split=%s, "
                 "sources=%s", len(self.data_list), self.loop, self.split,
                 [s['name'] for s in self._specs])

    # ------------------------------------------------------------------
    # Source resolution / construction
    # ------------------------------------------------------------------

    def _normalize_source(self, source, default_idx):
        if isinstance(source, str):
            source = {'name': source}
        elif not isinstance(source, dict):
            raise TypeError(f"each source must be str or dict, got "
                            f"{type(source).__name__}")
        root = source.get('root', source.get('data_root'))
        name = source.get('name')
        if root is None:
            if name is None:
                raise ValueError("source needs 'root' or 'name'")
            if self._base_data_root is None:
                raise ValueError(f"source by name {name!r} needs a base "
                                 "data_root on the dataset")
            root = os.path.join(self._base_data_root, name)
        if name is None:
            name = os.path.basename(os.path.normpath(root))
        return {
            'name': str(name),
            'root': root,
            'label': int(source.get('label', default_idx)),
            'config_id': int(source.get('config_id', default_idx)),
            'weight': float(source.get('weight', 1.0)),
            'split': source.get('split', ''),
        }

    def _build_sources(self):
        out = []
        for spec in self._specs:
            cfg = dict(self._source_dataset_cfg)
            cfg['data_root'] = spec['root']
            cfg['split'] = spec['split']
            cfg['transform'] = None        # the wrapper owns the transform
            cfg['loop'] = 1
            out.append({'dataset': build_dataset(cfg), **spec})
        return out

    # ------------------------------------------------------------------
    # Identity + holdout
    # ------------------------------------------------------------------

    def _event_loc(self, sub, local_idx):
        """(file_idx, event_num) for a sub-dataset's local index."""
        return sub._canonical_reader.locate(local_idx)

    def _event_key(self, sub, file_idx, event_num):
        """``(file_index, source_event_idx)`` — the within-config identity key.

        ``file_index`` is the intrinsic shard id from the file's config attr
        (falls back to the source-relative file position). ``source_event_idx``
        resolves from the config vector (WAND), else
        ``global_event_offset + event_num`` (doraemon — shard-local, hence the
        file_index is what makes the full identity unique), else the event
        number. All inputs intrinsic → identity stable under shard
        reorder/add-remove (F1)."""
        reader = sub._canonical_reader
        try:
            meta = read_shard_meta(reader.h5_files[file_idx])
        except Exception:
            meta = {}
        file_index = meta.get('file_index')
        if file_index is None:
            file_index = file_idx
        sei_vec = meta.get('source_event_idx')
        offset = meta.get('global_event_offset')
        if sei_vec is not None and 0 <= event_num < len(sei_vec):
            sei = int(sei_vec[event_num])
        elif offset is not None:
            sei = int(offset) + int(event_num)
        else:
            sei = int(event_num)
        return int(file_index), sei

    def _build_data_list(self):
        """Per source: identity → holdout role → keep matching split; then
        cap, then mixture-replicate. Produces (source_idx, local_idx) rows."""
        h = self._holdout or {}
        n_per_config = h.get('n_per_config')
        seed = int(h.get('seed', 0))

        kept_per_source = []
        for source_idx, source in enumerate(self.datasets):
            sub = source['dataset']
            cid = source['config_id']
            rows = []          # (local_idx, u)
            for local_idx in range(len(sub)):
                file_idx, event_num = self._event_loc(sub, local_idx)
                file_index, sei = self._event_key(sub, file_idx, event_num)
                u = _holdout_uniform(seed, (cid, file_index, sei))
                rows.append((local_idx, u))

            if n_per_config is not None:    # implies self._holdout is truthy
                # Smallest-u events per source are the holdout pool; the rest
                # are train. Deterministic via the same hash. NOTE: in this mode
                # 'val' and 'test' return the SAME pool (no val/test split) —
                # unlike the fractions mode below, which partitions three
                # disjoint buckets.
                order = sorted(range(len(rows)), key=lambda i: rows[i][1])
                holdout_set = set(order[:int(n_per_config)])
                if self.split == 'train':
                    sel = [i for i in range(len(rows)) if i not in holdout_set]
                elif self.split == 'all':
                    sel = list(range(len(rows)))
                else:  # val/test/holdout → the held-out events
                    sel = sorted(holdout_set)
            elif self.split == 'all':
                sel = list(range(len(rows)))
            else:
                tr, va, _ = h.get('fractions', (0.9, 0.05, 0.05)) \
                    if self._holdout else (1.0, 0.0, 0.0)
                sel = []
                for i, (_, u) in enumerate(rows):
                    role = ('train' if u < tr
                            else 'val' if u < tr + va else 'test')
                    if role == self.split:
                        sel.append(i)

            if self._min_points > 0:
                sub = source['dataset']
                sel = [i for i in sel
                       if self._passes_min_points(sub, rows[i][0])]
            if self._max_events_per_source > 0:
                sel = sel[:self._max_events_per_source]
            kept_per_source.append([rows[i][0] for i in sel])

        if self._min_points > 0:
            # The min_points scan opened sensor handles in the PARENT; release
            # them so DataLoader workers reopen fresh after the fork (the
            # reader pid-guard would reopen anyway, but the parent must not hold
            # live HDF5 fds across the fork).
            for source in self.datasets:
                r = getattr(source['dataset'], 'sensor_reader', None)
                if r is not None:
                    r.close()

        return self._mix(kept_per_source)

    def _passes_min_points(self, sub, local_idx):
        """Per-event point-count filter (LUCiD SSL min_points, strict ``>``).

        Counts the source's unique fired PMTs (matching the aggregated point
        count the SSL pipeline consumes). Reads each event, so it is the
        expensive index path — only taken when ``min_points>0``. A source with
        no ``sensor_reader`` is not filtered."""
        reader = getattr(sub, 'sensor_reader', None)
        if reader is None:
            return True
        sid = reader.read_event(local_idx).get('sensor_idx')
        if sid is None:
            return True
        return int(np.unique(sid).size) > self._min_points

    def _mix(self, kept_per_source):
        """Apply per-source mixture weights via integer replication."""
        mode = self._mixture.get('weights', 'natural')
        counts = [len(k) for k in kept_per_source]
        if mode == 'natural':
            reps = [1] * len(kept_per_source)
        elif mode == 'balanced':
            mx = max(counts) if counts else 0
            reps = [max(1, round(mx / c)) if c else 0 for c in counts]
        elif isinstance(mode, (list, tuple)):
            reps = [max(0, int(round(w))) for w in mode]
            if len(reps) != len(kept_per_source):
                raise ValueError("mixture weights length != number of sources")
        else:
            raise ValueError(f"unknown mixture weights {mode!r}")

        data_list = []
        for source_idx, locals_ in enumerate(kept_per_source):
            for _ in range(reps[source_idx]):
                data_list.extend((source_idx, li) for li in locals_)
        return data_list

    # ------------------------------------------------------------------
    # Public API consumed by the eval probe / trainer
    # ------------------------------------------------------------------

    def event_identity(self, idx):
        """Stable, unique ``(config_id, file_index, source_event_idx)`` for a
        dataset index — invariant to shard reorder/add-remove (F1)."""
        source_idx, local_idx = self.data_list[idx % len(self.data_list)]
        sub = self.datasets[source_idx]['dataset']
        file_idx, event_num = self._event_loc(sub, local_idx)
        file_index, sei = self._event_key(sub, file_idx, event_num)
        return (self.datasets[source_idx]['config_id'], file_index, sei)

    def get_data_name(self, idx):
        source_idx, local_idx = self.data_list[idx % len(self.data_list)]
        source = self.datasets[source_idx]
        return f"{source['name']}/{source['dataset'].get_data_name(local_idx)}"

    def get_data(self, idx):
        source_idx, local_idx = self.data_list[idx % len(self.data_list)]
        source = self.datasets[source_idx]
        sample = source['dataset'].get_data(local_idx)
        sample['name'] = self.get_data_name(idx)
        sample['split'] = self.split
        # Per-event labels: top-level length-1 (probe per-event path) AND
        # per-point broadcast into each point-bearing modality so a downstream
        # Collect(part=...) carries them (D28 event_broadcast).
        label = source['label']
        config_id = source['config_id']
        sample['event_label'] = np.array([label], dtype=np.int64)
        sample['config_id'] = np.array([config_id], dtype=np.int64)
        for v in sample.values():
            if isinstance(v, dict) and 'coord' in v:
                n = v['coord'].shape[0]
                v['event_label'] = np.full((n, 1), label, dtype=np.int64)
                v['config_id'] = np.full((n, 1), config_id, dtype=np.int64)
        return sample

    def prepare_train_data(self, idx):
        return self.transform(self.get_data(idx))

    def __getitem__(self, idx):
        return self.prepare_train_data(idx)

    def __len__(self):
        return len(self.data_list) * self.loop

    def __del__(self):
        for source in getattr(self, 'datasets', []):
            sub = source.get('dataset')
            if sub is not None:
                try:
                    sub.__del__()
                except Exception:
                    pass
