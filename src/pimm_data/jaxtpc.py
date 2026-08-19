"""
JAXTPCDataset — multimodal dataset for LArTPC detector simulation output.

Loads from co-indexed HDF5 files produced by JAXTPC's production pipeline:

* ``step/`` — 3D truth deposits
* ``sensor/`` — raw sparse wire / pixel readout
* ``hits/`` — per-instance sensor decomposition
* ``labl/`` — track_id → label lookup tables

Modality strings: ``'step'``, ``'sensor'``, ``'hits'``, ``'labl'``. See
README §Modality combinations for the combination matrix.

Returns a **nested** dict: each loaded modality owns a sub-dict with clean
unprefixed keys::

    {
      'step':   {'coord': (N,3), 'energy': (N,1), 'volume_id': ..., ...},
      'sensor': {'coord': (M,D), 'energy': (M,1), 'plane_id': ...,
                 'readout_type': 'wire'|'pixel',
                 'raw': {plane_label: {'wire'|'py'+'pz', 'time', 'value'}}},
      'hits':   {'coord': (E,D), 'energy': (E,1), 'instance': ..., ...,
                 'readout_type': 'wire'|'pixel',
                 'raw': {plane_label: {'wire'|'py'+'pz', 'time',
                                       'group_id', 'charge'}}},
      'labl':   {'v0': {'track_ids': (T,), 'track_pdg': (T,),
                        'deposit_to_track': (N_v,), ...}, 'v1': {...}},
      'bridges':{'group_to_track_v0': (G,), 'deposit_to_group_v0': (N_v,),
                 'qs_fractions_v0': ..., ...},
      'name': str, 'split': str,
    }

Missing modalities have no top-level key. There is no bare ``coord`` / no
precedence / no prefixed aliases — transforms pick a modality explicitly (see
``Apply(on=...)`` and ``Collect(part=...)``).

Registered in :data:`pimm_data.DATASETS` for config-driven construction
via ``dict(type="JAXTPCDataset", ...)``.
"""

import re
import logging
import warnings

import numpy as np

from .builder import DATASETS
from ._dataset_base import ShardEventDataset
from ._label_decorate import gather_with_fill
from .readers.jaxtpc_step import JAXTPCStepReader
from .readers.jaxtpc_sensor import JAXTPCSensorReader
from .readers.jaxtpc_labl import JAXTPCLablReader
from .readers.jaxtpc_hits import JAXTPCHitsReader

log = logging.getLogger(__name__)

# Wire-readout plane types, in canonical order. A plane label is
# ``volume_{v}_{T}`` (T in _PLANE_TYPES) or ``volume_{v}_Pixel``.
_PLANE_TYPES = ('U', 'V', 'Y')


def canonical_plane_id(label):
    """Stable, file-independent global id for a plane label.

    ``volume_{v}_{U|V|Y}`` -> ``v * 3 + {U:0,V:1,Y:2}``; ``volume_{v}_Pixel`` -> ``v``.
    Unlike the per-event positional ``plane_id`` (a dense index into the planes
    present *in that event*), this never shifts when a plane is empty or under
    ``volume=`` filtering, so a downstream densify can key per-plane grids/geometry
    by it consistently across a batch.
    """
    parts = label.split('_')
    if len(parts) != 3 or parts[0] != 'volume':
        raise ValueError(
            f"canonical_plane_id: expected a 'volume_{{v}}_{{U|V|Y|Pixel}}' label, "
            f"got {label!r}")
    try:
        v = int(parts[1])
    except ValueError:
        raise ValueError(f"canonical_plane_id: non-integer volume in {label!r}")
    t = parts[2]
    if t == 'Pixel':
        # NOTE: pixel ids share the integer space with wire ids by volume; the
        # dense path is wire-only (plane_geometry() returns {} for pixel), so a
        # mixed wire+pixel registry is unsupported.
        return v
    if t not in _PLANE_TYPES:
        raise ValueError(
            f"canonical_plane_id: unknown plane type {t!r} in {label!r} "
            f"(expected one of {_PLANE_TYPES} or 'Pixel')")
    return v * len(_PLANE_TYPES) + _PLANE_TYPES.index(t)


@DATASETS.register_module()
class JAXTPCDataset(ShardEventDataset):
    """LArTPC multimodal dataset with nested per-modality output.

    Parameters
    ----------
    data_root : str
        Root directory with ``step/``, ``sensor/``, ``hits/``, ``labl/``
        subdirectories.
    split : str
        Split name for file discovery (one run directory).
    runs : list[str] | str, optional
        Multi-run selection in one dataset (B2): a list of run names, or a
        glob pattern (``'run_*'``) expanded once (sorted) against the first
        modality root with matches. Mutually exclusive with ``split=``.
        Event names become ``'<run>/<file>_evtNNN'`` — unique across runs —
        so content-addressed noise seeds, holdout hashing and cache identity
        never collide (single-run ``split=`` names are unchanged).
    modalities : tuple[str]
        Any subset of ``'step'``, ``'sensor'``, ``'hits'``, ``'labl'``.
        ``('labl',)`` and ``('sensor', 'labl')`` are invalid (see
        README §Modality combinations).
    dataset_name : str
        File prefix (e.g., ``'sim'`` for ``sim_step_0000.h5``).
    volume : int or None
        Load only this volume index. ``None`` = all volumes.
    labels : str | dict | None
        Opt-in label-source decoration. ``labl`` is a label *source* (a
        dimension table FK-joined onto ``step``/``hits``), **not** a modality.
        A column name (e.g. ``'pdg'``) selects the segment column; a dict
        carries ``key``/``keys``/``config``. ``None`` = no labels. (Legacy:
        ``'labl'`` in ``modalities=`` also requests labels, deprecated.)
    label_key : str
        Which labl column to decorate the point clouds with. Must match a
        column in the labl files (``'pdg'``, ``'cluster'``, ``'interaction'``,
        ``'ancestor'``). Raw values from ``track_{label_key}`` are broadcast
        to each deposit / pixel entry; use a downstream ``RemapSegment`` to
        map raw values to task-specific class indices.
    min_deposits : int
        Minimum 3D deposits per event (step reader filter).
    include_physics : bool
        Whether step reader loads dx, theta, phi, charge, photons, etc.
    label_keys : list or None
        Which label datasets to load from labl files (None → all).
    transform : list or None
        Transform pipeline.
    loop, max_len, ignore_index : standard dataset parameters.
    """


    def __init__(
        self,
        data_root,
        split='train',
        runs=None,
        modalities=('step',),
        dataset_name='sim',
        volume=None,
        labels=None,
        label_key='pdg',
        label_config=None,
        min_deposits=0,
        include_physics=True,
        label_keys=None,
        transform=None,
        loop=1,
        max_len=-1,
        ignore_index=-1,
        strict_lengths=False,
        cache=False,
        num_time_steps=None,
        n_wires_per_plane=None,
        pedestal_per_plane=None,
        wire_lengths_per_plane=None,
    ):
        self._modalities = tuple(modalities)
        self._validate_modalities(self._modalities)
        # Per-plane wire lengths (METERS) for incoherent-noise ENC, keyed by
        # plane TYPE ('U'/'V'/'Y') or full label, value an (n_wires,) array or a
        # (lo, hi) pair expanded via linspace. Detector geometry — not in the
        # file — so it is config, surfaced through plane_geometry().
        self._wire_lengths_per_plane = dict(wire_lengths_per_plane or {})

        # A3: min_deposits filters on step deposit counts, so it is a no-op
        # (and silently so) without the step modality. Fail loud instead.
        if min_deposits > 0 and 'step' not in self._modalities:
            raise ValueError(
                f"min_deposits={min_deposits} filters on step deposit counts "
                f"but modalities={self._modalities} does not include 'step'. "
                "Add 'step' to modalities or set min_deposits=0.")

        # ``labels=`` is the opt-in for label-source decoration: ``labl`` is a
        # label *source* (a dimension table FK-joined onto a point cloud), NOT a
        # modality. A string sets the label column (== legacy ``label_key``); a
        # dict carries ``key``/``keys``/``config``. Back-compat: ``'labl'`` in
        # ``modalities=`` still requests labels (deprecated — use ``labels=``).
        if isinstance(labels, str):
            label_key = labels
        elif isinstance(labels, dict):
            label_key = labels.get('key', label_key)
            label_keys = labels.get('keys', label_keys)
            label_config = labels.get('config', label_config)
        self._want_labels = (labels is not None) or ('labl' in self._modalities)
        if self._want_labels and not ({'step', 'hits'} & set(self._modalities)):
            raise ValueError(
                "labels were requested (labels=/'labl') but modalities has no "
                "decoratable point cloud — need 'step' or 'hits'; "
                f"modalities={self._modalities}.")
        # Make the silent labels-skip-<modality> coupling loud: only 'step'/'hits'
        # are FK-decoratable; any OTHER requested modality (e.g. 'sensor' — raw
        # readout, no instance separation) passes through with no segment/instance.
        if self._want_labels:
            _undecorated = [m for m in self._modalities
                            if m not in ('step', 'hits', 'labl')]
            if _undecorated:
                warnings.warn(
                    f"labels= was requested but modality/-ies {_undecorated} are "
                    "not label-decoratable (only 'step'/'hits' receive "
                    "segment/instance) — they pass through raw, with no labels. "
                    "Expected for 'sensor' (raw readout); flagged so you don't "
                    "rely on supervision that isn't there.", stacklevel=2)

        self._dataset_name = dataset_name
        self._volume = volume
        self._label_key = label_key
        self._label_config = label_config
        self._validate_label_config()
        self._min_deposits = min_deposits
        self._include_physics = include_physics
        self._label_keys = label_keys
        self._max_len = max_len
        self._strict_lengths = strict_lengths
        self._source_data_root = data_root
        # B2 runs=: multi-run selection in ONE dataset. A list/tuple names
        # runs verbatim; a string is a glob expanded once (sorted) against
        # the first modality's root. Readers receive the explicit list, and
        # get_data_name() qualifies names as '<run>/<file>_evtNNN' so noise
        # seeds / holdout hashes / cache identity stay unique across runs
        # (single-run split= names are unchanged — byte-identical legacy).
        if runs is not None:
            if split != 'train':
                raise ValueError(
                    f"Pass either runs= (multi-run) or split= (single run), "
                    f"not both (got runs={runs!r}, split={split!r}).")
            split = self._expand_runs(runs, self._modalities)
        self._source_split = split

        self.step_reader = None
        self.sensor_reader = None
        self.labl_reader = None
        self.hits_reader = None

        if 'step' in self._modalities:
            self.step_reader = JAXTPCStepReader(
                data_root=self._modality_root('step'), split=split,
                dataset_name=dataset_name, min_deposits=min_deposits,
                include_physics=include_physics, volume=volume)

        # sensor/hits readout auto-detection. Build the readers unfiltered
        # first so they can detect readout_type, then apply the volume
        # plane filter using the correct plane labels for this readout.
        if 'sensor' in self._modalities:
            self.sensor_reader = JAXTPCSensorReader(
                data_root=self._modality_root('sensor'), split=split,
                dataset_name=dataset_name, planes='all',
                n_wires_per_plane=n_wires_per_plane,
                num_time_steps=num_time_steps,
                pedestal_per_plane=pedestal_per_plane)

        if self._want_labels:
            self.labl_reader = JAXTPCLablReader(
                data_root=self._modality_root('labl'), split=split,
                dataset_name=dataset_name, label_keys=label_keys)

        if 'hits' in self._modalities:
            self.hits_reader = JAXTPCHitsReader(
                data_root=self._modality_root('hits'), split=split,
                dataset_name=dataset_name, planes='all')

        # Resolve readout_type once from whichever reader can tell us.
        self._readout_type = 'wire'
        for r in (self.sensor_reader, self.hits_reader):
            if r is not None:
                self._readout_type = r.readout_type
                break

        # Now that readout_type is known, apply per-volume plane filter.
        if volume is not None:
            if self._readout_type == 'pixel':
                planes = [f'volume_{volume}_Pixel']
            else:
                planes = [f'volume_{volume}_U', f'volume_{volume}_V',
                          f'volume_{volume}_Y']
            if self.sensor_reader is not None:
                self.sensor_reader.planes = planes
            if self.hits_reader is not None:
                self.hits_reader.planes = planes

        self._canonical_reader = (self.step_reader or self.sensor_reader
                                  or self.hits_reader or self.labl_reader)
        # Phase A / D42: build ONE joint cross-modality event index and inject
        # it into every reader, so a single global idx maps to the SAME
        # physics event in all modalities. (Replaces the old
        # `_n_events = min(len(r) ...)`, which left each reader mapping idx
        # through its own present-key index → silent desync under
        # min_deposits>0 or a gap present in some-but-not-all modalities.)
        self._build_joint_index(
            source_label=f"JAXTPCDataset({self._source_data_root!r})",
            filter_label=(f"min_deposits={self._min_deposits}"
                          if self._min_deposits > 0 else ''))

        super().__init__(
            split=split, data_root=data_root,
            transform=transform, ignore_index=ignore_index, loop=loop,
        )

        # Fail fast on empty data_list — otherwise get_data() crashes later
        # with an opaque ZeroDivisionError on `idx % len(self.data_list)`.
        if len(self.data_list) == 0:
            raise ValueError(
                f"JAXTPCDataset(data_root={data_root!r}) yielded 0 events "
                f"after filters (min_deposits={min_deposits}, "
                f"max_len={max_len}). Lower min_deposits or verify the "
                f"dataset has events meeting it.")

    def get_data(self, idx):
        """Load one event as a nested dict (schema: see module docstring)."""
        # data_list VALUES are event ids. Identical to `idx % len(...)` while
        # the list is range(n), and the only correct form once it is not.
        real_idx = self.data_list[idx % len(self.data_list)]

        data = {
            'name': self.get_data_name(real_idx),
            'split': self.split if isinstance(self.split, str) else 'custom',
        }

        labl_by_volume = {}
        if self.labl_reader is not None:
            labl_by_volume = self._build_labl(self.labl_reader.read_event(real_idx))
            if self._volume is not None:
                # Drop labl volumes the user isn't loading (keeps the
                # dataset's view consistent with ``volume=`` on other readers).
                keep = f'v{self._volume}'
                labl_by_volume = {k: v for k, v in labl_by_volume.items()
                                  if k == keep}
            if labl_by_volume:
                data['labl'] = labl_by_volume

        if self.hits_reader is not None:
            hits_raw = self.hits_reader.read_event(real_idx)
            data['hits'] = self._build_hits_cloud(hits_raw, labl_by_volume)
            bridges = self._build_bridges(hits_raw)
            if bridges:
                data['bridges'] = bridges

        if self.sensor_reader is not None:
            data['sensor'] = self._build_sensor_cloud(
                self.sensor_reader.read_event(real_idx))
            # Surface the event id for transforms that need a stable per-event
            # seed (e.g. AddNoise) while scoped to the sensor sub-dict.
            data['sensor']['name'] = data['name']

        if self.step_reader is not None:
            data['step'] = self._build_step_cloud(
                self.step_reader.read_event(real_idx), labl_by_volume)

        return data

    # ------------------------------------------------------------------
    # Per-modality builders
    # ------------------------------------------------------------------

    def _build_step_cloud(self, step_raw, labl_by_volume):
        """3D deposit sub-dict; decorates with segment/instance if labl present."""
        sub = {}
        for k, v in step_raw.items():
            sub[k] = v  # coord, energy, volume_id, physics — readers emit bare

        if labl_by_volume and 'volume_id' in sub:
            segment, instance = self._decorate_step_from_labl(
                sub['volume_id'], labl_by_volume)
            sub['segment'] = segment
            sub['instance'] = instance
            for out, kind, lk in self._track_axes():
                if kind == 'self':
                    sub[out] = instance[:, None]   # the per-deposit track id
                    continue
                seg_axis, _ = self._decorate_step_from_labl(
                    sub['volume_id'], labl_by_volume, label_key=lk)
                sub[out] = seg_axis[:, None]

        return sub

    def _build_sensor_cloud(self, sensor_raw):
        """Merge per-plane sensor raw into a point cloud + raw passthrough."""
        coord_keys = self._coord_keys()
        planes, coord, energy, plane_id, raw = self._merge_plane_dotted(
            sensor_raw, prefix='sensor', value_key='value',
            coord_keys=coord_keys)
        sub = {
            'coord': coord, 'energy': energy, 'plane_id': plane_id,
            'planes': planes, 'raw': raw,
            'readout_type': self._readout_type,
        }
        # Per-plane fixed grid extent for a downstream Densify, from the reader's
        # surfaced geometry. Wire readout only; absent when geometry is unknown
        # (Densify then errors unless require_shape=False). Pedestal (raw-ADC
        # offset) is surfaced alongside for a downstream Digitize.
        if self._readout_type != 'pixel' and self.sensor_reader is not None:
            nts = getattr(self.sensor_reader, 'num_time_steps', None)
            nwbp = getattr(self.sensor_reader, 'n_wires_per_plane', {})
            shape = {p: (int(nwbp[p]), int(nts)) for p in planes
                     if p in nwbp and nts is not None}
            if shape:
                sub['shape'] = shape
            pbp = getattr(self.sensor_reader, 'pedestal_per_plane', {})
            pedestal = {p: int(pbp[p]) for p in planes if p in pbp}
            if pedestal:
                sub['pedestal'] = pedestal
            # Flat integer scatter inputs for the dense path (a downstream
            # Collect(into='sensor') pulls these; densify scatters them).
            # coord/energy are float32 but wire/time are small exact integers.
            if planes:
                sub['wire'] = coord[:, 0].astype(np.int32)
                sub['time'] = coord[:, 1].astype(np.int32)
                sub['value'] = energy[:, 0].astype(np.float32)
                # canonical (stable) plane id per point, parallel to the
                # positional plane_id (which is left unchanged for back-compat)
                gid_of_pos = np.array([canonical_plane_id(p) for p in planes],
                                      dtype=np.int32)
                sub['plane_gid'] = gid_of_pos[plane_id[:, 0]].astype(np.int32)
        return sub

    def _wire_lengths_for(self, label, n_wires):
        """Per-wire lengths (m) for a plane: (n_wires,) array, or (lo,hi)->linspace."""
        ptype = label.split('_')[-1]
        spec = self._wire_lengths_per_plane.get(
            label, self._wire_lengths_per_plane.get(ptype))
        if spec is None:
            return None
        arr = np.asarray(spec, dtype=np.float32)
        if arr.ndim == 1 and arr.shape[0] == n_wires:
            return arr
        if arr.size == 2:
            return np.linspace(float(arr.flat[0]), float(arr.flat[1]),
                               n_wires, dtype=np.float32)
        raise ValueError(
            f"wire_lengths for {label!r}: expected ({n_wires},) array or "
            f"(lo, hi) pair, got shape {arr.shape}")

    def plane_geometry(self):
        """Canonical-plane-id -> geometry registry for the dense path.

        Returns ``{gid: {'n_wires', 'n_ticks', 'pedestal', ['wire_lengths']}}``
        built from the sensor reader's file-surfaced geometry plus the
        ``wire_lengths_per_plane`` config. Geometry is sourced HERE (the registry),
        not carried per-sample through collate; the post-collate densify/noise
        stages take this dict. ``wire_lengths`` is present only when configured
        (the incoherent-noise stage requires it; densify/coherent/digitize do not).
        Note the reader populates ``n_wires_per_plane`` lazily — read at least one
        event (or build after indexing) before relying on this.
        """
        if self.sensor_reader is None or self._readout_type == 'pixel':
            return {}
        # the reader fills n_wires_per_plane lazily on read; populate it if this
        # is called before the loader has iterated (e.g. trainer before_train).
        if not getattr(self.sensor_reader, 'n_wires_per_plane', {}):
            try:
                self.sensor_reader.read_event(0)
            except Exception:
                pass
        nts = getattr(self.sensor_reader, 'num_time_steps', None)
        nwbp = getattr(self.sensor_reader, 'n_wires_per_plane', {})
        pbp = getattr(self.sensor_reader, 'pedestal_per_plane', {})
        geom = {}
        for label, nw in nwbp.items():
            entry = {'n_wires': int(nw)}
            if nts is not None:
                entry['n_ticks'] = int(nts)
            if label in pbp:
                entry['pedestal'] = int(pbp[label])
            wl = self._wire_lengths_for(label, int(nw))
            if wl is not None:
                entry['wire_lengths'] = wl
            geom[canonical_plane_id(label)] = entry
        return geom

    def _build_hits_cloud(self, hits_raw, labl_by_volume):
        """Merge per-plane hits raw into a point cloud + raw passthrough.

        Attaches ``segment`` when labl available (via group_to_track chain).
        ``instance`` is always attached (== group_id).
        """
        coord_keys = self._coord_keys()
        planes, coord, energy, plane_id, raw = self._merge_plane_dotted(
            hits_raw, prefix='hits', value_key='charge',
            coord_keys=coord_keys, extra_keys=('group_id',))
        # instance = per-entry group_id (already int32 from the reader, so
        # no astype copy needed)
        instance = (np.concatenate([raw[p]['group_id'] for p in planes], axis=0)
                    if planes else np.zeros(0, dtype=np.int32))

        sub = {
            'coord': coord, 'energy': energy, 'plane_id': plane_id,
            'instance': instance, 'planes': planes, 'raw': raw,
            'readout_type': self._readout_type,
        }
        if labl_by_volume:
            sub['segment'] = self._decorate_hits_from_labl(
                planes, raw, hits_raw, labl_by_volume)
            for out, kind, lk in self._track_axes():
                if kind == 'self':
                    sub[out] = instance[:, None]   # bare instance (group_id)
                    continue
                sub[out] = self._decorate_hits_from_labl(
                    planes, raw, hits_raw, labl_by_volume,
                    label_key=lk)[:, None]
        return sub

    def _coord_keys(self):
        """Per-plane column names that build the sensor/hits coord vector."""
        if self._readout_type == 'pixel':
            return ('py', 'pz', 'time')
        return ('wire', 'time')

    def _build_labl(self, labl_flat):
        """Convert flat labl_v{N}_col keys into nested {v{N}: {col: arr}}.

        Key format ``labl_v{idx}_{col}``; ``col`` may contain underscores, so
        the greedy ``(.+)`` captures it whole."""
        by_volume = {}
        for k, v in labl_flat.items():
            m = re.match(r'labl_(v\d+)_(.+)', k)
            assert m is not None, k
            by_volume.setdefault(m.group(1), {})[m.group(2)] = v
        return by_volume

    def _build_bridges(self, hits_raw):
        """Extract per-volume bridge arrays (g2t, deposit_to_group, qs_fractions).

        Under a ``volume=`` filter (F13), keep only that volume's bridges — the
        hits reader still loads every volume's ``*_v{N}`` tables, but the other
        volumes' points are not loaded, so carrying their group machinery is
        wasted payload that confuses downstream consumers."""
        want = None if self._volume is None else str(self._volume)
        bridges = {}
        for k, v in hits_raw.items():
            if (k.startswith('group_to_track_v')
                    or k.startswith('deposit_to_group_v')
                    or k.startswith('qs_fractions_v')):
                if want is not None and k.rsplit('_v', 1)[1] != want:
                    continue
                bridges[k] = v
        return bridges

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _merge_plane_dotted(raw_dict, prefix, value_key, coord_keys,
                            extra_keys=()):
        """Merge `{prefix}.{plane}.{col}` flat keys into a single point cloud.

        ``coord_keys`` is the ordered tuple of per-plane column names that
        form the coord vector (e.g. ``('wire', 'time')`` for wire readout,
        ``('py', 'pz', 'time')`` for pixel). Coord shape is ``(M, len(coord_keys))``.

        Returns (planes, coord, energy, plane_id, raw_nested).
        raw_nested is ``{plane_label: {*coord_keys, value_key, *extra_keys}}``.
        """
        coord_dim = len(coord_keys)
        # Discover planes by scanning for any coord-key suffix (they all
        # share the same plane set, so the first key is enough).
        first_key = coord_keys[0]
        planes = sorted(set(
            k.split('.')[1] for k in raw_dict
            if k.startswith(prefix + '.') and k.endswith('.' + first_key)
        ))
        # Gather per-plane columns + sizes (and build raw_nested) in one pass,
        # then fill preallocated output arrays per-plane. This avoids the
        # np.stack-per-plane + concatenate-across-planes copies (the single
        # biggest reader op on large events); the slice assignment casts
        # int -> float32 in place.
        raw_nested = {}
        plane_cols, sizes = [], []
        for plane in planes:
            cols_arrays = [raw_dict[f'{prefix}.{plane}.{ck}']
                           for ck in coord_keys]
            value = raw_dict[f'{prefix}.{plane}.{value_key}']
            plane_cols.append((cols_arrays, value))
            sizes.append(len(cols_arrays[0]))
            cols = {ck: arr for ck, arr in zip(coord_keys, cols_arrays)}
            cols[value_key] = value
            for ek in extra_keys:
                cols[ek] = raw_dict[f'{prefix}.{plane}.{ek}']
            raw_nested[plane] = cols

        N = sum(sizes)
        coord = np.empty((N, coord_dim), dtype=np.float32)
        energy = np.empty((N, 1), dtype=np.float32)
        plane_id = np.empty((N, 1), dtype=np.int32)
        off = 0
        for i, ((cols_arrays, value), n) in enumerate(zip(plane_cols, sizes)):
            sl = slice(off, off + n)
            for j, arr in enumerate(cols_arrays):
                coord[sl, j] = arr
            energy[sl, 0] = value
            plane_id[sl, 0] = i
            off += n

        return planes, coord, energy, plane_id, raw_nested

    def _validate_label_config(self):
        """Reject ``label_config`` specs the JAXTPC path can't honor (F4).

        JAXTPC labl is per-volume ``track`` tables value-keyed by ``track_ids``
        — there is no event-level or ``particle`` table as in LUCiD. So a spec
        that LUCiD's shared ``decorate_labels`` would satisfy (event scope,
        ``('particle'/'event', col)`` source) has no JAXTPC analog. Raise here
        rather than silently dropping the axis, so the same ``label_config``
        either produces the named key on both detectors or fails loud — never
        diverges quietly. Supported: ``scope='point'`` with ``source='self'``
        (emit the per-point track id) or ``source=('track', col)`` (value-keyed
        gather of ``track_<col>``)."""
        if not self._label_config:
            return
        for spec in self._label_config:
            out = spec.get('out', '?')
            scope = spec.get('scope', 'point')
            if scope != 'point':
                raise ValueError(
                    f"JAXTPCDataset label_config[{out!r}]: scope={scope!r} "
                    "unsupported (no event-level labl table); use scope='point'.")
            src = spec.get('source', 'self')
            ok = (src == 'self'
                  or (isinstance(src, (tuple, list)) and len(src) == 2
                      and src[0] == 'track'))
            if not ok:
                raise ValueError(
                    f"JAXTPCDataset label_config[{out!r}]: source={src!r} "
                    "unsupported; use 'self' or ('track', <column>). "
                    "(LUCiD's ('particle'/'event', …) tables have no JAXTPC "
                    "analog — this would silently drop the axis.)")
            kb = spec.get('keyed_by')
            if kb is not None and kb != 'track_ids':
                raise ValueError(
                    f"JAXTPCDataset label_config[{out!r}]: keyed_by={kb!r} "
                    "unsupported; JAXTPC track columns are keyed by 'track_ids'.")

    def _track_axes(self):
        """``(out_key, kind, column)`` for each point-scope ``label_config`` axis.

        ``kind='self'`` → emit the bare per-modality ``instance`` axis under the
        named key (parity with LUCiD ``source='self'``, where ``self`` == the
        ``instance`` identifier); ``kind='track'`` → value-keyed
        per-volume gather of ``track_<column>`` to emit named schema keys (e.g.
        ``segment_pid`` from ``track_pdg``). Validated by
        :meth:`_validate_label_config`, so unsupported specs raise at
        construction and are never silently dropped here."""
        if not self._label_config:
            return []
        axes = []
        for spec in self._label_config:
            if spec.get('scope', 'point') != 'point':
                continue
            src = spec.get('source', 'self')
            if src == 'self':
                axes.append((spec['out'], 'self', None))
            elif (isinstance(src, (tuple, list)) and len(src) == 2
                    and src[0] == 'track'):
                axes.append((spec['out'], 'track', src[1]))
        return axes

    def _decorate_step_from_labl(self, volume_id, labl_by_volume,
                                 label_key=None):
        """Broadcast per-track labl data onto each step deposit.

        Uses ``labl[vN]['deposit_to_track']`` (row-aligned to the volume's
        step deposits) as the per-deposit FK, then looks up
        ``labl[vN]['track_{label_key}']`` via binary search on ``track_ids``.
        ``label_key`` defaults to ``self._label_key``; pass an explicit key to
        gather a different axis (used by ``label_config``).
        """
        vid_flat = volume_id.ravel()
        n_total = vid_flat.shape[0]
        instance = np.full(n_total, -1, dtype=np.int32)
        segment = np.full(n_total, -1, dtype=np.int32)
        meta_col = f'track_{label_key or self._label_key}'

        for vkey, vdata in labl_by_volume.items():
            vol_num = int(vkey[1:])
            mask = vid_flat == vol_num
            if not mask.any():
                continue
            if 'deposit_to_track' not in vdata:
                continue
            per_dep_tid = vdata['deposit_to_track'].astype(np.int32)
            n_vol = int(mask.sum())
            if per_dep_tid.shape[0] != n_vol:
                log.warning("labl.%s.deposit_to_track len %d != step vol %d len %d",
                            vkey, per_dep_tid.shape[0], vol_num, n_vol)
                continue
            instance[mask] = per_dep_tid

            if 'track_ids' in vdata and meta_col in vdata:
                # value-keyed gather on track_ids (shared primitive; guards
                # empty track_ids, which the old inline searchsorted crashed on)
                segment[mask] = gather_with_fill(
                    per_dep_tid, vdata[meta_col], keyed_by=vdata['track_ids'])

        return segment, instance

    def _decorate_hits_from_labl(self, planes, raw_nested, hits_flat,
                                 labl_by_volume, label_key=None):
        """Per-hits-entry segment label via group_to_track → track lookup.

        ``label_key`` defaults to ``self._label_key``; pass an explicit key to
        gather a different axis (used by ``label_config``)."""
        meta_col = f'track_{label_key or self._label_key}'
        all_labels = []
        for plane in planes:
            cols = raw_nested[plane]
            gid = cols['group_id']
            # plane label is 'volume_{v}_{U|V|Y}' — extract volume index
            vol_idx_str = plane.split('_')[1]
            vkey = f'v{vol_idx_str}'

            n = gid.shape[0]
            labels = np.full(n, -1, dtype=np.int32)

            g2t_key = f'group_to_track_v{vol_idx_str}'
            g2t = hits_flat.get(g2t_key)
            if g2t is None or vkey not in labl_by_volume:
                all_labels.append(labels)
                continue
            vdata = labl_by_volume[vkey]
            if 'track_ids' not in vdata or meta_col not in vdata:
                all_labels.append(labels)
                continue

            # group_id → track_id hop, then value-keyed gather on track_ids
            valid = (gid >= 0) & (gid < len(g2t))
            tids = np.where(valid, g2t[gid], -1)
            labels = gather_with_fill(
                tids, vdata[meta_col], keyed_by=vdata['track_ids'])
            all_labels.append(labels)

        if not all_labels:
            return np.zeros(0, dtype=np.int32)
        return np.concatenate(all_labels, axis=0)
