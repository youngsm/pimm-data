"""
Detector-specific transforms.

``PDGToSemantic``: derives semantic labels from per-deposit PDG codes
in step data when no labl file is available (fallback path).

``RemapSegment``: maps the values in ``data_dict['segment']`` through a
user-provided ``{old_value: new_value}`` dict, with a default for
unmapped values. Typical use: after loading with ``label_key='pdg'``
(so ``segment`` holds raw PDG codes), remap to a task-specific set of
class indices before loss. Works for any int-valued segment column
(pdg, interaction, cluster) — not PDG-specific.

``Densify`` / ``AddNoise`` / ``Digitize``: the dense sensor-modality chain. The
full forward path mirrors production ``make_noisy``::

    Apply(on='sensor', transforms=[
        dict(type='Densify'),    # sparse COO -> dense (n_wires, n_ticks)
        dict(type='AddNoise'),   # img += generate_noise(...)  (incoherent/coherent)
        dict(type='Digitize'),   # round(img+pedestal).clip(0, adc_max) - pedestal
    ])

All three operate on the ``sensor`` sub-dict — wrap them in
``Apply(on='sensor', transforms=[...])``.
"""

import copy
import hashlib

import numpy as np
import torch

from .transform import TRANSFORMS, Compose
from .noise import (generate_noise, digitize, DEFAULT_ENC, DEFAULT_SERIES_SPECTRUM,
                    DEFAULT_SAMPLING_RATE_HZ)
from .utils.pdg import pdg_to_semantic, MOTIF_MAP, PID_MAP

_NAMED_SCHEMES = {
    'motif_5cls': (MOTIF_MAP, 4),
    'pid_6cls': (PID_MAP, 5),
}


@TRANSFORMS.register_module()
class Apply:
    """Scope a sub-pipeline to one or more parts (REDESIGN §4).

    During the pre-collate pipeline a sample is nested per part
    (``{'step': {'coord': …}, 'sensor': {…}}``). The augmentation library hardcodes
    bare keys (``'coord'``/``'segment'``), so wrap it in ``Apply`` to scope it to a
    part::

        dict(type='Apply', on='step', transforms=[
            dict(type='GridSample', grid_size=0.5),
            dict(type='RandomRotate', keys=('coord',)),
        ])

    ``on`` is one part (str) or several (tuple/list). A **multi-part** ``Apply`` is
    **implicitly shared**: it restores the numpy RNG state before each part so every
    *global* random decision (rotation angle, flip, scale) is **identical** across
    them — co-registering parts that share a coordinate frame. There is **no
    ``shared`` flag**: the config shape carries the intent — a tuple ``on=`` is
    shared/together; **independent augmentation = separate ``Apply`` blocks**.
    Missing parts are a no-op unless ``required``.

    (Scope of "shared" = the global draws — a fixed, size-independent count
    (angle/flip/scale); per-point transforms like jitter have no well-defined
    cross-part "same". The inner transforms must use the global ``np.random``.)
    """

    def __init__(self, on, transforms=None, required=False):
        self.on = [on] if isinstance(on, str) else list(on)
        self.required = bool(required)
        self.inner = Compose(transforms) if transforms else None

    def __call__(self, data_dict):
        if self.inner is None:
            return data_dict
        missing = [m for m in self.on if m not in data_dict]
        if missing and self.required:
            raise KeyError(
                f"Apply(on={self.on!r}) applied but data_dict is missing {missing!r}; "
                f"available: {sorted(k for k in data_dict if isinstance(data_dict.get(k), dict))}")
        present = [m for m in self.on if m in data_dict]
        if not present:
            return data_dict
        # multi-part is IMPLICITLY shared: replay the same RNG draw per part so
        # geometric decisions co-register. Single part -> no state juggling.
        state = np.random.get_state() if len(present) > 1 else None
        for m in present:
            if state is not None:
                np.random.set_state(state)
            data_dict[m] = self.inner(data_dict[m])
        return data_dict


def _knn_edges(coord, k):
    """Directed kNN edges ``(2, E)`` (torch): row0 src, row1 the k nearest neighbours.

    Uses ``torch.cdist`` + ``topk`` (GPU-ready). Returns a ``torch.long`` tensor —
    Collect/collate carry it through; the pre-collate numpy parts coexist fine.
    """
    coord = torch.as_tensor(coord).float()
    n = coord.shape[0]
    if n <= 1:
        return torch.zeros((2, 0), dtype=torch.long)
    k = min(int(k), n - 1)
    d = torch.cdist(coord, coord)
    d.fill_diagonal_(float('inf'))
    nbr = d.topk(k, largest=False).indices                 # (n, k)
    src = torch.arange(n).repeat_interleave(k)
    return torch.stack([src, nbr.reshape(-1)]).long()


@TRANSFORMS.register_module()
class SetupGraph:
    """Producer: build a kNN graph on a part's points (REDESIGN §4, the graph case).

    Reads ``<on>.coord``, writes ``<on>.edge_index`` (2, E) into the part and stamps
    its role ``('edge','self')`` in the part's ``_roles`` so collate shifts edges by
    the part's running node count. List ``edge_index`` in that part's ``Collect`` keys.
    """

    def __init__(self, on='step', k=16, coord_key='coord', out_key='edge_index'):
        self.on, self.k = on, int(k)
        self.coord_key, self.out_key = coord_key, out_key

    def __call__(self, data_dict):
        part = data_dict[self.on]
        part[self.out_key] = _knn_edges(np.asarray(part[self.coord_key]), self.k)
        part.setdefault('_roles', {})[self.out_key] = ('edge', 'self')
        return data_dict


@TRANSFORMS.register_module()
class BuildNexus:
    """Producer: bipartite cross-store edges between two parts (NuGraph 'nexus').

    Reads ``<src>.coord`` and ``<dst>.coord``, writes a NEW edge-only part
    (``to``, default 'nexus') holding ``edge_index`` (2, E) — row0 indexes ``src``,
    row1 indexes ``dst`` — with role ``('edge',(src,dst))`` so collate shifts each
    row by the corresponding part's node count. Collect the edge part with
    ``offset_keys_dict={}`` (it has no points of its own).
    """

    def __init__(self, on=('hit', 'sp'), to='nexus', k=1, coord_key='coord'):
        self.src, self.dst = on
        self.to, self.k, self.coord_key = to, int(k), coord_key

    def __call__(self, data_dict):
        s = torch.as_tensor(data_dict[self.src][self.coord_key]).float()
        d = torch.as_tensor(data_dict[self.dst][self.coord_key]).float()
        if s.shape[0] == 0 or d.shape[0] == 0:
            ei = torch.zeros((2, 0), dtype=torch.long)
        else:
            dim = min(s.shape[1], d.shape[1])              # tolerate differing coord dims
            dist = torch.cdist(s[:, :dim], d[:, :dim])      # (Ns, Nd)
            kk = min(self.k, d.shape[0])
            nbr = dist.topk(kk, largest=False).indices      # (Ns, kk)
            src = torch.arange(s.shape[0]).repeat_interleave(kk)
            ei = torch.stack([src, nbr.reshape(-1)]).long()
        data_dict[self.to] = {'edge_index': ei,
                              '_roles': {'edge_index': ('edge', (self.src, self.dst))}}
        return data_dict


@TRANSFORMS.register_module()
class Align:
    """Reorder parts row-for-row to match a reference part (REDESIGN §8).

    Multi-task per-voxel heads need parts (e.g. ``image`` and ``cluster``) aligned
    row-for-row. ``Align(to='image', parts=('cluster',))`` permutes each listed
    part's per-point rows so row ``i`` corresponds to ``image.coord[i]`` (SPINE's
    ``clean_sparse_data``). Each aligned part must hold the SAME coord set as ``to``.
    """

    def __init__(self, to, parts, coord_key='coord'):
        self.to = to
        self.parts = [parts] if isinstance(parts, str) else list(parts)
        self.coord_key = coord_key

    def __call__(self, data_dict):
        ref = np.asarray(data_dict[self.to][self.coord_key])
        ref_rows = [tuple(r) for r in ref.tolist()]
        for p in self.parts:
            part = data_dict[p]
            pc = np.asarray(part[self.coord_key])
            if pc.shape[0] != ref.shape[0]:
                raise ValueError(
                    f"Align: part {p!r} has {pc.shape[0]} rows != reference "
                    f"{self.to!r} ({ref.shape[0]}); can only align identical coord sets.")
            pos = {tuple(r): j for j, r in enumerate(pc.tolist())}
            try:
                order = np.array([pos[r] for r in ref_rows], dtype='int64')
            except KeyError as e:
                raise ValueError(
                    f"Align: part {p!r} coord row {e} not found in reference "
                    f"{self.to!r} — coord sets differ.") from None
            n = ref.shape[0]
            for k, v in list(part.items()):
                if k == '_roles':
                    continue
                arr = v
                if hasattr(arr, 'shape') and getattr(arr, 'ndim', 0) >= 1 and arr.shape[0] == n:
                    part[k] = arr[order]
        return data_dict


@TRANSFORMS.register_module()
class MultiCrop:
    """Produce packed multi-crop view PARTS from a source part (SSL multi-view).

    Reads the source part ``on`` (a nested sub-dict with ``coord`` + ``view_keys``)
    and writes two new parts (default ``global``/``local``), each **packing**
    ``global_view_num``/``local_view_num`` center-sampled crops concatenated with an
    ``offset`` separating them. ``global_shared_transform`` runs ONCE on the global
    source (so all global crops share it); per-crop ``global_transform`` /
    ``local_transform`` run on each crop with independent draws. Local crop centers
    are sampled WITHIN the major global crop (co-location).

    Collect the resulting parts (``Collect(parts={'global': dict(keys=('coord',
    'offset'), offset_keys_dict={}, feat_keys=…), 'local': …})``) to flat
    ``global_*``/``local_*``; collate's offset cumsum then packs ``B*num`` crops. The
    per-crop ``offset`` is carried through (``offset_keys_dict={}`` suppresses the
    derived one).

    A focused port of pimm's ``MultiViewGenerator`` (anchor/cnms center sampling
    omitted — uses uniform random centers).
    """

    def __init__(self, on='step', view_keys=('coord', 'energy'),
                 global_view_num=2, global_view_scale=(0.4, 1.0),
                 local_view_num=6, local_view_scale=(0.1, 0.4),
                 global_shared_transform=None, global_transform=None,
                 local_transform=None, global_name='global', local_name='local',
                 max_size=65536):
        assert 'coord' in view_keys, "MultiCrop: 'coord' must be in view_keys"
        self.on = on
        self.view_keys = tuple(view_keys)
        self.gnum, self.gscale = int(global_view_num), tuple(global_view_scale)
        self.lnum, self.lscale = int(local_view_num), tuple(local_view_scale)
        self.gshared = Compose(global_shared_transform)
        self.gtrans = Compose(global_transform)
        self.ltrans = Compose(local_transform)
        self.gname, self.lname = global_name, local_name
        self.max_size = int(max_size)

    def _get_view(self, src, center, scale):
        coord = src['coord']
        max_size = min(self.max_size, coord.shape[0])
        size = max(1, min(max_size, int(np.random.uniform(*scale) * max_size)))
        idx = np.argsort(np.sum((coord - center) ** 2, axis=-1))[:size]
        return {k: src[k][idx] for k in self.view_keys if k in src}

    def _pack(self, crops):
        out = {}
        for k in self.view_keys:
            if all(k in c for c in crops):
                out[k] = np.concatenate([c[k] for c in crops], axis=0)
        out['offset'] = np.cumsum([c['coord'].shape[0] for c in crops]).astype('int64')
        return out

    def __call__(self, data_dict):
        src0 = data_dict[self.on]
        # global source: shared transform applied ONCE, then crops sampled from it
        gsrc = self.gshared(copy.deepcopy(src0))
        gcoord = gsrc['coord']
        major = self._get_view(gsrc, gcoord[np.random.randint(gcoord.shape[0])], self.gscale)
        globals_ = [major]
        for _ in range(self.gnum - 1):
            c = major['coord'][np.random.randint(major['coord'].shape[0])]
            globals_.append(self._get_view(gsrc, c, self.gscale))
        data_dict[self.gname] = self._pack(
            [self.gtrans(copy.deepcopy(v)) for v in globals_])
        # local crops: centers within the major global crop (co-location)
        locals_ = []
        for _ in range(self.lnum):
            c = major['coord'][np.random.randint(major['coord'].shape[0])]
            locals_.append(self._get_view(src0, c, self.lscale))
        data_dict[self.lname] = self._pack(
            [self.ltrans(copy.deepcopy(v)) for v in locals_])
        return data_dict


@TRANSFORMS.register_module()
class PDGToSemantic:
    """Fallback: derive approximate semantic labels from PDG codes.

    Schemes
    -------
    motif_5cls : shower(0), track(1), michel(2), delta(3), led(4)
    pid_6cls : photon(0), electron(1), muon(2), pion(3), proton(4), other(5)
    custom : user-provided {pdg_code: class_index} dict
    """

    def __init__(self, scheme='motif_5cls', custom_map=None):
        if scheme not in ('motif_5cls', 'pid_6cls', 'custom', 'none'):
            raise ValueError(f"Unknown label scheme: {scheme}")
        if scheme == 'custom':
            assert custom_map is not None
        self.scheme = scheme
        self.custom_map = custom_map

    def __call__(self, data_dict):
        if self.scheme == 'none' or 'pdg' not in data_dict:
            return data_dict

        if 'segment' in data_dict or 'segment_motif' in data_dict:
            return data_dict

        pdg = data_dict['pdg']
        labels = pdg_to_semantic(pdg, scheme=self.scheme,
                                 custom_map=self.custom_map)
        data_dict['segment_motif'] = labels[:, None]

        if self.scheme == 'motif_5cls':
            pid = pdg_to_semantic(pdg, scheme='pid_6cls')
            data_dict['segment_pid'] = pid[:, None]
        elif self.scheme == 'pid_6cls':
            data_dict['segment_pid'] = labels[:, None]

        n = len(labels)

        if 'instance_particle' not in data_dict and 'track_ids' in data_dict:
            track_ids = data_dict['track_ids']
            mask = track_ids >= 0
            if mask.any():
                _, inverse = np.unique(track_ids[mask], return_inverse=True)
                out = np.full(n, -1, dtype=np.int32)
                out[mask] = inverse
                data_dict['instance_particle'] = out[:, None]
            else:
                data_dict['instance_particle'] = np.full((n, 1), -1, dtype=np.int32)

        if 'instance_interaction' not in data_dict and 'interaction_ids' in data_dict:
            iids = data_dict['interaction_ids']
            mask = iids >= 0
            if mask.any():
                _, inverse = np.unique(iids[mask], return_inverse=True)
                out = np.full(n, -1, dtype=np.int32)
                out[mask] = inverse
                data_dict['instance_interaction'] = out[:, None]
            else:
                data_dict['instance_interaction'] = np.full((n, 1), -1, dtype=np.int32)

            data_dict['segment_interaction'] = (iids[:, None] != -1).astype(np.int32)

        return data_dict


@TRANSFORMS.register_module()
class RemapSegment:
    """Remap integer values in a segment-like field through a lookup dict.

    Motivating case: production labl files store raw PDG codes in the
    ``pdg`` column. A downstream task wants 5-class indices. Configure
    ``JAXTPCDataset`` with ``label_key='pdg'`` (so ``segment`` contains
    raw PDG codes per deposit), then add::

        dict(type="RemapSegment", scheme="motif_5cls")

    or with an explicit map::

        dict(type="RemapSegment",
             mapping={22: 0, 11: 0, 13: 1, 211: 1, 2212: 1},
             default=2)

    Parameters
    ----------
    mapping : dict[int, int], optional
        Explicit ``{source_value: target_class}`` map. Mutually
        exclusive with ``scheme``.
    default : int, optional
        Class index for values not in ``mapping``. If ``mapping`` is
        given, default is ``max(mapping.values()) + 1``; if ``scheme``
        is given, scheme-specific default (4 for motif_5cls, 5 for
        pid_6cls). Can be overridden explicitly.
    scheme : str, optional
        One of ``'motif_5cls'``, ``'pid_6cls'``. Loads the built-in
        PDG→class map. Overridden by ``mapping`` if both are given.
    key : str, optional
        Which data-dict field to remap. Default ``'segment'``.
    ignore_value : int, optional
        Source values equal to this are written as-is (bypass default).
        Default ``-1`` so ignore-index sentinels survive remapping.
    """

    def __init__(self, mapping=None, default=None, scheme=None,
                 key='segment', ignore_value=-1):
        if mapping is None and scheme is None:
            raise ValueError("RemapSegment needs either `mapping` or `scheme`")
        if mapping is not None:
            self._map = {int(k): int(v) for k, v in mapping.items()}
            self._default = int(default) if default is not None \
                else max(self._map.values()) + 1
        else:
            if scheme not in _NAMED_SCHEMES:
                raise ValueError(f"Unknown scheme {scheme!r}; "
                                 f"pick from {list(_NAMED_SCHEMES)} or pass `mapping`")
            base_map, scheme_default = _NAMED_SCHEMES[scheme]
            self._map = dict(base_map)
            self._default = int(default) if default is not None else scheme_default
        self._key = key
        self._ignore_value = int(ignore_value)

    def __call__(self, data_dict):
        seg = data_dict.get(self._key)
        if seg is None:
            return data_dict

        seg = np.asarray(seg)
        orig_shape = seg.shape
        flat = seg.ravel()

        out = np.full(flat.shape, self._default, dtype=np.int32)
        # Preserve ignore sentinels
        ignore_mask = flat == self._ignore_value
        out[ignore_mask] = self._ignore_value
        # Apply the map
        for src, dst in self._map.items():
            out[(flat == src) & ~ignore_mask] = dst

        data_dict[self._key] = out.reshape(orig_shape)
        return data_dict


@TRANSFORMS.register_module()
class AggregateSensorHits:
    """Aggregate a LUCiD sensor event's multiple hits per PMT into one point.

    A raw LUCiD ``sensor`` event has several hits per PMT; this groups by
    ``sensor_idx`` and emits one point per PMT — ``coord`` (the shared PMT
    position), ``energy`` (summed PE), ``time`` (aggregated by
    ``time_aggregation``), ``sensor_idx`` (unique). It reads the ``modality``
    sub-dict (``MultiModalEventDataset`` emits nested dicts) and writes the
    aggregated arrays to the **top level** — the flat ``coord``/``energy``/
    ``time`` keys the event-SSL pipeline consumes — then drops the consumed
    sub-dict. This replaces the inline aggregation the dissolved
    ``LUCiDEventSSLDataset`` did (D32 / the colleague's registered-transform
    suggestion).

    ``time_aggregation`` ∈ ``{'earliest', 'mean', 'pe_weighted', 'first'}``
    (the LUCiDEventSSLDataset set): minimum / count-mean / PE-weighted-mean /
    first-hit time per PMT. When ``modality`` is absent (already flat) it
    operates on the top-level keys in place.
    """

    _STRATEGIES = ('earliest', 'mean', 'pe_weighted', 'first')

    def __init__(self, modality='sensor', time_aggregation='earliest',
                 coord_key='coord', energy_key='energy', time_key='time',
                 sensor_key='sensor_idx', flatten=True):
        if time_aggregation not in self._STRATEGIES:
            raise ValueError(
                f"time_aggregation must be one of {self._STRATEGIES}, "
                f"got {time_aggregation!r}")
        self.modality = modality
        self.time_aggregation = time_aggregation
        self.coord_key = coord_key
        self.energy_key = energy_key
        self.time_key = time_key
        self.sensor_key = sensor_key
        # flatten=True (default, legacy): lift the aggregated point cloud to the
        # TOP level and drop the sub-dict (the old event-SSL flat flow).
        # flatten=False keeps it NESTED in the modality sub-dict, which the new
        # flat-prefixed flow needs (Apply(on=) / MultiCrop(on=) /
        # Collect(parts=) all read a sub-dict).
        self.flatten = flatten

    def __call__(self, data_dict):
        sub = data_dict.get(self.modality)
        src = sub if isinstance(sub, dict) else data_dict
        sensor_idx = src.get(self.sensor_key)
        if sensor_idx is None:
            return data_dict
        agg = self._aggregate(
            np.asarray(sensor_idx).reshape(-1),
            np.asarray(src[self.coord_key]),
            np.asarray(src[self.energy_key]),
            np.asarray(src[self.time_key]))
        if isinstance(sub, dict) and not self.flatten:
            sub.update(agg)                       # keep nested (new-API flow)
        else:
            for k, v in agg.items():
                data_dict[k] = v
            if isinstance(sub, dict):             # lift out of the modality
                data_dict.pop(self.modality, None)
        return data_dict

    def _aggregate(self, sensor_idx, coord, energy, time):
        out = {self.coord_key: coord.astype(np.float32, copy=False),
               self.energy_key: energy.astype(np.float32, copy=False),
               self.time_key: time.astype(np.float32, copy=False),
               self.sensor_key: sensor_idx.astype(np.int64, copy=False)}
        if sensor_idx.size == 0:
            return out
        order = np.argsort(sensor_idx, kind='stable')
        s_sid = sensor_idx[order]
        uniq, starts = np.unique(s_sid, return_index=True)
        e_sorted = energy[order]
        t_sorted = time[order]
        energy_a = np.add.reduceat(e_sorted, starts, axis=0)
        if self.time_aggregation == 'earliest':
            time_a = np.minimum.reduceat(t_sorted, starts, axis=0)
        elif self.time_aggregation == 'mean':
            counts = np.diff(np.r_[starts, s_sid.shape[0]])
            counts = counts.reshape((-1,) + (1,) * (t_sorted.ndim - 1))
            time_a = np.add.reduceat(t_sorted, starts, axis=0) / counts
        elif self.time_aggregation == 'pe_weighted':
            weighted = np.add.reduceat(t_sorted * e_sorted, starts, axis=0)
            time_a = weighted / np.maximum(energy_a, 1.0e-6)
        else:  # 'first'
            time_a = t_sorted[starts]
        out[self.coord_key] = coord[order][starts].astype(np.float32, copy=False)
        out[self.energy_key] = energy_a.astype(np.float32, copy=False)
        out[self.time_key] = time_a.astype(np.float32, copy=False)
        out[self.sensor_key] = uniq.astype(np.int64, copy=False)
        return out


@TRANSFORMS.register_module()
class Densify:
    """Scatter sparse COO sensor planes into dense ``(n_wires, n_ticks)`` images.

    Operates on a ``sensor`` sub-dict (wrap in
    ``Apply(on='sensor', ...)``). Reads ``sub['raw'][label]``
    (``{'wire', 'time', 'value'}``, already absolute indices + pedestal-
    subtracted by the reader) and the per-plane grid extent from
    ``sub['shape'][label] = (n_wires, n_ticks)``. Writes
    ``sub['dense'][label] = (n_wires, n_ticks) float32`` and **keeps** the point
    cloud (``coord`` / ``energy`` / ``raw``) intact.

    Wire readout only — a dense 2-D image is meaningless for pixel readout, so
    ``on_pixel='raise'`` (default) errors and ``on_pixel='skip'`` no-ops, letting
    one config run mixed datasets.

    Parameters
    ----------
    fill : float
        Value for empty (unhit) cells. Default ``0.0`` — matches the reader's
        pedestal-subtracted convention (empty = baseline = 0).
    dtype : str
        Output dtype. Default ``'float32'``.
    on_pixel : {'raise', 'skip'}
        Behaviour when the modality is pixel readout.
    require_shape : bool
        If ``True`` (default), raise when a plane has no ``shape`` entry. If
        ``False``, fall back to data-inferred extents (``max+1``) — note this is
        event-dependent and breaks fixed-geometry batching.
    dense_key : str
        Sub-dict key to write. Default ``'dense'``.
    """

    scope = 'sample'  # per-event independent; placeable pre- OR post-collate

    def __init__(self, geom=None, *, modality=None, fill=0.0, dtype='float32',
                 on_pixel='raise', require_shape=True, dense_key='dense',
                 assert_unique=True, wire_key='wire', time_key='time',
                 value_key='value', plane_key='plane_gid', offset_key='offset'):
        if on_pixel not in ('raise', 'skip'):
            raise ValueError(f"on_pixel must be 'raise' or 'skip', got {on_pixel!r}")
        self.fill = float(fill)
        self.dtype = dtype
        self.on_pixel = on_pixel
        self.require_shape = bool(require_shape)
        self.dense_key = dense_key
        self.assert_unique = bool(assert_unique)
        # flat collated-batch path (post-collate, torch): geom + the flat COO keys.
        self.geom = geom
        self.modality = modality
        self.wire_key, self.time_key, self.value_key = wire_key, time_key, value_key
        self.plane_key, self.offset_key = plane_key, offset_key

    def __call__(self, data):
        # DISPATCH: a post-collate collated batch carries the flat
        # ``<mod>_offset`` key (added by Collect/collate); a per-event sensor
        # sub-dict never does. Same op (scatter COO -> grid), scope='sample'
        # either way; place it after ToDevice for the on-GPU path.
        pfx = f'{self.modality}_' if self.modality is not None else ''
        if (pfx + self.offset_key) in data:
            return self._densify_batch(data)
        sub = data
        if sub.get('readout_type') == 'pixel':
            if self.on_pixel == 'raise':
                raise ValueError(
                    "Densify supports wire readout only; got pixel. Pass "
                    "on_pixel='skip' to no-op on pixel streams.")
            return sub

        raw = sub.get('raw', {})
        shapes = sub.get('shape', {})
        out = {}
        for label, cols in raw.items():
            wire_raw = np.asarray(cols['wire'])
            time_raw = np.asarray(cols['time'])
            if not (np.issubdtype(wire_raw.dtype, np.integer)
                    and np.issubdtype(time_raw.dtype, np.integer)):
                raise TypeError(
                    f"Densify: plane {label!r} wire/time must be integer grid "
                    f"indices (got {wire_raw.dtype}/{time_raw.dtype}); densify "
                    "reads the immutable raw COO, never a mutated float coord.")
            wire = wire_raw.astype(np.intp, copy=False)
            time = time_raw.astype(np.intp, copy=False)
            val = np.asarray(cols['value']).astype(self.dtype, copy=False)

            if label in shapes:
                n_wires, n_ticks = int(shapes[label][0]), int(shapes[label][1])
            elif self.require_shape:
                raise KeyError(
                    f"Densify: no shape for plane {label!r}. Surface "
                    "n_wires/num_time_steps from the sensor reader, or pass "
                    "require_shape=False to infer extents from the data.")
            else:
                n_wires = int(wire.max()) + 1 if wire.size else 0
                n_ticks = int(time.max()) + 1 if time.size else 0

            img = np.full((n_wires, n_ticks), self.fill, dtype=self.dtype)
            if wire.size:
                if self.assert_unique:
                    # production sparse COO is duplicate-free; a duplicate
                    # (wire,time) makes the scatter ambiguous (assignment last-wins
                    # vs index_add_ sum diverge), so catch it rather than silently
                    # pick a convention.
                    flat = wire.astype(np.int64) * n_ticks + time.astype(np.int64)
                    if np.unique(flat).size != flat.size:
                        raise ValueError(
                            f"Densify: duplicate (wire,time) in plane {label!r} "
                            "— sparse COO must be unique per cell.")
                img[wire, time] = val
            out[label] = img

        sub[self.dense_key] = out
        return sub

    def _densify_batch(self, batch):
        """Post-collate path: scatter the flat collated COO (``<mod>_wire/.../offset``)
        into per-plane grids ``<mod>_dense = {gid: (B, W, T)}`` via the torch scatter
        in :mod:`dense_ops` — born on whatever device the inputs are on (GPU after
        ToDevice). Same per-event-independent op as the sub-dict path."""
        from . import dense_ops
        pfx = f'{self.modality}_' if self.modality is not None else ''
        m = self.modality if self.modality is not None else '<bare batch>'
        wire = batch[pfx + self.wire_key]
        off = batch[pfx + self.offset_key]
        n = wire.reshape(-1).shape[0]
        if off.numel() and int(off[-1]) != n:
            raise ValueError(
                f"Densify({m!r}): offset total {int(off[-1])} != {n} raw-COO rows "
                f"({pfx + self.wire_key!r}). A coord-mutating transform "
                "(GridSample/SphereCrop/RandomDropout) ran on this part pre-collate "
                "— densify needs the raw COO aligned with offset. Voxelize a SEPARATE "
                "sparse view, not the one you densify.")
        batch[pfx + self.dense_key] = dense_ops.densify(
            wire, batch[pfx + self.time_key], batch[pfx + self.value_key],
            batch[pfx + self.plane_key], off, self.geom)
        return batch


@TRANSFORMS.register_module()
class AddNoise:
    """Inject forward detector noise on dense sensor planes (after ``Densify``).

    Thin adapter over :func:`pimm_data.noise.add_noise`. Operates on a ``sensor``
    sub-dict that already carries ``dense`` (so it must run *after* ``Densify``):
    each plane image is perturbed in place by incoherent and/or per-group
    coherent noise. Tags (``incoherent`` / ``coherent`` and the parameters) are
    set from the config.

    Reproducibility: the per-event RNG is derived from a stable event id
    (``sub['name']``, surfaced by the dataset) hashed with ``base_seed`` — so the
    same event gets the same noise every epoch regardless of which DataLoader
    worker happens to draw it, without touching numpy's global RNG state.

    Default tags are ``incoherent=False, coherent=True``: JAXTPC output already
    carries incoherent noise, so the common load-time use is adding the coherent
    component it omits. Set ``incoherent=True`` (with ``wire_lengths_m``) to add
    incoherent noise to noise-free input.
    """

    scope = 'sample'  # per-event independent; placeable pre- OR post-collate

    def __init__(self, incoherent=False, coherent=True, group_size=64,
                 wire_lengths_m=None, enc=DEFAULT_ENC,
                 series_spectrum=DEFAULT_SERIES_SPECTRUM,
                 sampling_rate_hz=DEFAULT_SAMPLING_RATE_HZ, coh_rms=2.5,
                 coh_corner_freq_hz=20000.0, coh_spectral_slope=1.5, beta=0.15,
                 base_seed=0, dense_key='dense', planes=None, name_key='name',
                 geom=None, modality=None, coherent_numpy=False, offset_key='offset'):
        self.incoherent = bool(incoherent)
        # flat collated-batch path (post-collate, torch): geom + flat dense key.
        self.geom = geom
        self.modality = modality
        self.coherent_numpy = bool(coherent_numpy)
        self.offset_key = offset_key
        self.coherent = bool(coherent)
        self.group_size = int(group_size)
        self.wire_lengths_m = wire_lengths_m
        self.enc = tuple(enc)
        self.series_spectrum = series_spectrum
        self.sampling_rate_hz = float(sampling_rate_hz)
        self.coh_rms = float(coh_rms)
        self.coh_corner_freq_hz = float(coh_corner_freq_hz)
        self.coh_spectral_slope = float(coh_spectral_slope)
        self.beta = float(beta)
        self.base_seed = int(base_seed)
        self.dense_key = dense_key
        self.planes = planes
        self.name_key = name_key

    def _event_rng(self, name):
        h = hashlib.blake2b(str(name).encode('utf-8'), digest_size=8).digest()
        seed = (int.from_bytes(h, 'little') ^ self.base_seed) & ((1 << 64) - 1)
        return np.random.default_rng(seed)

    def __call__(self, data):
        # DISPATCH: a collated batch carries the flat <mod>_offset key (torch grids
        # {gid:(B,W,T)}); a per-event sub-dict does not (numpy grids {gid:(W,T)}).
        pfx = f'{self.modality}_' if self.modality is not None else ''
        if (pfx + self.offset_key) in data:
            return self._noise_batch(data, pfx)
        sub = data
        if sub.get('readout_type') == 'pixel':
            return sub
        dense = sub.get(self.dense_key)
        if dense is None:
            raise KeyError(
                f"AddNoise: no {self.dense_key!r} in the sensor modality — run "
                "Densify before AddNoise.")
        rng = self._event_rng(sub.get(self.name_key, ''))
        labels = self.planes if self.planes is not None else list(dense)
        for label in labels:
            if label not in dense:
                continue
            img = dense[label]
            noise = generate_noise(
                img.shape, rng=rng, wire_lengths_m=self.wire_lengths_m,
                incoherent=self.incoherent, coherent=self.coherent,
                enc=self.enc, series_spectrum=self.series_spectrum,
                sampling_rate_hz=self.sampling_rate_hz,
                group_size=self.group_size, coh_rms=self.coh_rms,
                coh_corner_freq_hz=self.coh_corner_freq_hz,
                coh_spectral_slope=self.coh_spectral_slope, beta=self.beta)
            dense[label] = img + noise
        return sub

    def _noise_batch(self, batch, pfx):
        """Post-collate path: add fresh per-event noise to the collated grids
        ``batch[<mod>_dense]`` (torch, on the inputs' device). Seeds self-derive
        from ``batch['name']`` folded with base_seed/_epoch/_rank."""
        from . import dense_ops
        from .batch_transforms import _seeds_for, content_seed
        grids = batch[pfx + self.dense_key]
        seeds = _seeds_for(batch, self.base_seed)
        if seeds is None:                              # no 'name' -> position fallback
            seeds = [content_seed(f"_idx{i}", self.base_seed)
                     for i in range(next(iter(grids.values())).shape[0])]
        dense_ops.add_intrinsic_noise(
            grids, self.geom, seeds=seeds, enc=self.enc,
            coherent=self.coherent, incoherent=self.incoherent,
            sampling_rate_hz=self.sampling_rate_hz, group_size=self.group_size,
            coh_rms=self.coh_rms, coh_corner_freq_hz=self.coh_corner_freq_hz,
            coh_spectral_slope=self.coh_spectral_slope, beta=self.beta,
            series_spectrum=self.series_spectrum, coherent_numpy=self.coherent_numpy)
        return batch


@TRANSFORMS.register_module()
class Digitize:
    """Quantize dense sensor planes to integer ADC codes (production digitize).

    Per plane: ``round(img*gain + pedestal).clip(0, adc_max) - pedestal`` — the
    pedestal-subtracted output of JAXTPC's ``_digitize_signal`` / the doraemon
    ``make_noisy`` path. Run it LAST in the dense chain
    (``Densify -> AddNoise -> Digitize``), so the analog ``signal + noise`` is
    quantized exactly as the detector would.

    Pedestal resolution (raw-ADC offset that sets where 0 and saturation fall):
    the ``pedestal`` arg wins (a scalar applied to every plane, or a
    ``{plane_label: pedestal}`` dict), else the per-plane pedestal surfaced by
    the reader (``sub['pedestal']``), else ``0`` — or raise when
    ``require_pedestal=True``. ``adc_max`` defaults to ``(1 << n_bits) - 1``
    (12-bit → 4095). Wire readout only (pixel streams pass through unchanged).

    Parameters
    ----------
    n_bits : int
        Code depth; sets ``adc_max = (1 << n_bits) - 1`` when ``adc_max`` is None.
    adc_max : float or None
        Explicit max code (overrides ``n_bits``).
    gain : float
        Scale applied before adding the pedestal (1.0 = input already in ADC).
    pedestal : float, dict, or None
        Pedestal override (scalar or per-plane). None → use ``sub['pedestal']``.
    require_pedestal : bool
        Raise if a plane has no resolvable pedestal (instead of defaulting to 0).
    dense_key : str
        Which dense field to quantize. Default ``'dense'``.
    planes : list or None
        Restrict to these plane labels (None → all dense planes).
    """

    scope = 'sample'  # per-event independent; placeable pre- OR post-collate

    def __init__(self, n_bits=12, adc_max=None, gain=1.0, pedestal=None,
                 require_pedestal=False, dense_key='dense', planes=None,
                 geom=None, modality=None, offset_key='offset'):
        self.n_bits = int(n_bits)
        self.adc_max = adc_max
        self.gain = float(gain)
        self.pedestal = pedestal
        self.require_pedestal = bool(require_pedestal)
        self.dense_key = dense_key
        self.planes = planes
        self.geom = geom or {}
        self.modality = modality
        self.offset_key = offset_key

    def _pedestal_for(self, label, sub):
        if isinstance(self.pedestal, dict):
            if label in self.pedestal:
                return float(self.pedestal[label])
        elif self.pedestal is not None:
            return float(self.pedestal)
        ped_map = sub.get('pedestal', {})
        if label in ped_map:
            return float(ped_map[label])
        if self.require_pedestal:
            raise KeyError(
                f"Digitize: no pedestal for plane {label!r}; surface it from "
                "the reader (sensor file pedestal attr) or pass pedestal=.")
        return 0.0

    def __call__(self, data):
        # DISPATCH: collated batch (flat <mod>_offset present) -> torch path;
        # per-event sub-dict otherwise.
        pfx = f'{self.modality}_' if self.modality is not None else ''
        if (pfx + self.offset_key) in data:
            return self._digitize_batch(data, pfx)
        sub = data
        if sub.get('readout_type') == 'pixel':
            return sub
        dense = sub.get(self.dense_key)
        if dense is None:
            raise KeyError(
                f"Digitize: no {self.dense_key!r} in the sensor modality — run "
                "Densify (and AddNoise) before Digitize.")
        marker = f'_digitized_{self.dense_key}'
        if sub.get(marker):
            raise RuntimeError(
                f"Digitize: {self.dense_key!r} already digitized — digitize is "
                "not idempotent (pedestal/gain round-trip); run it at most once.")
        labels = self.planes if self.planes is not None else list(dense)
        for label in labels:
            if label not in dense:
                continue
            ped = self._pedestal_for(label, sub)
            dense[label] = digitize(dense[label], ped, n_bits=self.n_bits,
                                    adc_max=self.adc_max, gain=self.gain)
        sub[marker] = True
        return sub

    def _digitize_batch(self, batch, pfx):
        """Post-collate path: quantize the collated grids ``batch[<mod>_dense]``
        (torch, on the inputs' device) via dense_ops. Pedestal: the ``pedestal``
        arg wins, else per-plane from ``geom``."""
        from . import dense_ops
        ped = self.pedestal
        if ped is None:
            ped = {gid: e.get('pedestal', 0) for gid, e in self.geom.items()}
        batch[pfx + self.dense_key] = dense_ops.digitize(
            batch[pfx + self.dense_key], ped, n_bits=self.n_bits,
            adc_max=self.adc_max, gain=self.gain)
        return batch
