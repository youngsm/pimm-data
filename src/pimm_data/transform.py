"""
3D point cloud augmentation

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

from __future__ import annotations

import random
import scipy
import scipy.ndimage
import scipy.interpolate
import scipy.stats
from scipy.spatial import cKDTree
import numpy as np
import torch
import copy
from collections.abc import Sequence, Mapping

from ._registry import Registry

TRANSFORMS = Registry("transforms")


# Generic multi-task label schema (D22/D25): any key named segment_*/
# instance_*/target_* is a per-point label axis, plus these per-point FK
# columns. They must survive N-changing transforms (GridSample, dropout, …)
# rather than being silently dropped or length-mismatched.
_INDEX_SCHEMA_PREFIXES = ("segment", "instance", "target")
_INDEX_PER_POINT_KEYS = ("particle_idx", "sensor_idx", "plane_id")

# Keys index_operator always carries through a point subsample. The schema
# axes (segment_*/instance_*/target_*) and per-point FK columns are NOT listed
# here — _augment_index_valid_keys adds whichever are present, per-point-guarded
# (so a per-event target_* is not subset). One source of truth for the schema
# axes, instead of literal ∪ predicate with a hidden overlap.
_DEFAULT_INDEX_VALID_KEYS = [
    "coord", "color", "normal", "strength", "energy", "local_shape",
    "momentum",
    # JAXTPCDataset physics columns
    "track_ids", "group_ids", "pdg", "volume_id", "interaction_ids",
    "ancestor_track_ids", "charge", "photons", "qs_fractions", "t0_us",
    "dx", "theta", "phi",
    "vertex",
]


def _augment_index_valid_keys(data_dict):
    """Append present schema/FK keys to ``index_valid_keys`` (D25).

    Underscore-boundary prefix match on ``segment_*``/``instance_*``/
    ``target_*`` plus the per-point FK columns. A key qualifies only if its
    leading dim equals the point count — so per-event ``target_*`` (e.g. a
    length-3 ``target_vertex``) is NOT point-subset.
    """
    coord = data_dict.get("coord")
    if coord is None or not hasattr(coord, "shape"):
        return
    n_points = coord.shape[0]
    valid = data_dict["index_valid_keys"]
    present = set(valid)
    for k in list(data_dict.keys()):
        if k in present or k == "index_valid_keys":
            continue
        is_schema = any(k == p or k.startswith(p + "_")
                        for p in _INDEX_SCHEMA_PREFIXES)
        if not (is_schema or k in _INDEX_PER_POINT_KEYS):
            continue
        v = data_dict[k]
        if getattr(v, "shape", None) and v.shape[0] == n_points:
            valid.append(k)
            present.add(k)


def _valid_vertex_mask(data_dict):
    vertex = data_dict.get("vertex")
    if vertex is None:
        return None
    if vertex.ndim != 2 or vertex.shape[1] != 3:
        return None
    # v3 uses (-1, -1, -1) as a missing-vertex sentinel. v2 vertices are
    # corrected labels and should transform with the coordinates.
    return ~(vertex == -1).all(axis=1)


def _apply_to_v3_vertex(data_dict, transform):
    """Co-transform a per-event/per-point ``vertex`` label with the geometric
    op applied to ``coord`` (PILArNet v3 / panda). No-op when absent."""
    mask = _valid_vertex_mask(data_dict)
    if mask is None or not mask.any():
        return
    data_dict["vertex"][mask] = transform(data_dict["vertex"][mask])


def _translate_axis(points, dim, value):
    points = points.copy()
    points[:, dim] += value
    return points


def _negate_axis(points, dim):
    points = points.copy()
    points[:, dim] = -points[:, dim]
    return points


_ROT_AXIS_DIM = {"x": 0, "y": 1, "z": 2}


def _rotation_matrix(axis, angle):
    """3×3 rotation about a principal ``axis`` (radians)."""
    c, s = np.cos(angle), np.sin(angle)
    if axis == "x":
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    if axis == "y":
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    if axis == "z":
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    raise NotImplementedError


def _rotate_about_center(data_dict, rot_t, center=None):
    """Rotate ``coord`` (+ co-transform ``vertex`` and ``normal``) by ``rot_t``
    about ``center`` (bbox midpoint when None). Shared by the rotate transforms."""
    if "coord" in data_dict:
        if center is None:
            center = (data_dict["coord"].min(axis=0)
                      + data_dict["coord"].max(axis=0)) / 2
        center = np.asarray(center)
        data_dict["coord"] = (data_dict["coord"] - center) @ rot_t.T + center
        _apply_to_v3_vertex(
            data_dict, lambda v: (v - center) @ rot_t.T + center)
    if "normal" in data_dict:
        data_dict["normal"] = data_dict["normal"] @ rot_t.T
    return data_dict


def _remap_self_edges(data_dict, index):
    """For each ``('edge','self')`` key in ``data_dict['_roles']``, keep edges whose
    BOTH endpoints survive ``index`` and reindex to the new row positions."""
    roles = data_dict.get("_roles")
    if not roles or "coord" not in data_dict:
        return
    edge_keys = [k for k, r in roles.items()
                 if isinstance(r, (tuple, list)) and len(r) == 2
                 and r[0] == "edge" and r[1] == "self" and k in data_dict]
    if not edge_keys:
        return
    n_old = data_dict["coord"].shape[0]
    idx = np.asarray(index)
    kept = np.where(idx)[0] if idx.dtype == bool else idx.astype("int64")
    old2new = np.full(int(n_old), -1, dtype="int64")
    old2new[kept] = np.arange(len(kept))
    for k in edge_keys:
        e = data_dict[k]
        is_torch = isinstance(e, torch.Tensor)
        ea = e.cpu().numpy() if is_torch else np.asarray(e)
        keep = (old2new[ea[0]] >= 0) & (old2new[ea[1]] >= 0)
        remapped = np.stack([old2new[ea[0][keep]], old2new[ea[1][keep]]])
        data_dict[k] = torch.as_tensor(remapped) if is_torch else remapped


def index_operator(data_dict, index, duplicate=False):
    # index selection operator for keys in "index_valid_keys"
    # custom these keys by "Update" transform in config
    if "index_valid_keys" not in data_dict:
        data_dict["index_valid_keys"] = list(_DEFAULT_INDEX_VALID_KEYS)
    # PILArNet v3 carries a per-point primary-particle flag alongside the
    # cluster vertex; keep it aligned through subsampling/cropping.
    if data_dict.get("revision") == "v3" and "is_primary" in data_dict \
            and "is_primary" not in data_dict["index_valid_keys"]:
        data_dict["index_valid_keys"].append("is_primary")
    # plane_id (per-point sparse plane label) is carried via _INDEX_PER_POINT_KEYS;
    # wire/time/value/plane_gid (dense-path scatter inputs) are intentionally not
    # subset — densify consumes them as immutable raw COO.
    _augment_index_valid_keys(data_dict)
    # roles-aware edge remap (REDESIGN §4): a self-edge key indexes this part's rows,
    # so subsampling must DROP edges to removed rows and REINDEX, not slice. Makes
    # "subsample then build graph" order-independent. Cross-store edges index other
    # parts (producer-only, built last) and are left untouched.
    _remap_self_edges(data_dict, index)
    if not duplicate:
        for key in data_dict["index_valid_keys"]:
            if key in data_dict:
                data_dict[key] = data_dict[key][index]
        return data_dict
    else:
        data_dict_ = dict()
        for key in data_dict.keys():
            if key in data_dict["index_valid_keys"]:
                data_dict_[key] = data_dict[key][index]
            else:
                data_dict_[key] = data_dict[key]
        return data_dict_


@TRANSFORMS.register_module()
class Collect(object):
    """Pack a flat batch dict for the model from possibly-nested input.

    The batch unit is a **part** — an offset-bearing key group (a modality, an
    SSL view, a graph store, a second row-space), matching the ``_roles`` /
    ``Apply(on=)`` vocabulary. Two forms:

    * **single (bare):** ``part=`` scopes the source to ``data_dict[part]`` and
      emits BARE keys (``coord``, ``segment``, ``feat``, …) — what single-cloud
      models read::

          Collect(part='step', keys=['coord', 'segment'], feat_keys=['coord', 'energy'])

    * **multi (flat-prefixed):** ``parts={name: spec}`` emits FLAT keys
      ``<name>_<key>`` (``step_coord``, …) + a ``_roles`` map, one self-contained
      part per name.

    Extracted numpy arrays are converted to torch tensors automatically.
    This enables zero-copy IPC when used with ``DataLoader(num_workers>0)``
    — PyTorch transfers tensors via file-descriptor sharing (~400 bytes)
    instead of pickling the full array.
    """

    def __init__(self, keys=None, offset_keys_dict=None, part=None, parts=None,
                 **kwargs):
        # `part=`/`parts=` are the ONLY names. The batch unit is a *part*: an
        # offset-bearing key group (a modality, an SSL view, a graph store, a
        # second row-space), matching the `_roles`/`Apply(on=)` vocabulary.
        #
        # `modality=`/`modalities=` were accepted as aliases. They are not, now:
        # `modalities=` is ALSO a live dataset-constructor argument with a
        # different meaning (which modalities to READ), so the same word named
        # two things one config level apart. **kwargs would swallow them
        # silently, so reject them by name.
        for _dead, _use in (("modality", "part"), ("modalities", "parts")):
            if _dead in kwargs:
                raise TypeError(
                    f"Collect({_dead}=...) was removed; use {_use}=. "
                    f"(`modalities=` on a DATASET is unrelated and unchanged — "
                    f"it selects which modalities to read.)")
        # Two forms (mutually exclusive):
        #  * single  — keys=/part=/feat *_keys → a flat (bare) dict, scoped to
        #    data_dict[part] (or the top level). Byte-identical to before.
        #  * multi   — parts={name: spec} → a NAMESPACED dict {name:{...}, …}, one
        #    self-contained sub-dict per part, each with its OWN offset.
        #    spec = dict(keys=…, offset_keys_dict=…, <feat>_keys=…). collate_fn
        #    packs the nesting (Mapping recursion + per-sub-dict offset cumsum).
        if parts is not None:
            assert keys is None and part is None and not kwargs, (
                "Collect: pass EITHER (keys=/part=/feat_keys=) for one part "
                "OR parts={...} for the namespaced multi form — not both.")
            self.parts_spec = dict(parts)
            # build-time validation (REDESIGN §9): fail at construction, not epoch 1.
            for pname, spec in self.parts_spec.items():
                if 'keys' not in spec:
                    raise ValueError(
                        f"Collect: part {pname!r} spec needs 'keys=' (got {sorted(spec)}).")
                bad = set(dict(spec.get('roles', {}))) - set(spec['keys'])
                if bad:
                    raise ValueError(
                        f"Collect: part {pname!r} declares roles for non-collected "
                        f"keys {sorted(bad)} — add them to keys= or drop the role.")
        else:
            assert keys is not None, "Collect requires keys= (or parts=)."
            self.parts_spec = None
            if offset_keys_dict is None:
                offset_keys_dict = dict(offset="coord")
            self.keys = keys
            self.offset_keys = offset_keys_dict
            self.part = part
            self.kwargs = kwargs

    @staticmethod
    def _to_tensor(v):
        if isinstance(v, torch.Tensor):
            return v
        if isinstance(v, np.ndarray):
            return torch.from_numpy(v)
        return v

    def _project(self, source, keys, offset_keys, feat_specs):
        """Project one modality sub-dict → flat {key: tensor}: the listed keys
        (tensorized), the derived offset(s), and concatenated feature tensors.
        Everything not listed (e.g. ragged ``raw``) is dropped — which is what
        keeps the projected dict collate-safe."""
        data = dict()
        keys = [keys] if isinstance(keys, str) else keys
        for key in keys:
            data[key] = self._to_tensor(source[key])
        for okey, src_key in offset_keys.items():
            data[okey] = torch.tensor([source[src_key].shape[0]])
        for name, feat_keys in feat_specs.items():
            name = name.removesuffix("_keys") if name.endswith("_keys") else name
            assert isinstance(feat_keys, Sequence) and not isinstance(feat_keys, str)
            data[name] = torch.cat(
                [self._to_tensor(source[key]).float() for key in feat_keys], dim=1)
        return data

    _OFFSET_DEFAULT = object()   # sentinel: distinguish "not given" from explicit {}

    @staticmethod
    def _parse_spec(spec):
        spec = dict(spec)
        keys = spec.pop('keys')
        # explicit `offset_keys_dict={}` SUPPRESSES the derived offset (the part
        # already carries its own, e.g. MultiCrop's per-crop packed offset); only a
        # MISSING key falls back to the default. (Don't use `or` — {} is falsy.)
        ok = spec.pop('offset_keys_dict', Collect._OFFSET_DEFAULT)
        offset_keys = dict(offset="coord") if ok is Collect._OFFSET_DEFAULT else dict(ok)
        roles = dict(spec.pop('roles', {}))   # {key: role spec}, REDESIGN §3
        return keys, offset_keys, roles, spec  # remaining entries are <feat>_keys specs

    def __call__(self, data_dict):
        if self.parts_spec is not None:
            # REDESIGN: namespaced form emits FLAT underscore-prefixed keys
            # (`step_coord`, `step_offset`, `step_feat`) + a `_roles` map, NOT a
            # nested `{step:{…}}` dict. collate routes flat+_roles to the role-driven
            # reduce. Single-part form (below) stays bare/byte-identical.
            out = dict()
            roles = dict()
            for pname, spec in self.parts_spec.items():
                if not isinstance(data_dict.get(pname), dict):
                    avail = [k for k in data_dict if isinstance(data_dict.get(k), dict)]
                    raise KeyError(
                        f"Collect(parts=): part {pname!r} absent or not a "
                        f"sub-dict; available: {avail}")
                keys, offset_keys, part_roles, feat_specs = self._parse_spec(spec)
                projected = self._project(data_dict[pname], keys, offset_keys, feat_specs)
                for k, v in projected.items():
                    out[f'{pname}_{k}'] = v
                # roles: producer-stamped (sub-dict '_roles') first, then spec roles=
                for k, r in data_dict[pname].get('_roles', {}).items():
                    if k in keys:                     # only for keys actually collected
                        roles[f'{pname}_{k}'] = r
                for k, r in part_roles.items():       # explicit spec roles= win
                    roles[f'{pname}_{k}'] = r
            out['_roles'] = roles                     # always present -> flat-roles collate
        else:
            source = data_dict[self.part] if self.part is not None else data_dict
            out = self._project(source, self.keys, self.offset_keys, self.kwargs)
        # G2: name/split are the cross-boundary identity carriers (content-
        # addressed seeding reads top-level ``batch['name']``) — pass them
        # through UNCONDITIONALLY, at the top level, for both forms.
        # event_label/config_id/weight are MultiModalEventDataset's event-level
        # label/identity carriers — also passed through so event-classification
        # batches keep their target (collate stacks them to (B,) via event role).
        for passthrough in ("name", "split", "event_label", "config_id", "weight"):
            if passthrough in data_dict and passthrough not in out:
                out[passthrough] = data_dict[passthrough]
        return out


@TRANSFORMS.register_module()
class Copy(object):
    def __init__(self, keys_dict=None):
        if keys_dict is None:
            keys_dict = dict(coord="origin_coord", segment="origin_segment")
        self.keys_dict = keys_dict

    def __call__(self, data_dict):
        for key, value in self.keys_dict.items():
            if isinstance(data_dict[key], np.ndarray):
                data_dict[value] = data_dict[key].copy()
            elif isinstance(data_dict[key], torch.Tensor):
                data_dict[value] = data_dict[key].clone().detach()
            else:
                data_dict[value] = copy.deepcopy(data_dict[key])
        return data_dict


@TRANSFORMS.register_module()
class Update(object):
    def __init__(self, keys_dict=None):
        if keys_dict is None:
            keys_dict = dict()
        self.keys_dict = keys_dict

    def __call__(self, data_dict):
        for key, value in self.keys_dict.items():
            data_dict[key] = value
        return data_dict


@TRANSFORMS.register_module()
class ToTensor(object):
    def __call__(self, data):
        if isinstance(data, torch.Tensor):
            return data
        elif isinstance(data, str):
            # note that str is also a kind of sequence, judgement should before sequence
            return data
        elif isinstance(data, int):
            return torch.LongTensor([data])
        elif isinstance(data, float):
            return torch.FloatTensor([data])
        elif isinstance(data, np.ndarray) and np.issubdtype(data.dtype, bool):
            return torch.from_numpy(data)
        elif isinstance(data, np.ndarray) and np.issubdtype(data.dtype, np.integer):
            return torch.from_numpy(data).long()
        elif isinstance(data, np.ndarray) and np.issubdtype(data.dtype, np.floating):
            return torch.from_numpy(data).float()
        elif isinstance(data, Mapping):
            result = {sub_key: self(item) for sub_key, item in data.items()}
            return result
        elif isinstance(data, Sequence):
            result = [self(item) for item in data]
            return result
        else:
            raise TypeError(f"type {type(data)} cannot be converted to tensor.")


@TRANSFORMS.register_module()
class NormalizeCoord(object):
    def __init__(self, center=None, scale=None):
        self.center = center
        self.scale = scale

    def __call__(self, data_dict):
        if "coord" in data_dict.keys():
            # modified from pointnet2
            if self.center is None:
                centroid = np.mean(data_dict["coord"], axis=0)
            else:
                centroid = np.array(self.center)
            data_dict["coord"] -= centroid

            if self.scale is None:
                m = np.max(np.sqrt(np.sum(data_dict["coord"] ** 2, axis=1)))
                scale = m
            else:
                scale = self.scale
            data_dict["coord"] = data_dict["coord"] / scale
            _apply_to_v3_vertex(
                data_dict, lambda vertex: (vertex - centroid) / scale)
        return data_dict


@TRANSFORMS.register_module()
class LogTransform(object):
    """Map an energy-like key to ``[-1, 1]`` (log or linear).

    **Domain / clip (F9, D11).** The map normalizes ``[min_val, max_val]`` onto
    ``[-1, 1]``; with ``clip=False`` (default) a value *above* ``max_val`` maps
    *beyond* ``+1`` rather than saturating. On real WAND PE this bites — PE
    reaches ~424 and ~0.1% of hits exceed the ``max_val=50`` default, landing
    near ``+1.5`` outside the model's expected range. Pass ``clip=True`` to
    clamp into-domain first (those points saturate at ``+1``), or raise
    ``max_val``. Left default-off to preserve historical behavior; choose
    deliberately per task.
    """

    def __init__(self, min_val=1.0e-2, max_val=20.0, log=True, keys=("energy",),
                 clip=False):
        self.min_val = min_val
        self.max_val = max_val
        self.log = log
        self.clip = clip
        if not isinstance(keys, tuple):
            keys = (keys,)
        self.keys = keys

    def log_transform(self, x):
        """Transform energy to logarithmic scale on [-1,1]"""
        if self.clip:
            x = np.clip(x, 0.0, self.max_val)
        # [emin, emax] -> [-1,1]. Lower-clamp ALWAYS (independent of the upper
        # `clip`): non-positive energy (zero/negative pixel charge) would give
        # log10(<=0) -> -inf/nan and silently poison the model.
        y0 = np.log10(self.min_val)
        y1 = np.log10(self.max_val + self.min_val)
        return 2 * (np.log10(np.maximum(x, 0.0) + self.min_val) - y0) / (y1 - y0) - 1

    def linear_transform(self, x):
        """Transform energy to linear scale on [-1,1]"""
        if self.clip:
            x = np.clip(x, self.min_val, self.max_val)
        return 2 * (x - self.min_val) / (self.max_val - self.min_val) - 1

    def __call__(self, data_dict):
        for k in self.keys:
            if k in data_dict.keys():
                data_dict[k] = (
                    self.log_transform(data_dict[k])
                    if self.log
                    else self.linear_transform(data_dict[k])
                )
            else:
                raise ValueError(f"Key {k} not found in data_dict")
        return data_dict


@TRANSFORMS.register_module()
class RelativeLogNormalize(object):
    """Per-array log normalization relative to the array minimum.

    Subtracts the per-array min (so the domain is non-negative — handles
    negative times, the LUCiD case), clips to ``max_val``, applies
    ``log1p(x/scale)`` normalized to ``[out_min, out_max]``. Value-only
    (point count unchanged). Non-idempotent / order-sensitive: run before any
    transform that reorders/subsets the target array.

    **Deliberate ceiling (D13).** ``max_val`` is a *hard* clip: every value
    with ``x - min(x) >= max_val`` saturates to ``out_max`` and becomes
    indistinguishable. The default ``4000`` is a model choice for the PMT
    hit-time axis — the late-time tail (reflections, scattered / afterpulse
    light) is intentionally compressed into the top bin rather than dilating
    the informative early-time dynamic range. This is lossy by design: if a
    downstream task needs to resolve the long tail, raise ``max_val`` (and
    re-tune ``scale``) rather than assuming the tail survived.
    """

    def __init__(self, keys=("time",), scale=50.0, max_val=4000.0,
                 out_min=-1.0, out_max=1.0):
        self.scale = float(scale)
        self.max_val = float(max_val)
        self.out_min = float(out_min)
        self.out_max = float(out_max)
        if self.scale <= 0:
            raise ValueError("scale must be positive")
        if self.max_val <= 0:
            raise ValueError("max_val must be positive")
        if self.out_max <= self.out_min:
            raise ValueError("out_max must be greater than out_min")
        if not isinstance(keys, tuple):
            keys = (keys,)
        self.keys = keys
        self.denom = np.log1p(self.max_val / self.scale)

    def relative_log_transform(self, x):
        x = np.asarray(x, dtype=np.float32)
        x = x - np.min(x)
        x = np.clip(x, 0.0, self.max_val)
        y = np.log1p(x / self.scale) / self.denom
        y = self.out_min + y * (self.out_max - self.out_min)
        return np.clip(y, self.out_min, self.out_max).astype(
            np.float32, copy=False)

    def __call__(self, data_dict):
        for k in self.keys:
            if k in data_dict.keys():
                data_dict[k] = self.relative_log_transform(data_dict[k])
            else:
                raise ValueError(f"Key {k} not found in data_dict")
        return data_dict

@TRANSFORMS.register_module()
class MomentumTransform(object):
    def __init__(self, keys=("momentum",)):
        self.keys = keys

    def __call__(self, data_dict):
        for k in self.keys:
            if k in data_dict.keys():
                mom = data_dict[k]
                mom = np.where(mom > 0, np.log10(np.clip(mom, 1e-6, None)), mom)
                data_dict[k] = mom
        return data_dict


@TRANSFORMS.register_module()
class PositiveShift(object):
    def __call__(self, data_dict):
        if "coord" in data_dict.keys():
            coord_min = np.min(data_dict["coord"], 0)
            data_dict["coord"] -= coord_min
            _apply_to_v3_vertex(data_dict, lambda vertex: vertex - coord_min)
        return data_dict


@TRANSFORMS.register_module()
class CenterShift(object):
    def __init__(self, apply_z=True, axes=("x", "y", "z")):
        self.apply_z = apply_z
        if not isinstance(axes, tuple):
            axes = (axes,)
        self.axes = axes
        
    def __call__(self, data_dict):
        if "coord" in data_dict.keys():
            # Each axis's center depends only on that axis's column (untouched
            # by the other axes' shifts), so one min/max pass is equivalent to
            # the old per-branch recompute.
            centers = (data_dict["coord"].min(axis=0)
                       + data_dict["coord"].max(axis=0)) / 2
            for dim, ax in enumerate(("x", "y", "z")):
                if ax not in self.axes:
                    continue
                shift = centers[dim]
                data_dict["coord"][:, dim] -= shift
                _apply_to_v3_vertex(
                    data_dict,
                    lambda vertex, dim=dim, shift=shift: _translate_axis(
                        vertex, dim, -shift))
        return data_dict

@TRANSFORMS.register_module()
class ConditionalRandomTransform(object):
    _max_value_pilarnet = 2 * pow(3, 0.5) / 3 # (768) / (768 * 3 ** 0.5 / 2)
    def __init__(self, p=0.5, axes=("x", "y", "z"), buffer_size=0.05, bounds=((-1, 1), (-1, 1), (-1, 1))):
        self.p = p
        if not isinstance(axes, tuple):
            axes = (axes,)
        self.axes = axes
        self.buffer_size = buffer_size
        self.bounds = bounds

    def __call__(self, data_dict):
        if "coord" not in data_dict.keys():
            return data_dict
        coord = data_dict["coord"]

        for dim, axis in enumerate(("x", "y", "z")):
            if axis not in self.axes:
                continue

            bounds = self.bounds[dim]
            min_val = np.min(coord[:, dim])
            max_val = np.max(coord[:, dim])

            # skip if entirely interior (not near any wall)
            if (min_val >= bounds[0] + self.buffer_size) and (max_val <= bounds[1] - self.buffer_size):
                continue

            if random.random() <= self.p:
                lower, upper = bounds
                near_lower = min_val <= lower + self.buffer_size
                near_upper = max_val >= upper - self.buffer_size

                # base feasibility to keep all points within bounds
                t_low = lower - min_val
                t_high = upper - max_val

                if near_lower and not near_upper:
                    # keep near lower wall
                    t_high = min(t_high, (lower + self.buffer_size) - min_val)
                elif near_upper and not near_lower:
                    # keep near upper wall
                    t_low = max(t_low, (upper - self.buffer_size) - max_val)
                elif near_lower and near_upper:
                    # keep near both walls
                    t_low = max(t_low, (upper - self.buffer_size) - max_val)
                    t_high = min(t_high, (lower + self.buffer_size) - min_val)
                else:
                    # interior (should have been caught above)
                    continue

                if t_low <= t_high:
                    translation = np.random.uniform(t_low, t_high)
                    coord[:, dim] += translation
                    _apply_to_v3_vertex(
                        data_dict,
                        lambda vertex, dim=dim, translation=translation:
                            _translate_axis(vertex, dim, translation))

        data_dict["coord"] = coord
        return data_dict

@TRANSFORMS.register_module()
class RandomShift(object):
    def __init__(self, shift=((-0.2, 0.2), (-0.2, 0.2), (0, 0))):
        self.shift = shift

    def __call__(self, data_dict):
        if "coord" in data_dict.keys():
            shift_x = np.random.uniform(self.shift[0][0], self.shift[0][1])
            shift_y = np.random.uniform(self.shift[1][0], self.shift[1][1])
            shift_z = np.random.uniform(self.shift[2][0], self.shift[2][1])
            shift = np.array([shift_x, shift_y, shift_z])
            data_dict["coord"] += shift
            _apply_to_v3_vertex(data_dict, lambda vertex: vertex + shift)
        return data_dict


@TRANSFORMS.register_module()
class PointClip(object):
    def __init__(self, point_cloud_range=(-80, -80, -3, 80, 80, 1)):
        self.point_cloud_range = point_cloud_range

    def __call__(self, data_dict):
        if "coord" in data_dict.keys():
            data_dict["coord"] = np.clip(
                data_dict["coord"],
                a_min=self.point_cloud_range[:3],
                a_max=self.point_cloud_range[3:],
            )
        return data_dict


@TRANSFORMS.register_module()
class RandomDropout(object):
    def __init__(self, dropout_ratio=0.2, dropout_application_ratio=0.5):
        """
        upright_axis: axis index among x,y,z, i.e. 2 for z
        """
        self.dropout_ratio = dropout_ratio
        self.dropout_application_ratio = dropout_application_ratio

    def __call__(self, data_dict):
        if random.random() < self.dropout_application_ratio:
            n = len(data_dict["coord"])
            idx = np.random.choice(n, int(n * (1 - self.dropout_ratio)), replace=False)
            if "sampled_index" in data_dict:
                # for ScanNet data efficient, we need to make sure labeled point is sampled.
                idx = np.unique(np.append(idx, data_dict["sampled_index"]))
                mask = np.zeros_like(data_dict["segment"]).astype(bool)
                mask[data_dict["sampled_index"]] = True
                data_dict["sampled_index"] = np.where(mask[idx])[0]
            data_dict = index_operator(data_dict, idx)
        return data_dict


@TRANSFORMS.register_module()
class RandomRotate(object):
    def __init__(self, angle=None, center=None, axis="z", always_apply=False, p=0.5):
        self.angle = [-1, 1] if angle is None else angle
        self.axis = axis
        self.always_apply = always_apply
        self.p = p if not self.always_apply else 1
        self.center = center

    def __call__(self, data_dict):
        if random.random() > self.p:
            return data_dict
        angle = np.random.uniform(self.angle[0], self.angle[1]) * np.pi
        return _rotate_about_center(
            data_dict, _rotation_matrix(self.axis, angle), self.center)


@TRANSFORMS.register_module()
class RandomRotateTargetAngle(object):
    def __init__(
        self, angle=(1 / 2, 1, 3 / 2), center=None, axis="z", always_apply=False, p=0.75
    ):
        self.angle = angle
        self.axis = axis
        self.always_apply = always_apply
        self.p = p if not self.always_apply else 1
        self.center = center

    def __call__(self, data_dict):
        if random.random() > self.p:
            return data_dict
        angle = np.random.choice(self.angle) * np.pi
        return _rotate_about_center(
            data_dict, _rotation_matrix(self.axis, angle), self.center)


@TRANSFORMS.register_module()
class RandomScale(object):
    def __init__(self, scale=None, anisotropic=False):
        self.scale = scale if scale is not None else [0.95, 1.05]
        self.anisotropic = anisotropic

    def __call__(self, data_dict):
        if "coord" in data_dict.keys():
            scale = np.random.uniform(
                self.scale[0], self.scale[1], 3 if self.anisotropic else 1
            )
            data_dict["coord"] *= scale
            _apply_to_v3_vertex(data_dict, lambda vertex: vertex * scale)
        return data_dict


@TRANSFORMS.register_module()
class RandomFlip(object):
    def __init__(self, p=0.5, axes=("x", "y",)):
        self.p = p
        if not isinstance(axes, tuple):
            axes = (axes,)
        self.axes = axes

    def __call__(self, data_dict):
        for axis in self.axes:
            dim = _ROT_AXIS_DIM.get(axis)
            # unknown axis: skip without drawing (matches the old if/elif chain)
            if dim is None:
                continue
            if random.random() < self.p:
                data_dict["coord"][:, dim] = -data_dict["coord"][:, dim]
                _apply_to_v3_vertex(
                    data_dict, lambda v, d=dim: _negate_axis(v, d))
                if "normal" in data_dict.keys():
                    data_dict["normal"][:, dim] = -data_dict["normal"][:, dim]
        return data_dict


@TRANSFORMS.register_module()
class RandomJitter(object):
    def __init__(self, sigma=0.01, clip=0.05, keys=("coord",), p=1.0):
        assert clip > 0
        self.sigma = sigma
        self.clip = clip
        if not isinstance(keys, tuple):
            keys = (keys,)
        self.keys = keys
        self.p = p
    def __call__(self, data_dict):
        if random.random() > self.p:
            return data_dict
        for k in self.keys:
            if k in data_dict.keys():
                jitter = np.clip(
                    self.sigma
                    * np.random.randn(data_dict[k].shape[0], data_dict[k].shape[1]),
                    -self.clip,
                    self.clip,
                )
                data_dict[k] += jitter
            else:
                raise ValueError(f"Key {k} not found in data_dict")
        return data_dict


@TRANSFORMS.register_module()
class MultiplicativeRandomJitter(object):
    def __init__(self, sigma=0.05, clip=0.05, keys=("energy",), p=0.5):
        assert clip > 0
        self.sigma = sigma
        self.clip = clip
        if not isinstance(keys, tuple):
            keys = (keys,)
        self.keys = keys
        self.p = p

    def __call__(self, data_dict):
        if random.random() > self.p:
            return data_dict
        for k in self.keys:
            if k in data_dict.keys():
                noise = np.clip(
                    np.random.randn(*data_dict[k].shape) * self.sigma,
                    -self.clip,
                    self.clip,
                )
                data_dict[k] *= 1.0 + noise
            else:
                raise ValueError(f"Key {k} not found in data_dict")
        return data_dict

@TRANSFORMS.register_module()
class SetRandomValue(object):
    def __init__(self, sigma=0.05, clip=0.05, keys=("energy",)):
        assert clip > 0
        self.sigma = sigma
        self.clip = clip
        self.keys = keys if isinstance(keys, tuple) else (keys,)

    def __call__(self, data_dict):
        for k in self.keys:
            if k in data_dict.keys():
                data_dict[k] = np.clip(
                    np.random.randn(*data_dict[k].shape) * self.sigma,
                    -self.clip,
                    self.clip,
                )
            else:
                raise ValueError(f"Key {k} not found in data_dict")
        return data_dict


@TRANSFORMS.register_module()
class ClipGaussianJitter(object):
    def __init__(self, scalar=0.02, store_jitter=False):
        self.scalar = scalar
        self.mean = np.zeros(3)  # zero-mean jitter (was np.mean(3) -> scalar 3.0, a +3 offset bug)
        self.cov = np.identity(3)
        self.quantile = 1.96
        self.store_jitter = store_jitter

    def __call__(self, data_dict):
        if "coord" in data_dict.keys():
            jitter = np.random.multivariate_normal(
                self.mean, self.cov, data_dict["coord"].shape[0]
            )
            jitter = self.scalar * np.clip(jitter / 1.96, -1, 1)
            data_dict["coord"] += jitter
            if self.store_jitter:
                data_dict["jitter"] = jitter
        return data_dict


@TRANSFORMS.register_module()
class EnergeticTranslation(object):
    def __init__(self, p=0.95, ratio=0.05):
        self.p = p
        self.ratio = ratio

    def __call__(self, data_dict):
        if "energy" in data_dict.keys() and np.random.rand() < self.p:
            tr = (np.random.rand(1) - 0.5) * 2 * self.ratio
            data_dict["energy"] = np.clip(tr + data_dict["energy"], -1, 1)
        return data_dict


@TRANSFORMS.register_module()
class EnergyJitter(object):
    def __init__(self, p=0.5, jitter_ratio=0.005, min_val=0.0):
        self.p = p
        self.jitter_ratio = jitter_ratio
        self.min_val = min_val

    def __call__(self, data_dict):
        if "energy" in data_dict.keys() and np.random.rand() < self.p:
            # jitter by +/- 0.5%
            jitter = (
                np.random.rand(*data_dict["energy"].shape) * 2 - 1
            ) * self.jitter_ratio
            data_dict["energy"] = np.clip(
                data_dict["energy"] * (1 + jitter), self.min_val, None
            )
        return data_dict


@TRANSFORMS.register_module()
class ElasticDistortion(object):
    def __init__(self, distortion_params=None):
        self.distortion_params = (
            [[0.2, 0.4], [0.8, 1.6]] if distortion_params is None else distortion_params
        )

    @staticmethod
    def elastic_distortion(coords, granularity, magnitude):
        """
        Apply elastic distortion on sparse coordinate space.
        pointcloud: numpy array of (number of points, at least 3 spatial dims)
        granularity: size of the noise grid (in same scale[m/cm] as the voxel grid)
        magnitude: noise multiplier
        """
        blurx = np.ones((3, 1, 1, 1)).astype("float32") / 3
        blury = np.ones((1, 3, 1, 1)).astype("float32") / 3
        blurz = np.ones((1, 1, 3, 1)).astype("float32") / 3
        coords_min = coords.min(0)

        # Create Gaussian noise tensor of the size given by granularity.
        noise_dim = ((coords - coords_min).max(0) // granularity).astype(int) + 3
        noise = np.random.randn(*noise_dim, 3).astype(np.float32)

        # Smoothing.
        for _ in range(2):
            noise = scipy.ndimage.filters.convolve(
                noise, blurx, mode="constant", cval=0
            )
            noise = scipy.ndimage.filters.convolve(
                noise, blury, mode="constant", cval=0
            )
            noise = scipy.ndimage.filters.convolve(
                noise, blurz, mode="constant", cval=0
            )

        # Trilinear interpolate noise filters for each spatial dimensions.
        ax = [
            np.linspace(d_min, d_max, d)
            for d_min, d_max, d in zip(
                coords_min - granularity,
                coords_min + granularity * (noise_dim - 2),
                noise_dim,
            )
        ]
        interp = scipy.interpolate.RegularGridInterpolator(
            ax, noise, bounds_error=False, fill_value=0
        )
        coords += interp(coords) * magnitude
        return coords

    def __call__(self, data_dict):
        if "coord" in data_dict.keys() and self.distortion_params is not None:
            if random.random() < 0.95:
                for granularity, magnitude in self.distortion_params:
                    data_dict["coord"] = self.elastic_distortion(
                        data_dict["coord"], granularity, magnitude
                    )
        return data_dict


@TRANSFORMS.register_module()
class GridSample(object):
    def __init__(
        self,
        grid_size=0.05,
        hash_type="fnv",
        mode="train",
        return_inverse=False,
        return_grid_coord=False,
        return_min_coord=False,
        return_displacement=False,
        project_displacement=False,
        sum_keys=None,
        min_keys=None,
        reducers=None,
    ):
        self.grid_size = grid_size
        self.hash = self.fnv_hash_vec if hash_type == "fnv" else self.ravel_hash_vec
        assert mode in ["train", "test"]
        self.mode = mode
        self.return_inverse = return_inverse
        self.return_grid_coord = return_grid_coord
        self.return_min_coord = return_min_coord
        self.return_displacement = return_displacement
        self.project_displacement = project_displacement
        self.sum_keys = sum_keys or []
        self.min_keys = min_keys or []
        # Per-key voxel reducer map {key: 'sum'|'min'|'max'|'mean'|'first'}.
        # Back-compat: sum_keys/min_keys fold in here; an explicit `reducers`
        # entry wins on conflict.
        self._reducers = {}
        for _k in self.sum_keys:
            self._reducers[_k] = "sum"
        for _k in self.min_keys:
            self._reducers[_k] = "min"
        if reducers:
            self._reducers.update(reducers)
        _ops = {"sum", "min", "max", "mean", "first"}
        bad = {op for op in self._reducers.values() if op not in _ops}
        if bad:
            raise ValueError(f"GridSample: unknown reducer op(s) {bad}; "
                             f"valid: {sorted(_ops)}")

    def __call__(self, data_dict):
        assert "coord" in data_dict.keys()
        scaled_coord = data_dict["coord"] / np.array(self.grid_size)
        grid_coord = np.floor(scaled_coord).astype(int)
        min_coord = grid_coord.min(0)
        grid_coord -= min_coord
        scaled_coord -= min_coord
        min_coord = min_coord * np.array(self.grid_size)
        key = self.hash(grid_coord)
        idx_sort = np.argsort(key)
        key_sort = key[idx_sort]
        _, inverse, count = np.unique(key_sort, return_inverse=True, return_counts=True)
        if self.mode == "train":  # train mode
            idx_select = (
                np.cumsum(np.insert(count, 0, 0)[0:-1])
                + np.random.randint(0, count.max(), count.size) % count
            )
            idx_unique = idx_sort[idx_select]
            if "sampled_index" in data_dict:
                # for ScanNet data efficient, we need to make sure labeled point is sampled.
                idx_unique = np.unique(
                    np.append(idx_unique, data_dict["sampled_index"])
                )
                mask = np.zeros_like(data_dict["segment"]).astype(bool)
                mask[data_dict["sampled_index"]] = True
                data_dict["sampled_index"] = np.where(mask[idx_unique])[0]
            reduced = {}
            if self._reducers:
                voxel_of_point = np.empty(len(key), dtype=inverse.dtype)
                voxel_of_point[idx_sort] = inverse
                num_voxels = len(count)
                # First-in-sorted-order representative per voxel — deterministic
                # (independent of the random survivor picked above).
                first_sel = np.cumsum(np.insert(count, 0, 0)[:-1])
                for rk, op in self._reducers.items():
                    if rk not in data_dict:
                        continue
                    vals = data_dict[rk]
                    shape = (num_voxels,) + vals.shape[1:]
                    if op == "sum":
                        # Accumulate WIDE — summing in vals.dtype silently
                        # saturates/wraps narrow real dtypes (float16 sum
                        # 374771 -> 4096; int8/uint16 wrap). Keep the wide
                        # output (int64 / float32) rather than re-narrowing.
                        if np.issubdtype(vals.dtype, np.integer):
                            agg = np.zeros(shape, dtype=np.int64)
                            np.add.at(agg, voxel_of_point, vals)
                        else:
                            acc = np.zeros(shape, dtype=np.float64)
                            np.add.at(acc, voxel_of_point, vals)
                            agg = acc.astype(np.float32)
                    elif op == "min":
                        fill = (np.inf if np.issubdtype(vals.dtype, np.floating)
                                else np.iinfo(vals.dtype).max)
                        agg = np.full(shape, fill, dtype=vals.dtype)
                        np.minimum.at(agg, voxel_of_point, vals)
                    elif op == "max":
                        fill = (-np.inf if np.issubdtype(vals.dtype, np.floating)
                                else np.iinfo(vals.dtype).min)
                        agg = np.full(shape, fill, dtype=vals.dtype)
                        np.maximum.at(agg, voxel_of_point, vals)
                    elif op == "mean":
                        s = np.zeros(shape, dtype=np.float64)
                        np.add.at(s, voxel_of_point, vals)
                        c = count.reshape((-1,) + (1,) * (vals.ndim - 1))
                        # Casts back to vals.dtype — an integer column's mean is
                        # floored toward zero. Intended for float columns
                        # (energy); use a float column if you need a true mean.
                        agg = (s / c).astype(vals.dtype)
                    else:  # "first"
                        agg = vals[idx_sort[first_sel]]
                    reduced[rk] = agg
            data_dict = index_operator(data_dict, idx_unique)
            for rk, agg in reduced.items():
                data_dict[rk] = agg
            if self.return_inverse:
                data_dict["inverse"] = np.zeros_like(inverse)
                data_dict["inverse"][idx_sort] = inverse
            if self.return_grid_coord:
                data_dict["grid_coord"] = grid_coord[idx_unique]
                data_dict["index_valid_keys"].append("grid_coord")
            if self.return_min_coord:
                data_dict["min_coord"] = min_coord.reshape([1, 3])
            if self.return_displacement:
                displacement = (
                    scaled_coord - grid_coord - 0.5
                )  # [0, 1] -> [-0.5, 0.5] displacement to center
                if self.project_displacement:
                    displacement = np.sum(
                        displacement * data_dict["normal"], axis=-1, keepdims=True
                    )
                data_dict["displacement"] = displacement[idx_unique]
                data_dict["index_valid_keys"].append("displacement")
            return data_dict

        elif self.mode == "test":  # test mode
            data_part_list = []
            for i in range(count.max()):
                idx_select = np.cumsum(np.insert(count, 0, 0)[0:-1]) + i % count
                idx_part = idx_sort[idx_select]
                data_part = index_operator(data_dict, idx_part, duplicate=True)
                data_part["index"] = idx_part
                if self.return_inverse:
                    data_part["inverse"] = np.zeros_like(inverse)
                    data_part["inverse"][idx_sort] = inverse
                if self.return_grid_coord:
                    data_part["grid_coord"] = grid_coord[idx_part]
                    data_dict["index_valid_keys"].append("grid_coord")
                if self.return_min_coord:
                    data_part["min_coord"] = min_coord.reshape([1, 3])
                if self.return_displacement:
                    displacement = (
                        scaled_coord - grid_coord - 0.5
                    )  # [0, 1] -> [-0.5, 0.5] displacement to center
                    if self.project_displacement:
                        displacement = np.sum(
                            displacement * data_dict["normal"], axis=-1, keepdims=True
                        )
                    data_dict["displacement"] = displacement[idx_part]
                    data_dict["index_valid_keys"].append("displacement")
                data_part_list.append(data_part)
            return data_part_list
        else:
            raise NotImplementedError

    @staticmethod
    def ravel_hash_vec(arr):
        """
        Ravel the coordinates after subtracting the min coordinates.
        """
        assert arr.ndim == 2
        arr = arr.copy()
        arr -= arr.min(0)
        arr = arr.astype(np.uint64, copy=False)
        arr_max = arr.max(0).astype(np.uint64) + 1

        keys = np.zeros(arr.shape[0], dtype=np.uint64)
        # Fortran style indexing
        for j in range(arr.shape[1] - 1):
            keys += arr[:, j]
            keys *= arr_max[j + 1]
        keys += arr[:, -1]
        return keys

    @staticmethod
    def fnv_hash_vec(arr):
        """
        FNV64-1A
        """
        assert arr.ndim == 2
        # Floor first for negative coordinates
        arr = arr.copy()
        arr = arr.astype(np.uint64, copy=False)
        hashed_arr = np.uint64(14695981039346656037) * np.ones(
            arr.shape[0], dtype=np.uint64
        )
        for j in range(arr.shape[1]):
            hashed_arr *= np.uint64(1099511628211)
            hashed_arr = np.bitwise_xor(hashed_arr, arr[:, j])
        return hashed_arr


@TRANSFORMS.register_module()
class SphereCrop(object):
    def __init__(self, point_max=80000, sample_rate=None, mode="random"):
        self.point_max = point_max
        self.sample_rate = sample_rate
        assert mode in ["random", "center", "all"]
        self.mode = mode

    def __call__(self, data_dict):
        point_max = (
            int(self.sample_rate * data_dict["coord"].shape[0])
            if self.sample_rate is not None
            else self.point_max
        )

        assert "coord" in data_dict.keys()
        if data_dict["coord"].shape[0] > point_max:
            if self.mode == "random":
                center = data_dict["coord"][
                    np.random.randint(data_dict["coord"].shape[0])
                ]
            elif self.mode == "center":
                center = data_dict["coord"][data_dict["coord"].shape[0] // 2]
            else:
                raise NotImplementedError
            idx_crop = np.argsort(np.sum(np.square(data_dict["coord"] - center), 1))[
                :point_max
            ]
            data_dict = index_operator(data_dict, idx_crop)
        return data_dict


@TRANSFORMS.register_module()
class HardExampleCrop(object):
    def __init__(
        self,
        point_max=80000,
        sample_rate=None,
        hard_labels=(2, 3),
        min_hard_points=1,
        attempts=5,
        fallback="none",  # random | center | none
        p=0.5,
    ):
        self.point_max = point_max
        self.sample_rate = sample_rate
        self.hard_labels = tuple(hard_labels)
        self.min_hard_points = int(min_hard_points)
        self.attempts = int(attempts)
        assert fallback in ["random", "center", "none"]
        self.fallback = fallback
        self.p = p
    def __call__(self, data_dict):
        assert "coord" in data_dict
        if np.random.rand() >= self.p:
            return data_dict
        n_points = data_dict["coord"].shape[0]
        point_max = (
            int(self.sample_rate * n_points)
            if self.sample_rate is not None
            else self.point_max
        )
        if n_points <= point_max:
            return data_dict

        coord = data_dict["coord"]
        segment = data_dict.get("segment", None)

        if segment is not None:
            seg = segment.reshape(-1)
            hard_mask = np.isin(seg, self.hard_labels)
        else:
            hard_mask = np.zeros(n_points, dtype=bool)

        if hard_mask.any():
            hard_indices = np.where(hard_mask)[0]
            last_idx_crop = None
            for _ in range(max(1, self.attempts)):
                cidx = np.random.choice(hard_indices)
                center = coord[cidx]
                idx_crop = np.argsort(np.sum(np.square(coord - center), axis=1))[:point_max]
                last_idx_crop = idx_crop
                if self.min_hard_points <= np.count_nonzero(hard_mask[idx_crop]):
                    return index_operator(data_dict, idx_crop)
            return index_operator(data_dict, last_idx_crop)

        # fallback when no hard points present
        if self.fallback == "none":
            return data_dict
        elif self.fallback == "center":
            center = coord[n_points // 2]
        else:  # random
            center = coord[np.random.randint(n_points)]
        idx_crop = np.argsort(np.sum(np.square(coord - center), axis=1))[:point_max]
        return index_operator(data_dict, idx_crop)


@TRANSFORMS.register_module()
class ShufflePoint(object):
    def __call__(self, data_dict):
        assert "coord" in data_dict.keys()
        shuffle_index = np.arange(data_dict["coord"].shape[0])
        np.random.shuffle(shuffle_index)
        data_dict = index_operator(data_dict, shuffle_index)
        return data_dict


@TRANSFORMS.register_module()
class LocalCovarianceFeatures(object):
    def __init__(
        self,
        k=16,
        include_self=False,
        gaussian_weight=False,
        gaussian_sigma=None,
        out_keys=("local_eigvals", "local_shape"),
    ):
        self.k = int(k)
        self.include_self = bool(include_self)
        self.gaussian_weight = bool(gaussian_weight)
        self.gaussian_sigma = gaussian_sigma
        self.out_keys = out_keys

    def __call__(self, data_dict):
        if "coord" not in data_dict:
            return data_dict
        coord = np.asarray(data_dict["coord"]).astype(np.float32, copy=False)
        n_points = coord.shape[0]
        if n_points == 0:
            return data_dict

        # query kNN
        k_query = min(self.k + (0 if self.include_self else 1), max(1, n_points))
        kd = cKDTree(coord)
        dists, idxs = kd.query(coord, k=k_query)

        # ensure shape (N, K)
        if dists.ndim == 1:
            dists = dists[:, None]
            idxs = idxs[:, None]

        # drop self if requested
        if not self.include_self and idxs.shape[1] > 0:
            dists = dists[:, 1:]
            idxs = idxs[:, 1:]

        neighbors = coord[idxs]  # (N, K, 3)
        center = coord[:, None, :]  # (N, 1, 3)
        offsets = neighbors - center

        if self.gaussian_weight:
            if self.gaussian_sigma is None:
                sigma = np.median(dists, axis=1, keepdims=True) + 1e-6
            else:
                sigma = float(self.gaussian_sigma)
            w = np.exp(-0.5 * (dists / (sigma + 1e-6)) ** 2).astype(np.float32)
            w_sum = np.sum(w, axis=1, keepdims=True) + 1e-6
            mean = np.sum(offsets * w[..., None], axis=1, keepdims=True) / w_sum[..., None]
            centered = offsets - mean
            cov = np.einsum("nki,nkj->nij", centered * w[..., None], centered) / w_sum[..., None]
        else:
            mean = np.mean(offsets, axis=1, keepdims=True)
            centered = offsets - mean
            denom = float(max(1, centered.shape[1] - 1))
            cov = np.einsum("nki,nkj->nij", centered, centered) / denom

        eigvals, _ = np.linalg.eigh(cov)
        eigvals = np.clip(eigvals, 0.0, None)[:, ::-1]

        l1 = eigvals[:, 0] + 1e-12
        l2 = eigvals[:, 1] + 1e-12
        l3 = eigvals[:, 2] + 1e-12
        r21 = l2 / l1
        r32 = l3 / l2
        r31 = l3 / l1
        suml = (l1 + l2 + l3) + 1e-12
        curvature = l3 / suml
        local_shape = np.stack([r21, r32, r31, curvature], axis=1).astype(coord.dtype, copy=False)
        local_eigvals = eigvals.astype(coord.dtype, copy=False)

        data_dict[self.out_keys[0]] = local_eigvals
        data_dict[self.out_keys[1]] = local_shape

        data_dict["index_valid_keys"].append(self.out_keys[0])
        data_dict["index_valid_keys"].append(self.out_keys[1])
        return data_dict


@TRANSFORMS.register_module()
class RandomDrop(object):
    def __init__(self, key: str, p_apply: float = 0.5, p_drop: float = 0.2, value: float = 0.0):
        self.key = key
        self.p_apply = p_apply
        self.p_drop = p_drop
        self.value = value

    def __call__(self, data_dict):
        if self.key in data_dict.keys() and np.random.rand() < self.p_apply:
            n = data_dict[self.key].shape[0]
            idx = np.random.choice(n, int(n*self.p_drop), replace=False)
            # assign back through the fancy index; `[idx][:] =` wrote into a copy (no-op bug)
            data_dict[self.key][idx] = self.value
        return data_dict

class Compose(object):
    """Chain a list of transforms.

    Each entry may be:
      * a registry config dict like ``dict(type="GridSample", grid_size=...)``,
        which is resolved via the :data:`TRANSFORMS` registry, or
      * any plain callable ``fn(data_dict) -> data_dict`` (user lambdas,
        pre-built transform instances, etc.).

    Dict and callable entries may be mixed in the same list.

    A transform declares its ``scope`` (default ``'sample'``): ``'sample'`` sees
    one event and is a pre-collate ``map``; ``'batch'`` needs the collated set
    (offset, cross-event) and is post-collate only. ``Compose`` is the pre-collate
    site, so a ``scope='batch'`` entry is rejected here — place it in the
    post-collate stage list (see :mod:`pimm_data.batch_transforms`) instead.
    """

    def __init__(self, cfg=None):
        self.cfg = cfg if cfg is not None else []
        self.transforms = []
        for t_cfg in self.cfg:
            if isinstance(t_cfg, dict):
                t = TRANSFORMS.build(t_cfg)
            elif callable(t_cfg):
                t = t_cfg
            else:
                raise TypeError(
                    "Compose entries must be dict (registry config) or callable, "
                    f"got {type(t_cfg).__name__}")
            # fence: a scope='batch' transform reads the collated set; it cannot
            # run per-event in the worker. Catch the misplacement at build time.
            if getattr(t, 'scope', 'sample') == 'batch':
                raise ValueError(
                    f"{type(t).__name__} is scope='batch' (needs the collated "
                    "set) and cannot run pre-collate in Compose; place it in the "
                    "post-collate stage list (apply_batch_transforms).")
            self.transforms.append(t)

    def __call__(self, data_dict):
        for t in self.transforms:
            data_dict = t(data_dict)
        return data_dict
