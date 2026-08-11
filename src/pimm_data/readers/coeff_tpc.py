"""Coeff-corpus reader/writer — the flat-columnar wavelet-coefficient shards.

The coeff corpus is a purpose-built training cache (produced by helix DSP), read
hot in the loop, so it is **flat columnar**, not per-event groups: shard-wide
``/coord`` + ``/value`` arrays plus an ``event_offset`` index; one contiguous
slice per event; ``plane_gid`` is a column. This is pimm-data's own offset idiom
(cf. the optical second row-space) applied shard-wide.

One modality per file (``{name}_coeff_*.h5`` noisy input, ``{name}_coeff_clean_*``
the clean target), joined by identity — never mixed. The on-disk layout is the
shared contract defined in helix ``COEFF_CORPUS_DESIGN.md`` §3; this reader
implements it with **standalone h5py** (no helix import at read time — no cycle),
and is round-trip golden-tested against helix's reference codec.

Layout::

    /config  attrs: n_events, dataset_name, file_index, global_event_offset,
                    readout_type, wavelet, dwt_level, dwt_mode, n_ticks_raw, pad,
                    sigma_norm, basis_digest, removal_json, threshold_json
             datasets: band_lengths (n_bands,), gids (G,), n_wires (G,),
                       [norm_sigma (G, n_bands)]
    /coord   band uint8, plane_gid uint8, wire int32, tau int32   (all (M,))
             event_offset int64 (n_events+1,); sigma_threshold f32 (n_events,G,n_bands)
    /value   float32 (M,)
    /ident   run (n_events,), source_file (n_events,), event int64 (n_events,)
"""

import glob
import hashlib
import json
import os

import numpy as np
import h5py

from .._shard_meta import read_shard_meta
from ._base import ShardReaderBase

_STR = h5py.string_dtype(encoding="utf-8")


def coord_digest(band, plane_gid, wire, tau) -> int:
    """Order-sensitive 64-bit digest of one event's coords (mirrors helix).

    A co-supported clean shard stores values only — its coords are, by
    construction, the noisy shard's — so this is what makes the join checkable.
    Must stay byte-identical to ``helix.core.coeff_io.coord_digest``; the
    cross-repo golden test pins it.
    """
    h = hashlib.blake2b(digest_size=8)
    for a, dt in ((band, np.uint8), (plane_gid, np.int32),
                  (wire, np.int32), (tau, np.int32)):
        h.update(np.ascontiguousarray(np.asarray(a), dt).tobytes())
    return int.from_bytes(h.digest(), "little")


class CoeffTPCReader(ShardReaderBase):
    """Flat-columnar coeff-shard reader. ``modality`` selects the file family
    (``'coeff'`` noisy / ``'coeff_clean'`` target)."""

    _MODALITY = "coeff"

    def __init__(self, data_root, split="", dataset_name="coeff_tpc", modality="coeff", **kwargs):
        self.data_root = data_root
        self.split = split
        self.dataset_name = dataset_name
        self._MODALITY = modality               # instance attr shadows the class default
        # Shard-shared metadata. Taken from the first shard AND verified against
        # every other one: these tables are corpus-wide by contract (norm_sigma
        # in particular must be frozen across shards), so a disagreement means
        # the shards are not one corpus. Serving shard 0's tables for every event
        # silently mis-normalises everything outside shard 0.
        self.band_lengths = None
        self.gids = None
        self.n_wires = None
        self.norm_sigma = None
        self.sigma_norm = None
        self.config_attrs = None
        self.has_coords = True
        self._meta_source = None
        self._meta_by_path = {}                 # h5_path -> its /config tables
        self._ident_by_path = {}                # h5_path -> raw (run, source_file, event)
        self._dup_ident = {}                    # h5_path -> (rows, unique) when not 1:1
        self._id2pos = {}                       # h5_path -> {physical event id: write-order position}
        self._init_shards()
        self._validate_shard_agreement()

    def _validate_shard_agreement(self):
        """Every shard of one corpus must share the basis, plane set and — above
        all — ``norm_sigma``. Row i of that table normalises any event in any
        shard, so a per-shard table means the same physical coefficient is scaled
        differently depending on where it was written.

        Deliberately run OUTSIDE ``_build_index``: that loop catches every
        exception and downgrades it to a log warning, dropping the offending
        shard's events silently — so raising from ``_index_for_shard`` would turn
        a mis-built corpus into a quietly shorter one.
        """
        def _same(a, b):
            if a is None or b is None:
                return a is None and b is None
            return np.array_equal(np.asarray(a), np.asarray(b))

        for path, (rows, uniq) in self._dup_ident.items():
            raise ValueError(
                f"{path}: /ident does not uniquely identify its events — {rows} rows "
                f"collapse to {uniq} identities. (run, source_file, event) must be "
                f"unique within a shard; duplicates would be silently dropped.")
        if not self._meta_by_path:
            return
        (ref_path, ref), *rest = self._meta_by_path.items()
        for path, m in rest:
            for name in ("band_lengths", "gids", "n_wires", "norm_sigma"):
                if not _same(ref[name], m[name]):
                    a = None if ref[name] is None else np.asarray(ref[name]).tolist()
                    b = None if m[name] is None else np.asarray(m[name]).tolist()
                    raise ValueError(
                        f"{self._MODALITY} shards disagree on /config/{name}:\n"
                        f"  {ref_path} -> {a}\n  {path} -> {b}\n"
                        f"These shards are not one corpus. norm_sigma in particular must "
                        f"be FROZEN across a corpus (build with --norm-sigma).")
            if abs(float(m["sigma_norm"]) - float(ref["sigma_norm"])) > 1e-6:
                raise ValueError(
                    f"{self._MODALITY} shards disagree on sigma_norm: "
                    f"{ref_path} -> {ref['sigma_norm']}, {path} -> {m['sigma_norm']}")
            if bool(m["has_coords"]) != bool(ref["has_coords"]):
                raise ValueError(
                    f"{self._MODALITY} shards disagree on has_coords: "
                    f"{ref_path} -> {ref['has_coords']}, {path} -> {m['has_coords']}")

    def _find_files(self):
        """Glob shards with the tag anchored to a digit — ``{name}_{modality}_[0-9]*.h5``
        — so ``modality='coeff'`` does NOT also match ``{name}_coeff_clean_*.h5``
        (``coeff`` is a prefix of ``coeff_clean``)."""
        pat = f"{self.dataset_name}_{self._MODALITY}_[0-9]*.h5"
        if isinstance(self.split, (list, tuple)):
            files, file_runs = [], []
            for run in self.split:
                rf = sorted(glob.glob(os.path.join(self.data_root, run, pat)))
                if not rf:
                    raise FileNotFoundError(
                        f"runs=: no {self._MODALITY} shards for run {run!r} under {self.data_root}")
                files.extend(rf); file_runs.extend([run] * len(rf))
            self._file_runs = file_runs
            return files
        for p in (os.path.join(self.data_root, self.split, pat),
                  os.path.join(self.data_root, pat)):
            files = sorted(glob.glob(p))
            if files:
                return files
        return []

    def _index_for_shard(self, h5_path):
        """Present event **ids** for one flat shard — the ``/ident/event`` physical
        ids (in write order) so the cross-modality joint index aligns coeff↔coeff_clean
        by IDENTITY, not position. Also caches shared basis/geometry, and a
        physical-id→write-position map (event_offset is positional)."""
        meta = read_shard_meta(h5_path)
        with h5py.File(h5_path, "r", libver="latest", swmr=True) as f:
            cfg = f["config"]
            bl = cfg["band_lengths"][:].astype(np.int32)
            gids = cfg["gids"][:].astype(np.int32) if "gids" in cfg else None
            n_wires = cfg["n_wires"][:].astype(np.int32) if "n_wires" in cfg else None
            ns = cfg["norm_sigma"][:].astype(np.float32) if "norm_sigma" in cfg else None
            sn = float(cfg.attrs.get("sigma_norm", 1.0))
            hc = bool(cfg.attrs.get("has_coords", True))
            self._meta_by_path[str(h5_path)] = dict(
                band_lengths=bl, gids=gids, n_wires=n_wires, norm_sigma=ns,
                sigma_norm=sn, has_coords=hc)
            if self.band_lengths is None:
                self.config_attrs = dict(cfg.attrs)
                self.band_lengths, self.gids, self.n_wires = bl, gids, n_wires
                self.norm_sigma, self.sigma_norm = ns, sn
                self.has_coords = hc
                self._meta_source = str(h5_path)
            if "ident" in f and "event" in f["ident"]:
                ev = f["ident"]["event"][:].astype(np.int64)
                src = (f["ident"]["source_file"][:]
                       if "source_file" in f["ident"] else None)
                rn = f["ident"]["run"][:] if "run" in f["ident"] else None
                # Keep the RAW triple as well as the packed id. The packed id
                # exists to align coeff<->coeff_clean and is deliberately lossy
                # (a hash, once a corpus spans several runs); it cannot be turned
                # back into a source file and event number. Anything that wants
                # to join a coeff event to the simulation event it came from —
                # a probe target, a truth label — needs the triple itself.
                self._ident_by_path[str(h5_path)] = dict(
                    run=_as_str_list(rn, len(ev)),
                    source_file=_as_str_list(src, len(ev)),
                    event=ev.copy())
                ev = _identity_ids(src, ev, run=rn)
                # Record, don't raise: _build_index wraps this call in a bare
                # `except Exception` that downgrades any error to a log warning and
                # serves the shard as EMPTY — so raising here would turn a
                # detectable fault into a silently shorter dataset. Checked in
                # _validate_shard_agreement, which runs outside that loop.
                if len(np.unique(ev)) != ev.shape[0]:
                    self._dup_ident[str(h5_path)] = (ev.shape[0], len(np.unique(ev)))
            else:                                       # legacy shard without identity
                ev = np.arange(int(meta["n_events"]), dtype=np.int64)
        # keyed by path (not shard index) so it survives build_joint_index re-selection
        self._id2pos[str(h5_path)] = {int(e): i for i, e in enumerate(ev)}
        return np.asarray(ev, dtype=np.int64)

    def _locate_flat(self, idx):
        self._ensure_open()
        file_idx, event_id = self.locate(idx)           # event_id = physical /ident/event id
        path = str(self.h5_files[file_idx])
        pos = self._id2pos[path][int(event_id)]
        return self._h5data[file_idx], pos, path

    def read_event(self, idx):
        """One event → flat dict of coeff rows (sliced by ``event_offset`` at the
        physical event's write position).

        A values-only shard (``has_coords=False`` — the co-supported clean
        target) yields ``value`` and ``_coord_digest`` but no coordinate columns;
        :class:`~pimm_data.coeff.CoeffTPCDataset` supplies them from the paired
        noisy event after checking the digest.
        """
        f, ev, path = self._locate_flat(idx)
        coord = f["coord"]
        off = coord["event_offset"]
        a, b = int(off[ev]), int(off[ev + 1])
        out = {"value": f["value"][a:b].astype(np.float32)}
        if "coord_digest" in coord:
            out["_coord_digest"] = int(coord["coord_digest"][ev])
        ident = self._ident_by_path.get(path)
        if ident is not None:
            out["_ident"] = dict(run=ident["run"][ev],
                                 source_file=ident["source_file"][ev],
                                 event=int(ident["event"][ev]))
        if self.has_coords:
            out.update(
                band=coord["band"][a:b].astype(np.int32),
                plane_gid=coord["plane_gid"][a:b].astype(np.int32),
                wire=coord["wire"][a:b].astype(np.int32),
                tau=coord["tau"][a:b].astype(np.int32),
            )
        return out


def _as_str_list(arr, n):
    """h5py string column -> list[str] (``None`` -> empty strings)."""
    if arr is None:
        return [""] * n
    return [s.decode() if isinstance(s, bytes) else str(s) for s in arr]


def _identity_ids(source_file, event, run=None):
    """Per-event identity for the joint index: ``(run, source_file, event)``.

    ``/ident/event`` is the event's index WITHIN its source file, so it restarts
    at 0 in every file. A shard that spans more than one source file — which is
    exactly what loader mode produces, since it reads a run-wide joint index that
    crosses file boundaries — therefore contains repeated ids, and a map keyed on
    the id alone silently collapses them (measured: 250 rows into 200 slots, so
    20% of a 1000-event corpus was unreachable).

    ``run`` is load-bearing too: every detector run restarts shard numbering at
    ``*_sensor_0000.h5``, so ``(source_file, event)`` alone collides across runs.
    That is the identity the design doc specifies, and it is what a multi-run
    corpus needs. Encoded as ``file_ordinal << 32 | event`` for the common
    single-run case (keeps ids ORDERED, so iteration still follows the data), and
    as a stable 64-bit hash once more than one run is present or the source names
    are not numbered.
    """
    ev = np.asarray(event, np.int64)
    if source_file is None:
        return ev
    names = [s.decode() if isinstance(s, bytes) else str(s) for s in source_file]
    runs = ([r.decode() if isinstance(r, bytes) else str(r) for r in run]
            if run is not None else [""] * len(names))

    # Layout: [run 22b | file_ordinal 16b | event 24b] = 62 bits, positive int64.
    # The run goes in the HIGH bits so ordering WITHIN a run still follows the
    # data (index 0 is the first event, which a full hash would scramble), while
    # different runs occupy disjoint ranges. That matters because _index_for_shard
    # sees one shard — hence one run — at a time, so it cannot detect a cross-run
    # collision itself: every run restarts its shard numbering at *_sensor_0000.h5.
    _EVB, _FILEB = 24, 16
    sufs, ok = [], True
    for n in names:
        stem = n[:-3] if n.endswith(".h5") else n
        tail = stem.rsplit("_", 1)[-1]
        if tail.isdigit() and int(tail) < (1 << _FILEB):
            sufs.append(int(tail))
        else:
            ok = False
            break
    if ok and ev.size and int(ev.max()) < (1 << _EVB) and int(ev.min()) >= 0:
        rh = np.array(
            [0 if not r else
             int.from_bytes(hashlib.blake2b(r.encode(), digest_size=8).digest(),
                            "little") & 0x3FFFFF
             for r in runs], np.int64)
        return (rh << np.int64(_EVB + _FILEB)) | \
               (np.asarray(sufs, np.int64) << np.int64(_EVB)) | ev
    # names not numbered, or counts beyond the field widths: fall back to a full
    # hash. Ordering is lost, which is cosmetic; a collision deletes events.
    return np.array(
        [int.from_bytes(
            hashlib.blake2b(f"{r}#{n}#{int(e)}".encode(), digest_size=8).digest(),
            "little") & 0x7FFFFFFFFFFFFFFF
         for r, n, e in zip(runs, names, ev.tolist())], np.int64)


def basis_digest(*, wavelet, dwt_level, dwt_mode, n_ticks_raw, pad, band_lengths,
                 removal_json="{}", threshold_json="{}", sigma_norm=2.6) -> str:
    """sha256 of the canonical basis dict — mirrors ``BasisDescriptor.digest()``.

    Field NAMES are the descriptor's (``level``/``mode``), not the ``/config``
    attr names (``dwt_level``/``dwt_mode``). helix re-derives this on every read
    and refuses the shard on mismatch, so pimm-data computes it rather than
    trusting a caller-supplied string — a wrong digest produced a shard that
    pimm-data wrote happily and helix rejected. Pinned by the cross-repo golden.
    """
    d = dict(wavelet=str(wavelet), level=int(dwt_level), mode=str(dwt_mode),
             n_ticks_raw=int(n_ticks_raw), pad=int(pad),
             band_lengths=[int(x) for x in band_lengths],
             removal=json.loads(str(removal_json) or "{}"),
             threshold=json.loads(str(threshold_json) or "{}"),
             sigma_norm=float(sigma_norm))
    payload = json.dumps(d, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _ds(grp, name, data, comp):
    kw = dict(compression=comp, compression_opts=4) if comp == "gzip" else {}
    grp.create_dataset(name, data=data, **kw)


def write_coeff_shard(path, events, *, band_lengths, gids, n_wires, basis_attrs,
                      norm_sigma=None, dataset_name="", file_index=0,
                      global_event_offset=0, compression="gzip",
                      coords=True, noise=None, provenance=None):
    """Write a flat-columnar coeff shard (standalone; the layout helix reads).

    ``events`` is a list of per-event dicts with keys ``band, plane_gid, wire,
    tau, value, sigma_threshold, run, source_file, event``. All events share the
    plane set (``gids`` / ``n_wires``) and the basis. ``basis_attrs`` supplies the
    ``/config`` scalar attrs (wavelet, dwt_level, dwt_mode, n_ticks_raw, pad,
    sigma_norm, basis_digest, removal_json, threshold_json).

    ``coords=False`` writes a values-only shard for a co-supported target (see
    :func:`coord_digest`); ``noise`` records how the input was noised, so two
    shards built with different noise are distinguishable after the fact.

    ``provenance`` writes ``/config@provenance_json`` — the ``code`` block and the
    ``geom``/``spectrum`` hashes that :mod:`pimm_data.coeff_verify` compares
    across a corpus. This writer omitted it entirely, so no shard it produced
    could exercise those checks, and every fixture-based test of the verifier was
    silently testing the legacy path.
    """
    if not events:
        raise ValueError("write_coeff_shard: no events")

    # validate the basis contract so a pimm-written shard is not silently malformed
    # (helix's reader validates on read; fail here at write time instead).
    _REQUIRED = ("wavelet", "dwt_level", "dwt_mode", "n_ticks_raw", "pad", "sigma_norm")
    missing = [k for k in _REQUIRED if k not in basis_attrs]
    if missing:
        raise ValueError(f"basis_attrs missing required keys {missing}")
    # helix's reader re-hashes the basis and json-parses these on every read, so
    # anything it would reject must fail HERE — otherwise pimm-data happily writes
    # shards that the reference codec refuses, and the cross-repo contract is only
    # enforced in one direction.
    for k in ("removal_json", "threshold_json"):
        if k in basis_attrs:
            try:
                json.loads(str(basis_attrs[k]))
            except (json.JSONDecodeError, TypeError) as e:
                raise ValueError(f"basis_attrs[{k!r}] is not parseable JSON: {e}") from e
    import pywt
    padded = int(basis_attrs["n_ticks_raw"]) + int(basis_attrs["pad"])
    want = tuple(int(c.shape[-1]) for c in pywt.wavedec(
        np.zeros(padded, np.float32), str(basis_attrs["wavelet"]),
        level=int(basis_attrs["dwt_level"]), mode=str(basis_attrs["dwt_mode"])))
    if tuple(int(x) for x in band_lengths) != want:
        raise ValueError(
            f"band_lengths {tuple(int(x) for x in band_lengths)} inconsistent with basis "
            f"({basis_attrs['wavelet']} L{basis_attrs['dwt_level']} @ {padded}) → expected {want}")

    dig = basis_digest(
        wavelet=basis_attrs["wavelet"], dwt_level=basis_attrs["dwt_level"],
        dwt_mode=basis_attrs["dwt_mode"], n_ticks_raw=basis_attrs["n_ticks_raw"],
        pad=basis_attrs["pad"], band_lengths=band_lengths,
        removal_json=basis_attrs.get("removal_json", "{}"),
        threshold_json=basis_attrs.get("threshold_json", "{}"),
        sigma_norm=basis_attrs["sigma_norm"])
    given = basis_attrs.get("basis_digest")
    if given is not None and str(given) != dig:
        raise ValueError(
            f"basis_digest {str(given)!r} does not match the basis it describes "
            f"(hashes to {dig!r}). helix's reader recomputes this and refuses the "
            f"shard, so writing it would produce a file only pimm-data can read.")
    basis_attrs = {**basis_attrs, "basis_digest": dig}

    def cat(key, dt):
        parts = [np.asarray(e[key], dt) for e in events]
        return np.concatenate(parts) if parts else np.empty(0, dt)

    band = cat("band", np.uint8)
    plane_gid = cat("plane_gid", np.int32)          # int32: gid can exceed 255 (matches helix)
    wire = cat("wire", np.int32)
    tau = cat("tau", np.int32)
    value = cat("value", np.float32)
    counts = np.array([len(e["value"]) for e in events], np.int64)
    event_offset = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
    sigma = np.stack([np.asarray(e["sigma_threshold"], np.float32) for e in events])
    digests = np.array([coord_digest(e["band"], e["plane_gid"], e["wire"], e["tau"])
                        for e in events], np.uint64)

    with h5py.File(path, "w") as f:
        cfg = f.create_group("config")
        cfg.attrs["n_events"] = len(events)
        cfg.attrs["dataset_name"] = dataset_name
        cfg.attrs["file_index"] = file_index
        cfg.attrs["global_event_offset"] = global_event_offset
        cfg.attrs["readout_type"] = "wire"
        for k, v in basis_attrs.items():
            cfg.attrs[k] = v
        cfg.attrs["has_coords"] = bool(coords)
        cfg.attrs["noise_json"] = json.dumps(noise or {}, sort_keys=True)
        if provenance is not None:
            cfg.attrs["provenance_json"] = json.dumps(provenance, sort_keys=True)
        cfg.create_dataset("band_lengths", data=np.asarray(band_lengths, np.int32))
        cfg.create_dataset("gids", data=np.asarray(gids, np.int32))
        cfg.create_dataset("n_wires", data=np.asarray(n_wires, np.int32))
        if norm_sigma is not None:
            cfg.create_dataset("norm_sigma", data=np.asarray(norm_sigma, np.float32))

        coord = f.create_group("coord")
        if coords:
            _ds(coord, "band", band, compression)
            _ds(coord, "plane_gid", plane_gid, compression)
            _ds(coord, "wire", wire, compression)
            _ds(coord, "tau", tau, compression)
        coord.create_dataset("event_offset", data=event_offset)
        coord.create_dataset("coord_digest", data=digests)
        _ds(coord, "sigma_threshold", sigma, compression)

        _ds(f, "value", value, compression)

        ident = f.create_group("ident")
        ident.create_dataset("run", data=np.array([e.get("run", "") for e in events], dtype=_STR))
        ident.create_dataset("source_file", data=np.array([e.get("source_file", "") for e in events], dtype=_STR))
        ident.create_dataset("event", data=np.array([e.get("event", -1) for e in events], np.int64))
