"""CoeffTPCDataset / CoeffTPCReader — flat-columnar coeff corpus.

- self-contained: pimm-data write_coeff_shard → reader/Dataset/Collect round-trip.
- cross-repo golden: helix's reference codec ⇄ pimm-data reader agree on the layout
  (skipped when helix is not importable).
"""
import json
import os
import sys

import numpy as np
import pytest

from pimm_data.readers.coeff_tpc import CoeffTPCReader, write_coeff_shard
from pimm_data.coeff import CoeffTPCDataset
from pimm_data.transform import Compose
from pimm_data.coeff_verify import verify_corpus

BAND_LENGTHS = [8, 8, 16]
GIDS = [0, 1]
N_WIRES = [4, 4]
# NO basis_digest: the writer computes it. Hardcoding one ("deadbeef") produced
# shards that pimm-data wrote happily and helix's reader refused — the entire
# fixture set was unreadable by the reference codec.
BASIS_ATTRS = dict(
    wavelet="db2", dwt_level=2, dwt_mode="periodization", n_ticks_raw=32, pad=0,
    sigma_norm=2.6,
    removal_json=json.dumps(dict(kind="gate", kgate=3.0)),
    threshold_json=json.dumps(dict(method="universal")),
)


def _rand_event(rng, ev):
    band, gid, wire, tau, val = [], [], [], [], []
    for gi, g in enumerate(GIDS):
        for b, L in enumerate(BAND_LENGTHS):
            n = rng.integers(0, 6)
            if n == 0:
                continue
            band.append(np.full(n, b, np.uint8))
            gid.append(np.full(n, g, np.uint8))
            wire.append(rng.integers(0, N_WIRES[gi], n).astype(np.int32))
            tau.append(rng.integers(0, L, n).astype(np.int32))
            val.append(rng.standard_normal(n).astype(np.float32))
    cat = lambda xs, dt: (np.concatenate(xs).astype(dt) if xs else np.empty(0, dt))
    return dict(
        band=cat(band, np.uint8), plane_gid=cat(gid, np.uint8),
        wire=cat(wire, np.int32), tau=cat(tau, np.int32), value=cat(val, np.float32),
        sigma_threshold=rng.random((len(GIDS), len(BAND_LENGTHS))).astype(np.float32),
        run="run_000", source_file="cx_sensor_0000.h5", event=ev)


def _write_shard(path, n_events=3, seed=0):
    rng = np.random.default_rng(seed)
    events = [_rand_event(rng, e) for e in range(n_events)]
    write_coeff_shard(path, events, band_lengths=BAND_LENGTHS, gids=GIDS,
                      n_wires=N_WIRES, basis_attrs=BASIS_ATTRS, dataset_name="cx")
    return events


def test_reader_roundtrip(tmp_path):
    events = _write_shard(tmp_path / "cx_coeff_0000.h5")
    r = CoeffTPCReader(data_root=str(tmp_path), dataset_name="cx")
    assert len(r) == 3
    np.testing.assert_array_equal(r.band_lengths, BAND_LENGTHS)
    for i, e in enumerate(events):
        got = r.read_event(i)
        for k in ("band", "plane_gid", "wire", "tau", "value"):
            np.testing.assert_array_equal(got[k], e[k], err_msg=f"event {i} {k}")


def test_dataset_get_data(tmp_path):
    events = _write_shard(tmp_path / "cx_coeff_0000.h5")
    ds = CoeffTPCDataset(data_root=str(tmp_path), dataset_name="cx")
    assert len(ds) == 3
    d = ds.get_data(0)
    assert set(d["coeff"]) == {"band", "plane_gid", "wire", "tau", "value", "_meta"}
    # shard tables ride WITH the sample so a worker-side tokenizer is self-sufficient
    assert set(d["coeff"]["_meta"]) >= {"gids", "n_wires", "band_lengths", "norm_sigma"}
    np.testing.assert_array_equal(d["coeff"]["band"], events[0]["band"])
    assert d["coeff"]["value"].shape == (len(events[0]["value"]), 1)
    np.testing.assert_array_equal(ds.band_lengths, BAND_LENGTHS)


def test_sample_carries_source_identity(tmp_path):
    """``ident`` maps a coeff event back to the simulation event it came from.

    The corpus stores ``/ident`` (run, source_file, event) but the reader only
    ever consumed it as a packed 64-bit id for aligning coeff<->coeff_clean. That
    packing is deliberately lossy — a hash once a corpus spans several runs — so
    nothing downstream could recover the source event. Anything that wants to
    attach truth to a coeff event (a probe target, a per-cell label) needs the
    triple, and without it the only alternative is rebuilding the corpus to carry
    truth inline.
    """
    _write_shard(tmp_path / "cx_coeff_0000.h5")
    ds = CoeffTPCDataset(data_root=str(tmp_path), dataset_name="cx")

    seen = set()
    for i in range(len(ds)):
        ident = ds.get_data(i)["ident"]
        assert set(ident) == {"run", "source_file", "event"}
        assert isinstance(ident["event"], int)
        assert isinstance(ident["source_file"], str)
        seen.add((ident["run"], ident["source_file"], ident["event"]))
    assert len(seen) == len(ds), "identities must distinguish every event"

    # It identifies the SAMPLE, so it must not be mistaken for a row-space.
    assert "ident" not in ds.get_data(0)["coeff"]


def test_split_selects_the_events_actually_READ(tmp_path):
    """Assert on what the reader returns, not on what get_data_list says.

    The first version of this feature verified `get_data_list()` and declared
    success. But `get_data` did `real_idx = idx % len(self.data_list)` and passed
    that to the reader — it never dereferenced the list. So `data_list` said
    [18000, 18001, ...] while the events READ were [0, 1, ...]: the split was
    inert, train and val returned the same events, and the "validation" loss was
    still a training loss. Every assertion that looked at the list passed.

    Index == value held for every dataset until the first non-identity list, so
    the bug was latent in the base contract rather than in one dataset.
    """
    _write_shard(tmp_path / "cx_coeff_0000.h5", n_events=10)
    ident = lambda ds, i: ds.get_data(i)["ident"]["event"]

    head = CoeffTPCDataset(data_root=str(tmp_path), dataset_name="cx",
                           event_range=(0, 4))
    tail = CoeffTPCDataset(data_root=str(tmp_path), dataset_name="cx",
                           event_range=(6, 10))
    assert [ident(head, i) for i in range(len(head))] == [0, 1, 2, 3]
    assert [ident(tail, i) for i in range(len(tail))] == [6, 7, 8, 9]
    assert not ({ident(head, i) for i in range(len(head))} &
                {ident(tail, i) for i in range(len(tail))})

    holed = CoeffTPCDataset(data_root=str(tmp_path), dataset_name="cx",
                            event_range=(0, 10), exclude_range=(3, 6))
    assert [ident(holed, i) for i in range(len(holed))] == [0, 1, 2, 6, 7, 8, 9]

    with pytest.raises(ValueError, match="selects NO events"):
        CoeffTPCDataset(data_root=str(tmp_path), dataset_name="cx",
                        event_range=(0, 10), exclude_range=(0, 10))


def test_dataset_collect(tmp_path):
    _write_shard(tmp_path / "cx_coeff_0000.h5")
    collect = dict(type="Collect", parts={"coeff": dict(
        keys=("band", "plane_gid", "wire", "tau", "value"), feat_keys=("value",),
        offset_keys_dict=dict(offset="band"))})
    ds = CoeffTPCDataset(data_root=str(tmp_path), dataset_name="cx", transform=[collect])
    out = ds[0]
    assert "coeff_offset" in out and "coeff_value" in out
    n = int(out["coeff_offset"][-1])
    assert out["coeff_value"].shape[0] == n
    assert out["coeff_band"].shape[0] == n


# ── cross-repo golden: helix reference codec ⇄ pimm-data reader ──────────────

# The helix checkout to cross-check against. It pointed at a THIRD checkout
# (helix-consolidate) that no longer carries this code, so every test guarded by
# it skipped — silently, since a skip for "helix not importable" is
# indistinguishable from a machine that simply has no helix. Overridable so this
# does not rot again the next time the checkout moves.
_HELIX = os.environ.get("HELIX_ROOT", "/sdf/group/neutrino/omara/helix-extraction")


def _import_helix():
    if _HELIX not in sys.path:
        sys.path.insert(0, _HELIX)
    try:
        import helix.core.coeff_io  # noqa: F401
        return True
    except Exception:
        return False


@pytest.mark.skipif(not os.path.isdir(_HELIX) or not _import_helix(),
                    reason="helix (consolidation worktree) not importable")
def test_cross_repo_golden(tmp_path):
    from helix.core.backend import set_backend
    from helix.core.provenance import BasisDescriptor, derive_band_lengths
    from helix.core.wavelet import SparseResult
    from helix.core.coeff_event import CoeffEvent
    from helix.core.coeff_io import write_coeff_shard as hx_write
    set_backend("numpy")

    wl, lvl, mode, nt = "db2", 2, "periodization", 32
    bl = derive_band_lengths(wl, lvl, mode, nt)
    basis = BasisDescriptor(wavelet=wl, level=lvl, mode=mode, n_ticks_raw=nt, pad=0,
                            band_lengths=bl, removal=dict(kind="gate"),
                            threshold=dict(method="universal"), sigma_norm=2.6)
    rng = np.random.default_rng(7)
    results = {}
    for gid, nw in ((0, 4), (2, 3)):
        coeffs = []
        for L in bl:
            c = rng.standard_normal((nw, L)).astype(np.float32)
            c[rng.random((nw, L)) > 0.4] = 0.0
            coeffs.append(c)
        results[gid] = SparseResult(coeffs=coeffs, n_kept=0, n_total=0,
                                    sigma_per_band=np.ones(len(bl), np.float32),
                                    wavelet=wl, level=lvl, mode=mode)
    ce = CoeffEvent.from_sparse_results(results, basis=basis, run="run_000",
                                        source_file="cx_sensor_0000.h5", event=0)
    # helix WRITES the shard; pimm-data READS it
    path = tmp_path / "cx_coeff_0000.h5"
    hx_write(path, [ce], dataset_name="cx")
    r = CoeffTPCReader(data_root=str(tmp_path), dataset_name="cx")
    got = r.read_event(0)
    for k in ("band", "plane_gid", "wire", "tau", "value"):
        np.testing.assert_array_equal(got[k], getattr(ce, k), err_msg=f"golden {k}")
    np.testing.assert_array_equal(r.band_lengths, np.asarray(bl, np.int32))


# ── regression: audit fixes ─────────────────────────────────────────────────

def _tagged(path, event_ids, value_offset=0.0, gids=GIDS, nwires=N_WIRES):
    """One coeff per event; value encodes (event id + offset) so pairing is checkable."""
    events = [dict(
        band=np.array([1], np.uint8), plane_gid=np.array([gids[0]], np.int32),
        wire=np.array([0], np.int32), tau=np.array([0], np.int32),
        value=np.array([ev + value_offset], np.float32),
        sigma_threshold=np.ones((len(gids), len(BAND_LENGTHS)), np.float32),
        run="run_000", source_file="cx_sensor_0000.h5", event=ev) for ev in event_ids]
    write_coeff_shard(path, events, band_lengths=BAND_LENGTHS, gids=gids,
                      n_wires=nwires, basis_attrs=BASIS_ATTRS, dataset_name="cx")


def test_glob_no_coeff_clean_collision(tmp_path):
    _write_shard(tmp_path / "cx_coeff_0000.h5", n_events=3)
    _write_shard(tmp_path / "cx_coeff_clean_0000.h5", n_events=3)   # same dir
    assert len(CoeffTPCReader(data_root=str(tmp_path), dataset_name="cx", modality="coeff")) == 3
    assert len(CoeffTPCReader(data_root=str(tmp_path), dataset_name="cx", modality="coeff_clean")) == 3
    assert len(CoeffTPCDataset(data_root=str(tmp_path), dataset_name="cx")) == 3


def test_join_by_identity_not_position(tmp_path):
    _tagged(tmp_path / "cx_coeff_0000.h5", [10, 11, 12], value_offset=0.0)
    _tagged(tmp_path / "cx_coeff_clean_0000.h5", [10, 12], value_offset=0.5)   # event 11 dropped
    ds = CoeffTPCDataset(data_root=str(tmp_path), dataset_name="cx",
                         modalities=("coeff", "coeff_clean"), strict_lengths=False)
    assert len(ds) == 2                                    # intersection {10,12}
    for i in range(2):
        d = ds.get_data(i)
        nv = float(d["coeff"]["value"][0, 0])
        cv = float(d["coeff_clean"]["value"][0, 0])
        assert cv - nv == 0.5, f"idx {i}: noisy {nv} paired with clean {cv} (mispair)"


def test_writer_validates_basis(tmp_path):
    ev = [dict(band=np.array([1], np.uint8), plane_gid=np.array([0], np.int32),
               wire=np.array([0], np.int32), tau=np.array([0], np.int32),
               value=np.array([1.0], np.float32),
               sigma_threshold=np.ones((len(GIDS), len(BAND_LENGTHS)), np.float32),
               run="r", source_file="s.h5", event=0)]
    missing = {k: v for k, v in BASIS_ATTRS.items() if k != "pad"}
    with pytest.raises(ValueError, match="missing required"):
        write_coeff_shard(tmp_path / "x_coeff_0000.h5", ev, band_lengths=BAND_LENGTHS,
                          gids=GIDS, n_wires=N_WIRES, basis_attrs=missing, dataset_name="x")
    with pytest.raises(ValueError, match="inconsistent"):
        write_coeff_shard(tmp_path / "y_coeff_0000.h5", ev, band_lengths=[8, 8, 99],
                          gids=GIDS, n_wires=N_WIRES, basis_attrs=BASIS_ATTRS, dataset_name="y")


def test_plane_gid_beyond_255(tmp_path):
    _tagged(tmp_path / "cx_coeff_0000.h5", [0], gids=[300, 301])
    r = CoeffTPCReader(data_root=str(tmp_path), dataset_name="cx")
    assert int(r.read_event(0)["plane_gid"][0]) == 300      # no uint8 wrap


@pytest.mark.skipif(not os.path.isdir(_HELIX) or not _import_helix(),
                    reason="helix not importable")
def test_reverse_golden_pimm_write_helix_read(tmp_path):
    from helix.core.provenance import BasisDescriptor, derive_band_lengths
    from helix.core.coeff_io import read_coeff_event as hx_read
    bl = derive_band_lengths("db2", 2, "periodization", 32)      # == BAND_LENGTHS
    basis = BasisDescriptor(wavelet="db2", level=2, mode="periodization", n_ticks_raw=32,
                            pad=0, band_lengths=bl, removal={}, threshold={}, sigma_norm=2.6)
    battrs = dict(wavelet="db2", dwt_level=2, dwt_mode="periodization", n_ticks_raw=32,
                  pad=0, sigma_norm=2.6, basis_digest=basis.digest(),
                  removal_json="{}", threshold_json="{}")
    ev = [dict(band=np.array([1], np.uint8), plane_gid=np.array([0], np.int32),
               wire=np.array([0], np.int32), tau=np.array([0], np.int32),
               value=np.array([3.0], np.float32),
               sigma_threshold=np.ones((len(GIDS), len(bl)), np.float32),
               run="r", source_file="s.h5", event=0)]
    write_coeff_shard(tmp_path / "cx_coeff_0000.h5", ev, band_lengths=list(bl),
                      gids=GIDS, n_wires=N_WIRES, basis_attrs=battrs, dataset_name="cx")
    ce = hx_read(tmp_path / "cx_coeff_0000.h5", 0)              # helix reads + validates
    assert float(ce.value[0]) == 3.0 and int(ce.plane_gid[0]) == 0


# ── multi-shard: the seam every earlier test missed (all used one _0000.h5) ──

def _ev(rng, ev, n=6, gids=GIDS, nb=len(BAND_LENGTHS)):
    band = rng.integers(0, nb, n).astype(np.uint8)
    gid = rng.choice(np.asarray(gids), n).astype(np.int32)
    wire = rng.integers(0, 4, n).astype(np.int32)
    tau = np.array([rng.integers(0, BAND_LENGTHS[b]) for b in band], np.int32)
    # unique coords: a real CoeffEvent has one value per (gid,band,wire,tau)
    key = (gid.astype(np.int64) << 40) | (band.astype(np.int64) << 36) \
        | (wire.astype(np.int64) << 18) | tau
    _, keep = np.unique(key, return_index=True)
    keep = np.sort(keep)
    return dict(band=band[keep], plane_gid=gid[keep], wire=wire[keep], tau=tau[keep],
                value=rng.standard_normal(keep.size).astype(np.float32),
                sigma_threshold=np.full((len(gids), nb), 1.5, np.float32),
                run="run_000", source_file=f"cx_sensor_{ev // 2:04d}.h5", event=ev)


#: Fixtures write real provenance by DEFAULT. They used to write none, so every
#: fixture-based test of the verifier silently exercised the legacy
#: no-provenance path — including the cross-shard-damage test, which asserted a
#: clean corpus while the `code` comparison it was meant to reach was skipped.
FIXTURE_PROV = dict(geom="cubic_wireplane_geometry.json", geom_sha256="a" * 8,
                    spectrum="microboone", spectrum_sha256="c" * 8,
                    code=dict(version="0.0.0-test", git="0" * 40, git_dirty=False))


def _write_pair(dirpath, event_ids, file_index=0, norm_sigma=None, seed=0,
                gids=GIDS, values_only_clean=True, provenance=FIXTURE_PROV,
                basis_attrs=None):
    """A noisy shard + its co-supported clean target, as build_corpus writes them."""
    rng = np.random.default_rng(seed)
    evs = [_ev(rng, e, gids=gids) for e in event_ids]
    kw = dict(band_lengths=BAND_LENGTHS, gids=gids, n_wires=N_WIRES,
              basis_attrs=basis_attrs or BASIS_ATTRS, dataset_name="cx",
              file_index=file_index,
              norm_sigma=norm_sigma, provenance=provenance)
    write_coeff_shard(dirpath / f"cx_coeff_{file_index:04d}.h5", evs, **kw)
    clean = [{**e, "value": e["value"] * 0.5} for e in evs]      # same coords
    write_coeff_shard(dirpath / f"cx_coeff_clean_{file_index:04d}.h5", clean,
                      coords=not values_only_clean, **kw)
    return evs


def test_multi_shard_indexing_and_identity(tmp_path):
    """Two shards per modality. Every earlier test wrote exactly one, so the
    global index across shards was entirely unexercised."""
    ns = np.full((len(GIDS), len(BAND_LENGTHS)), 2.0, np.float32)
    a = _write_pair(tmp_path, [0, 1, 2], file_index=0, norm_sigma=ns, seed=1)
    b = _write_pair(tmp_path, [3, 4], file_index=1, norm_sigma=ns, seed=2)
    ds = CoeffTPCDataset(data_root=str(tmp_path), dataset_name="cx",
                         modalities=("coeff", "coeff_clean"))
    assert len(ds) == 5
    for i, want in enumerate(a + b):
        d = ds.get_data(i)
        np.testing.assert_array_equal(d["coeff"]["band"], want["band"],
                                      err_msg=f"event {i} crosses a shard boundary wrong")
        # the clean target is values-only: coords must come from the noisy side
        np.testing.assert_array_equal(d["coeff_clean"]["wire"], want["wire"])
        np.testing.assert_allclose(d["coeff_clean"]["value"][:, 0],
                                   want["value"] * 0.5, rtol=1e-6)


def test_norm_sigma_must_be_frozen_across_shards(tmp_path):
    """norm_sigma is corpus-wide: row i normalises any event in any shard. A
    per-shard table means the same physical coefficient is scaled differently
    depending on which shard it landed in — and the reader used to serve shard
    0's table for every event, silently."""
    ns0 = np.full((len(GIDS), len(BAND_LENGTHS)), 2.0, np.float32)
    ns1 = np.full((len(GIDS), len(BAND_LENGTHS)), 9.0, np.float32)     # disagrees
    _write_pair(tmp_path, [0, 1], file_index=0, norm_sigma=ns0, seed=1)
    _write_pair(tmp_path, [2, 3], file_index=1, norm_sigma=ns1, seed=2)
    with pytest.raises(ValueError, match="disagree on /config/norm_sigma"):
        CoeffTPCDataset(data_root=str(tmp_path), dataset_name="cx")

    # and the frozen case is accepted, serving the one true table
    ok = tmp_path / "ok"; ok.mkdir()
    _write_pair(ok, [0, 1], file_index=0, norm_sigma=ns0, seed=1)
    _write_pair(ok, [2, 3], file_index=1, norm_sigma=ns0, seed=2)
    ds = CoeffTPCDataset(data_root=str(ok), dataset_name="cx")
    np.testing.assert_allclose(ds.norm_sigma, ns0)


def test_shards_disagreeing_on_plane_set_are_rejected(tmp_path):
    _write_pair(tmp_path, [0, 1], file_index=0, seed=1)
    _write_pair(tmp_path, [2, 3], file_index=1, seed=2, gids=[7, 9])
    with pytest.raises(ValueError, match="disagree on /config/gids"):
        CoeffTPCDataset(data_root=str(tmp_path), dataset_name="cx")


def test_values_only_clean_shard_saves_space_and_detects_mispairing(tmp_path):
    import h5py
    _write_pair(tmp_path, [0, 1, 2], seed=3)
    p_noisy = tmp_path / "cx_coeff_0000.h5"
    p_clean = tmp_path / "cx_coeff_clean_0000.h5"
    with h5py.File(p_clean, "r") as f:
        assert not bool(f["config"].attrs["has_coords"])
        assert "band" not in f["coord"] and "coord_digest" in f["coord"]
    assert p_clean.stat().st_size < p_noisy.stat().st_size

    # swap in a clean shard built from DIFFERENT coords: same row counts, so only
    # the digest can catch it
    other = tmp_path / "other"; other.mkdir()
    _write_pair(other, [0, 1, 2], seed=99)
    import shutil
    shutil.copy(other / "cx_coeff_clean_0000.h5", p_clean)
    ds = CoeffTPCDataset(data_root=str(tmp_path), dataset_name="cx",
                         modalities=("coeff", "coeff_clean"))
    with pytest.raises(ValueError, match="coord_digest mismatch|not co-supported"):
        ds.get_data(0)


def test_clean_only_modality_reports_the_dependency(tmp_path):
    """A values-only target cannot be read alone — say so, rather than yielding
    rows with no coords."""
    _write_pair(tmp_path, [0, 1], seed=4)
    ds = CoeffTPCDataset(data_root=str(tmp_path), dataset_name="cx",
                         modalities=("coeff_clean",))
    with pytest.raises(ValueError, match="cannot be read alone"):
        ds.get_data(0)


def test_writer_rejects_a_shard_helix_would_refuse(tmp_path):
    """pimm-data's writer must reject anything helix's reader rejects, or the
    cross-repo contract is enforced in one direction only."""
    rng = np.random.default_rng(0)
    ev = [_ev(rng, 0)]
    kw = dict(band_lengths=BAND_LENGTHS, gids=GIDS, n_wires=N_WIRES, dataset_name="x")
    with pytest.raises(ValueError, match="basis_digest"):
        write_coeff_shard(tmp_path / "x_coeff_0000.h5", ev,
                          basis_attrs={**BASIS_ATTRS, "basis_digest": "deadbeef"}, **kw)
    with pytest.raises(ValueError, match="not parseable JSON"):
        write_coeff_shard(tmp_path / "y_coeff_0000.h5", ev,
                          basis_attrs={**BASIS_ATTRS, "removal_json": "{not json"}, **kw)


@pytest.mark.skipif(not os.path.isdir(_HELIX) or not _import_helix(),
                    reason="helix not importable")
def test_full_chain_to_model_keys_at_batch_gt_1(tmp_path):
    """CoeffTPCDataset -> CoeffTokenize -> Collect -> collate_fn, batch=2.

    Nothing exercised this end to end, which is why two separate defects lived
    here undetected: `to_fm` was written but never called (so the model's very
    first access, B["band_id"], raised KeyError), and the canonical Collect
    recipe in coeff.py's own docstring uses the multi-part `parts=` form, which
    PREFIXES every key to `coeff_band_id` and breaks the model contract just as
    surely. The single-part `Collect(part=...)` form is the correct one here.

    `cell`/`slot` are deliberately NOT collected. They are gather indices into
    the cell grid, so concatenating two events leaves the second event's indices
    pointing into the first event's cells — and no `_roles` kind rebases an index
    by another row-space's running count. They are needed only by the `--fused 0`
    ablation loss; the production loss (`losses_fused`, `--fused` defaults to 1)
    reads the dense (n_cells, n_slot) grids and per-cell scalars collected here.
    """
    import torch
    from torch.utils.data import DataLoader
    from helix.model.tokenize import CoeffTokenize
    from pimm_data.collate import collate_fn

    # a real corpus always carries norm_sigma (build_corpus always writes the
    # computed table); without it the tokenizer correctly refuses to normalise.
    ns = np.full((len(GIDS), len(BAND_LENGTHS)), 2.0, np.float32)
    _write_pair(tmp_path, [0, 1, 2, 3], seed=11, norm_sigma=ns)
    tok = CoeffTokenize(part="coeff", clean_part="coeff_clean")
    # offset counts CELLS (the row-space the model's dense grids live in), not
    # coefficient rows — the bare form otherwise defaults to a 'coord' key that
    # a tokenized part does not have.
    collect = dict(type="Collect", part="coeff", keys=(
        "band_id", "plane_id", "t_phys", "wire_pos", "wirefeat",
        "inp", "occ", "valid", "tgt"),
        offset_keys_dict=dict(offset="band_id"))
    ds = CoeffTPCDataset(data_root=str(tmp_path), dataset_name="cx",
                         modalities=("coeff", "coeff_clean"),
                         transform=[tok, collect])

    # the five keys that were missing entirely are present, and UNPREFIXED
    one = ds[0]
    for k in ("band_id", "plane_id", "t_phys", "wire_pos", "wirefeat"):
        assert k in one, f"{k} missing — the model gathers this by exact name"
    assert not any(k.startswith("coeff_") for k in one), \
        "single-part Collect must not prefix; the model uses bare names"
    assert one["wirefeat"].shape == (one["band_id"].shape[0], 1)

    # and a real DataLoader at batch=2 collates without exploding
    dl = DataLoader(ds, batch_size=2, num_workers=0, collate_fn=collate_fn)
    b = next(iter(dl))
    for k in ("band_id", "plane_id", "t_phys", "wire_pos", "wirefeat", "inp", "occ"):
        assert k in b
    # cell-space keys share one row count; the dense (n_cells, n_slot) grids the
    # PRODUCTION loss (losses_fused) reads line up with the per-cell scalars.
    n_cells = b["band_id"].shape[0]
    assert b["plane_id"].shape[0] == n_cells
    assert b["wirefeat"].shape == (n_cells, 1)
    assert b["inp"].shape[0] == n_cells and b["occ"].shape[0] == n_cells
    assert b["valid"].shape == b["inp"].shape
    # n_cells is deliberately not carried in the batch: post-collate it is just
    # the cell row count, which is what fm/train.py's mask generator needs.
    assert n_cells == b["inp"].shape[0]


def test_identity_is_source_file_plus_event_not_event_alone(tmp_path):
    """`/ident/event` is the event's index WITHIN its source file, so it restarts
    at 0 in every file. A shard that spans more than one source file therefore
    holds repeated ids — which is exactly what loader mode produces, since it
    reads a run-wide joint index that crosses file boundaries.

    Keying the id->position map on the event alone silently collapses them: the
    1000-event pilot came back as 800 joint events, with 20% of the corpus
    unreachable and no error anywhere. This only appeared once provenance was
    fixed to record the TRUE per-file event index; the previous fabricated
    sequential counter was wrong but happened to be unique.
    """
    rng = np.random.default_rng(5)
    # one shard, two source files, event ids 0,1 repeated in BOTH
    evs = []
    for src in ("cx_sensor_0000.h5", "cx_sensor_0001.h5"):
        for ev in (0, 1):
            e = _ev(rng, ev)
            e["source_file"] = src
            e["value"] = e["value"] + (0.0 if src.endswith("0000.h5") else 100.0)
            evs.append(e)
    kw = dict(band_lengths=BAND_LENGTHS, gids=GIDS, n_wires=N_WIRES,
              basis_attrs=BASIS_ATTRS, dataset_name="cx",
              norm_sigma=np.full((len(GIDS), len(BAND_LENGTHS)), 2.0, np.float32))
    write_coeff_shard(tmp_path / "cx_coeff_0000.h5", evs, **kw)
    write_coeff_shard(tmp_path / "cx_coeff_clean_0000.h5",
                      [{**e, "value": e["value"] * 0.5} for e in evs],
                      coords=False, **kw)

    r = CoeffTPCReader(data_root=str(tmp_path), dataset_name="cx")
    assert len(r) == 4, f"expected 4 events, got {len(r)} — duplicate ids collapsed"
    ds = CoeffTPCDataset(data_root=str(tmp_path), dataset_name="cx",
                         modalities=("coeff", "coeff_clean"))
    assert len(ds) == 4

    # every event is distinct and correctly paired: the +100 offset identifies
    # which source file a row came from, so a collapsed id would show up as a
    # duplicate value block rather than four distinct ones.
    firsts = sorted(float(ds.get_data(i)["coeff"]["value"][0, 0]) for i in range(4))
    assert len({round(v, 5) for v in firsts}) == 4, f"events not distinct: {firsts}"
    for i in range(4):
        d = ds.get_data(i)
        np.testing.assert_allclose(d["coeff_clean"]["value"][:, 0],
                                   d["coeff"]["value"][:, 0] * 0.5, rtol=1e-5)


def test_identity_includes_run_so_runs_do_not_collide():
    """Every detector run restarts shard numbering at *_sensor_0000.h5, so
    (source_file, event) alone collides across runs — the design doc specifies
    (run, source_file, event). Same 20%-loss class as the within-shard collision."""
    from pimm_data.readers.coeff_tpc import _identity_ids
    src = np.array([b"sim_wire_sensor_0000.h5"] * 3)
    ev = np.array([0, 1, 2], np.int64)
    a = _identity_ids(src, ev, run=np.array([b"run_A"] * 3))
    b = _identity_ids(src, ev, run=np.array([b"run_B"] * 3))
    assert not set(a.tolist()) & set(b.tolist()), "two runs produced colliding ids"
    # single-run stays ordered (iteration follows the data, not a hash)
    # ordering WITHIN a run is preserved (the run sits in the high bits), so the
    # dataset's index order still follows the data
    one = _identity_ids(np.array([b"s_0000.h5", b"s_0000.h5", b"s_0001.h5"]),
                        np.array([0, 1, 0], np.int64), run=np.array([b"r"] * 3))
    assert list(one) == sorted(one), "within-run ids must stay ordered"


def test_reader_rejects_a_shard_with_duplicate_identities(tmp_path):
    """A shard whose /ident does not uniquely identify its events cannot be
    indexed — the id->position map keeps one row per duplicate. Fail loudly
    instead of serving a quietly shorter dataset."""
    rng = np.random.default_rng(3)
    evs = []
    for _ in range(2):                       # SAME (run, source_file, event) twice
        e = _ev(rng, 7)
        e["source_file"] = "cx_sensor_0000.h5"
        evs.append(e)
    write_coeff_shard(tmp_path / "cx_coeff_0000.h5", evs, band_lengths=BAND_LENGTHS,
                      gids=GIDS, n_wires=N_WIRES, basis_attrs=BASIS_ATTRS,
                      dataset_name="cx")
    with pytest.raises(ValueError, match="does not uniquely identify"):
        CoeffTPCReader(data_root=str(tmp_path), dataset_name="cx")


PROV = FIXTURE_PROV


def test_verifier_compares_the_dsp_inputs(tmp_path):
    """A corpus mixing geometries or noise spectra is not one corpus.

    Both hashes are written by the corpus builder and neither was compared —
    ``verify_corpus`` pulled only ``code`` out of ``provenance_json``, while its
    own message claimed "basis_digest and the geom/spectrum hashes pin the DSP's
    inputs". A corpus assembled from shards built against different geometry, or
    against a different (or white) noise spectrum, passed clean. The
    coefficients are not comparable across such a split, and nothing else
    downstream can see it.

    This also exercises the writer's new ``provenance=``: it did not write
    ``provenance_json`` at all, so every fixture-based test of these checks was
    silently testing the legacy no-provenance path.
    """
    ns = np.full((len(GIDS), len(BAND_LENGTHS)), 2.0, np.float32)

    ok = tmp_path / "ok"; ok.mkdir()
    for k, ids in enumerate(([0, 1], [2, 3])):
        _write_pair(ok, ids, file_index=k, norm_sigma=ns, seed=k, provenance=PROV)
    assert verify_corpus(str(ok), "cx", expect_events=4) == []

    for name, second in (("geom", dict(PROV, geom="OTHER.json", geom_sha256="b" * 8)),
                         ("spectrum", dict(PROV, spectrum=None, spectrum_sha256=None))):
        d = tmp_path / name; d.mkdir()
        _write_pair(d, [0, 1], file_index=0, norm_sigma=ns, seed=0, provenance=PROV)
        _write_pair(d, [2, 3], file_index=1, norm_sigma=ns, seed=1, provenance=second)
        probs = verify_corpus(str(d), "cx")
        assert any("DSP inputs" in p for p in probs), f"{name}: {probs}"

    # Absence stays LEGACY, not disagreement — a pre-provenance corpus must not
    # be flagged as mixed, or the real signal drowns.
    legacy = tmp_path / "legacy"; legacy.mkdir()
    for k, ids in enumerate(([0, 1], [2, 3])):
        _write_pair(legacy, ids, file_index=k, norm_sigma=ns, seed=k)
    assert not any("DSP inputs" in p for p in verify_corpus(str(legacy), "cx"))


def test_corpus_verifier_catches_cross_shard_damage(tmp_path):
    """audit_shard is per-file and structurally cannot see these. Each one leaves
    every individual shard valid while silently changing how much data the corpus
    holds — which is exactly what an OOM-killed build produced."""
    from pimm_data.coeff_verify import verify_corpus
    ns = np.full((len(GIDS), len(BAND_LENGTHS)), 2.0, np.float32)
    for k, ids in enumerate(([0, 1], [2, 3], [4, 5])):
        _write_pair(tmp_path, ids, file_index=k, norm_sigma=ns, seed=k)
    assert verify_corpus(str(tmp_path), "cx", expect_events=6) == []

    # 1. a job died after writing only its noisy shard
    (tmp_path / "cx_coeff_clean_0001.h5").unlink()
    probs = verify_corpus(str(tmp_path), "cx")
    assert any("no coeff_clean pair" in p for p in probs), probs

    # 2. a shard missing from the middle of the run
    (tmp_path / "cx_coeff_0001.h5").unlink()
    assert any("file_index gap" in p for p in verify_corpus(str(tmp_path), "cx")), \
        verify_corpus(str(tmp_path), "cx")

    # 3. shards disagreeing on the frozen table
    other = tmp_path / "other"; other.mkdir()
    _write_pair(other, [0, 1], file_index=0, norm_sigma=ns, seed=9)
    _write_pair(other, [2, 3], file_index=1,
                norm_sigma=np.full_like(ns, 9.0), seed=9)
    assert any("norm_sigma" in p for p in verify_corpus(str(other), "cx")), \
        verify_corpus(str(other), "cx")

    # 4. overlapping --event-start ranges duplicating events across shards
    dup = tmp_path / "dup"; dup.mkdir()
    _write_pair(dup, [0, 1, 2], file_index=0, norm_sigma=ns, seed=4)
    _write_pair(dup, [2, 3, 4], file_index=1, norm_sigma=ns, seed=4)   # 2 repeats
    assert any("more than once" in p for p in verify_corpus(str(dup), "cx")), \
        verify_corpus(str(dup), "cx")


def test_verifier_rejects_a_corpus_that_records_no_builder(tmp_path):
    """A corpus where NOTHING records its builder must FAIL, not pass.

    The `code` comparison only looked at shards that carry the field, so a
    corpus with zero of them skipped the check and reported clean. Measured on
    the real corpus: 0 of 100 shards carry it, verify_corpus said OK, and a
    surviving build log shows 12 shards were built from a different worktree.
    """
    ns = np.full((len(GIDS), len(BAND_LENGTHS)), 2.0, np.float32)
    for k, ids in enumerate(([0, 1], [2, 3])):
        _write_pair(tmp_path, ids, file_index=k, norm_sigma=ns, seed=k,
                    provenance=None)                       # legacy: no provenance
    probs = verify_corpus(str(tmp_path), "cx")
    assert any("provenance['code']" in p for p in probs), probs


HOLDOUT = dict(seed=0, fractions=dict(train=0.6, val=0.2, probe=0.2))


def _split_map(root, role):
    ds = CoeffTPCDataset(data_root=str(root), dataset_name="cx",
                         holdout=HOLDOUT, split_role=role)
    return {(d["run"], d["source_file"], d["event"]) for d in ds.holdout_manifest()}


def test_holdout_survives_corpus_growth(tmp_path):
    """Adding a shard must not move any existing event between splits.

    This is the whole reason the split is keyed on identity rather than
    position. A positional range (`event_range=(17700, 18000)`) is a valid split
    exactly once: the joint index is built over a GLOB, so one more shard
    renumbers everything after it and a probe holdout silently becomes a set of
    trained-on events. The coeff corpus is explicitly built to grow — one run
    now, eight planned — so this is the failure that would actually happen.
    """
    # 40 events, not a handful: at 20% a six-event fixture can legitimately put
    # ZERO in probe, and the empty-split guard (correctly) refuses that.
    _write_pair(tmp_path, list(range(40)), file_index=0, seed=0)
    before = {r: _split_map(tmp_path, r) for r in ("train", "val", "probe")}
    assert sum(len(v) for v in before.values()) == 40     # every event placed

    # Growth means NEW events. Re-writing the same ids would give two shards
    # identical (run, source_file, event) identities, which the reader's own
    # /ident uniqueness check exists to reject.
    _write_pair(tmp_path, list(range(40, 80)), file_index=1, seed=1)
    after = {r: _split_map(tmp_path, r) for r in ("train", "val", "probe")}
    assert sum(len(v) for v in after.values()) == 80

    for role in before:
        assert before[role] <= after[role], (
            f"{role}: events changed split when a shard was added — "
            f"moved out: {before[role] - after[role]}")


def test_holdout_splits_are_disjoint_and_total(tmp_path):
    _write_pair(tmp_path, list(range(40)), file_index=0, seed=3)
    m = {r: _split_map(tmp_path, r) for r in ("train", "val", "probe")}
    allv = [e for v in m.values() for e in v]
    assert len(allv) == len(set(allv)) == 40          # disjoint AND complete


def test_holdout_requires_a_role_and_valid_fractions(tmp_path):
    _write_pair(tmp_path, list(range(40)), file_index=0)
    with pytest.raises(ValueError, match="split= was not"):
        CoeffTPCDataset(data_root=str(tmp_path), dataset_name="cx", holdout=HOLDOUT)
    with pytest.raises(ValueError, match="sum to 1.0"):
        CoeffTPCDataset(data_root=str(tmp_path), dataset_name="cx",
                        holdout=dict(seed=0, fractions=dict(train=0.5, val=0.2)),
                        split_role="train")


def test_shards_from_two_different_GATES_are_rejected(tmp_path):
    """The real corpus-mixing hazard, and the one every other check misses.

    /sdf/data/neutrino/omara/coeff_tpc and .../coeff_tpc_r1 differ ONLY in the
    coherent-removal gate: r1 records removal_json tau=0.05, the older tree
    records no tau at all. Their band_lengths, gids, n_wires, norm_sigma and
    sigma_norm are identical, so every pre-existing agreement check passes on a
    directory holding both — they would read as one corpus and train as two.

    Not hypothetical: submit_coeff_corpus.sh defaulted its output root to the
    older corpus while `--tau` unset already meant the newer gate, so the
    ordinary way to extend a corpus wrote r1 shards into the pre-r1 tree.
    """
    legacy = dict(BASIS_ATTRS)
    legacy["removal_json"] = json.dumps(dict(kind="gate", kgate=3.0))
    gated = dict(BASIS_ATTRS)
    gated["removal_json"] = json.dumps(dict(kind="gate", kgate=3.0, tau=0.05))

    _write_pair(tmp_path, [0, 1], file_index=0, seed=1, basis_attrs=legacy)
    _write_pair(tmp_path, [2, 3], file_index=1, seed=2, basis_attrs=gated)
    with pytest.raises(ValueError) as ei:
        CoeffTPCDataset(data_root=str(tmp_path), dataset_name="cx")
    msg = str(ei.value)
    assert "removal_json" in msg or "basis_digest" in msg, msg
    assert "different DSP" in msg, "the message must say WHY they are not one corpus"


def test_shards_with_a_different_WAVELET_are_rejected(tmp_path):
    """basis_digest covers the basis itself, not just the gate."""
    other = dict(BASIS_ATTRS)
    other["wavelet"] = "coif3"
    _write_pair(tmp_path, [0, 1], file_index=0, seed=1)
    _write_pair(tmp_path, [2, 3], file_index=1, seed=2, basis_attrs=other)
    with pytest.raises(ValueError, match="basis_digest"):
        CoeffTPCDataset(data_root=str(tmp_path), dataset_name="cx")


def test_one_gate_across_all_shards_is_accepted(tmp_path):
    """The guard must not refuse a good corpus — bytes-vs-str from h5py in
    particular, which is how a well-meant identity check rejects everything."""
    _write_pair(tmp_path, [0, 1], file_index=0, seed=1)
    _write_pair(tmp_path, [2, 3], file_index=1, seed=2)
    ds = CoeffTPCDataset(data_root=str(tmp_path), dataset_name="cx")
    assert len(ds) == 4


def test_a_gap_is_judged_against_the_SOURCE_not_assumed_to_be_a_failed_job(tmp_path):
    """A missing file_index means a dead build job — or a file that never existed.

    run_0027670361 has 90 source files upstream, not 100, so its corpus is
    COMPLETE with shards 51-56 and 94-97 absent. From inside the corpus that is
    indistinguishable from ten crashed jobs, and the difference is one directory
    away — so the verifier is given the source tree as evidence rather than a
    flag that would simply silence it. The result is a STRONGER check than the
    unconditional one: a gap whose source exists is still an error.
    """
    from pimm_data.coeff_verify import verify_corpus

    corpus = tmp_path / "corpus"; corpus.mkdir()
    src = tmp_path / "source"; src.mkdir()
    for i in (0, 1, 3):                      # shard 2 is missing
        _write_pair(corpus, [i * 2, i * 2 + 1], file_index=i, seed=i)

    # No source tree: the gap is reported, and the message says how to resolve it.
    probs = verify_corpus(str(corpus), "cx")
    assert any("file_index gap" in p for p in probs)
    assert any("--source-root" in p for p in probs), probs

    # Source for 2 ABSENT -> the corpus is complete.
    for i in (0, 1, 3):
        (src / f"cx_sensor_{i:04d}.h5").write_bytes(b"")
    assert verify_corpus(str(corpus), "cx", source_root=str(src)) == []

    # Source for 2 PRESENT -> still an error, and the message says so.
    (src / "cx_sensor_0002.h5").write_bytes(b"")
    probs = verify_corpus(str(corpus), "cx", source_root=str(src))
    assert any("source file IS present" in p for p in probs), probs
