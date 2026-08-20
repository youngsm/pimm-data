"""Corpus-level integrity check for a coeff corpus.

``helix.core.coeff_io.audit_shard`` is thorough but strictly PER FILE — it runs
inside the builder after every shard write and cannot see anything that spans
shards. Every failure the build actually produced in practice is cross-file:

- a job that died after writing its noisy shard, leaving no ``coeff_clean`` pair
  (observed: an OOM-killed 1000-event build left a complete, valid, readable
  1.36 GB noisy shard and nothing noticed);
- a shard missing from the middle of a run (a failed array task);
- overlapping ``--event-start`` ranges duplicating events across shards;
- shards that disagree on the frozen ``norm_sigma`` / basis / plane set.

None of these corrupt a file. Each one silently changes how much data the corpus
contains, which no per-file audit and no training run will ever report.

Usage::

    python -m pimm_data.coeff_verify <corpus_dir> --dataset-name sim_wire [--expect 1000]
"""

from __future__ import annotations

import glob
import json
import os
from collections import Counter

import numpy as np
import h5py


def verify_corpus(data_root, dataset_name, *, split="", expect_events=None,
                  modalities=("coeff", "coeff_clean"), source_root=None):
    """Check a built corpus for cross-shard problems. Returns a list of problem
    strings (empty == clean). Does not open the dataset; pure file inspection, so
    it works on a partially built corpus too."""
    probs = []
    base = os.path.join(data_root, split) if split else data_root

    def shards(mod):
        pat = os.path.join(base, f"{dataset_name}_{mod}_[0-9]*.h5")
        return sorted(glob.glob(pat))

    noisy = shards("coeff")
    if not noisy:
        return [f"no {dataset_name}_coeff_*.h5 shards under {base}"]

    def idx_of(p):
        return int(os.path.basename(p).rsplit("_", 1)[-1][:-3])

    # --- 1. pair completeness -------------------------------------------------
    if "coeff_clean" in modalities:
        n_idx = {idx_of(p) for p in noisy}
        c_idx = {idx_of(p) for p in shards("coeff_clean")}
        if n_idx - c_idx:
            probs.append(f"coeff shards with no coeff_clean pair: {sorted(n_idx - c_idx)}")
        if c_idx - n_idx:
            probs.append(f"coeff_clean shards with no coeff pair: {sorted(c_idx - n_idx)}")

    # --- 2. file_index contiguity --------------------------------------------
    # A gap usually means a build job died. It can also mean the SOURCE file was
    # never there: run_0027670361 upstream has 90 files, not 100, so shards
    # 51-56 and 94-97 can never exist and the corpus is complete without them.
    # Those two cases look identical from inside the corpus, and the difference
    # is one directory away — so `source_root` is asked for the evidence rather
    # than a flag being offered to silence the check. With it, a gap whose source
    # is ABSENT is fine and a gap whose source EXISTS is still an error, which is
    # a strictly stronger check than the unconditional one it replaces.
    have = sorted(idx_of(p) for p in noisy)
    missing = sorted(set(range(have[0], have[-1] + 1)) - set(have))
    if missing and source_root:
        src_dir = os.path.join(source_root, split) if split else source_root
        recoverable = [i for i in missing if os.path.exists(
            os.path.join(src_dir, f"{dataset_name}_sensor_{i:04d}.h5"))]
        upstream = [i for i in missing if i not in recoverable]
        if upstream:
            print(f"  note: shards {upstream} have no source under {src_dir} — "
                  f"absent upstream, not a failed build")
        missing = recoverable
    if missing:
        probs.append(f"file_index gap — shards {missing} absent between "
                     f"{have[0]} and {have[-1]} (a failed job leaves exactly this"
                     + ("; pass --source-root to distinguish an upstream gap)"
                        if not source_root else ", and the source file IS present)"))

    # --- 3. cross-shard agreement + identity uniqueness ----------------------
    ref = None
    ident = Counter()
    code = []
    dsp = []
    total = 0
    for p in noisy:
        try:
            with h5py.File(p, "r") as f:
                cfg = f["config"]
                cur = dict(
                    band_lengths=cfg["band_lengths"][:].tolist(),
                    gids=cfg["gids"][:].tolist(),
                    n_wires=cfg["n_wires"][:].tolist(),
                    norm_sigma=(cfg["norm_sigma"][:].tolist()
                                if "norm_sigma" in cfg else None),
                    basis_digest=str(cfg.attrs.get("basis_digest", "")),
                    noise_json=str(cfg.attrs.get("noise_json", "")),
                )
                # Which helix built this shard. Compared separately from the
                # fields above because absence is LEGACY, not disagreement:
                # shards predating the field simply have none, and flagging
                # every one of them would bury the case this exists to catch —
                # two shards of one corpus built by different code.
                _prov = json.loads(str(cfg.attrs.get("provenance_json", "{}")))
                code.append((os.path.basename(p), _prov.get("code")))
                # The DSP's INPUTS: which geometry file and which noise spectrum
                # produced this shard. Both hashes are already written by
                # build_coeff_corpus.py and were never compared — so a corpus
                # assembled from shards built against different geometry, or
                # against a different (or white) noise spectrum, passed clean.
                # Handled like `code`, not like the fields above: absence is
                # LEGACY, so only shards that carry them are compared.
                dsp.append((os.path.basename(p),
                            {k: _prov.get(k) for k in
                             ("geom", "geom_sha256", "spectrum", "spectrum_sha256")}
                            if any(k in _prov for k in
                                   ("geom_sha256", "spectrum_sha256")) else None))
                n = int(cfg.attrs["n_events"])
                total += n
                if "ident" in f:
                    runs = f["ident"]["run"][:]
                    srcs = f["ident"]["source_file"][:]
                    evs = f["ident"]["event"][:]
                    for r, s, e in zip(runs, srcs, evs):
                        r = r.decode() if isinstance(r, bytes) else str(r)
                        s = s.decode() if isinstance(s, bytes) else str(s)
                        ident[(r, s, int(e))] += 1
                else:
                    probs.append(f"{os.path.basename(p)}: no /ident group")
        except Exception as e:                       # unreadable/truncated
            probs.append(f"{os.path.basename(p)}: cannot read ({type(e).__name__}: {e})")
            continue
        if ref is None:
            ref, ref_p = cur, p
        else:
            for k in cur:
                if cur[k] != ref[k]:
                    probs.append(
                        f"shards disagree on {k}: {os.path.basename(ref_p)} vs "
                        f"{os.path.basename(p)}"
                        + ("  (norm_sigma must be FROZEN across a corpus — build "
                           "with --norm-sigma)" if k == "norm_sigma" else ""))

    # --- 3b. code version ----------------------------------------------------
    known = [(f, c) for f, c in code if c]
    if not known and code:
        # A corpus where NOTHING records its builder must fail, not pass. The
        # guard below only compares shards that carry `code`, so a corpus with
        # zero of them skipped the check entirely and reported clean — measured
        # on run_0027575715: 0 of 100 shards carry it, verify_corpus said OK,
        # and one surviving build log shows 12 shards were rebuilt from a
        # different worktree. "Which code produced this data" is the one
        # question a provenance check exists to answer.
        probs.append(
            f"NO shard records provenance['code'] ({len(code)} shards): the "
            f"corpus cannot say which helix built it, so a mixed-tree build is "
            f"undetectable. Rebuild with a current builder, or record the "
            f"provenance alongside the shards.")
    if known:
        seen = {json.dumps(c, sort_keys=True) for _, c in known}
        if len(seen) > 1:
            probs.append(
                f"shards were built by DIFFERENT helix versions: "
                + "; ".join(sorted(f"{v}" for v in seen))
                + "  (basis_digest and the geom/spectrum hashes pin the DSP's "
                  "inputs, not its code, so this is the only field that shows it)")
        elif any(c.get("git_dirty") for _, c in known):
            probs.append(
                f"corpus was built from a DIRTY working tree "
                f"({known[0][1].get('git')}) — the recorded commit does not "
                f"describe the code that ran")
        if len(known) != len(code):
            probs.append(
                f"{len(code) - len(known)} of {len(code)} shards predate the "
                f"provenance 'code' field (legacy); the rest record "
                f"{known[0][1].get('git', '?')}")

    # --- 3c. DSP inputs (geometry + noise spectrum) ---------------------------
    known_dsp = [(f, d) for f, d in dsp if d]
    if known_dsp:
        seen = {json.dumps(d, sort_keys=True) for _, d in known_dsp}
        if len(seen) > 1:
            probs.append(
                "shards were built against DIFFERENT DSP inputs (geometry or "
                "noise spectrum): " + "; ".join(sorted(seen))
                + "  (a corpus mixing geometries or noise models is not one "
                  "corpus; the coefficients are not comparable)")
        if len(known_dsp) != len(dsp):
            probs.append(
                f"{len(dsp) - len(known_dsp)} of {len(dsp)} shards predate the "
                f"geom/spectrum provenance hashes (legacy); the rest record "
                f"geom={known_dsp[0][1].get('geom')} "
                f"spectrum={known_dsp[0][1].get('spectrum')}")

    # --- 4. duplicate events -------------------------------------------------
    dups = [k for k, c in ident.items() if c > 1]
    if dups:
        probs.append(
            f"{len(dups)} event identities appear more than once (overlapping "
            f"--event-start ranges duplicate training data invisibly); first few: "
            f"{dups[:5]}")

    # --- 5. expected count ---------------------------------------------------
    if expect_events is not None and total != expect_events:
        probs.append(f"event count {total} != expected {expect_events}")

    return probs


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("data_root")
    ap.add_argument("--dataset-name", required=True)
    ap.add_argument("--split", default="")
    ap.add_argument("--expect", type=int, default=None,
                    help="expected total event count across the corpus")
    ap.add_argument("--modalities", nargs="*", default=["coeff", "coeff_clean"])
    ap.add_argument("--source-root", default=None,
                    help="the SENSOR tree the corpus was built from. Given, a "
                         "file_index gap is judged against it: absent upstream "
                         "is fine, present-but-unbuilt is still an error.")
    a = ap.parse_args(argv)
    probs = verify_corpus(a.data_root, a.dataset_name, split=a.split,
                          expect_events=a.expect, modalities=tuple(a.modalities),
                          source_root=a.source_root)
    if probs:
        print(f"CORPUS FAILED ({len(probs)} problem(s)):")
        for p in probs:
            print(f"  - {p}")
        return 1
    print("corpus OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
