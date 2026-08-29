"""Unit tests for the projection selection logic that iterated the most:
transitive locus chaining, window-entry fallback, and the one-contig
chromosome picker (rule, exclusion, thinning). All on a synthetic kit —
no sketch store needed."""

from types import SimpleNamespace

import pandas as pd

from kmer_clust.project import loci_of, whole_chrom_set, window_entry

BIN = 100_000


def fake_kit(n_chr1=50, n_chry=50):
    starts = [i * BIN for i in range(n_chr1)] + [i * BIN for i in range(n_chry)]
    bins = pd.DataFrame({
        "chrom": ["chr1"] * n_chr1 + ["chrY"] * n_chry,
        "start": starts,
        "end": [s + BIN for s in starts],
    })
    return SimpleNamespace(bins=bins, params=SimpleNamespace(bin_bp=BIN))


def rec(ehits, ejacc, hits=None, sims=None, cover=0.97):
    return {"cover": cover, "hits": hits or [], "sims": sims or [],
            "ehits": ehits, "ejacc": ejacc}


def test_loci_of_chains_transitively():
    kit = fake_kit()
    # four adjacent chrY bins interleaved (by score) with one chr1 bin:
    # they must merge into ONE locus, not fragments that evict chr1
    hits = [60, 62, 10, 61, 63]
    sims = [0.9, 0.85, 0.8, 0.7, 0.65]
    loci = loci_of(kit, hits, sims)
    assert len(loci) == 2
    assert loci[0]["chrom"] == "chrY" and len(loci[0]["bins"]) == 4
    assert loci[1]["chrom"] == "chr1"
    assert loci[0]["sim"] == 0.9  # locus carries its best member's score


def test_loci_of_splits_on_gap():
    kit = fake_kit()
    # gap of 10 bins > gap_bins=3 -> two distinct chrY loci
    loci = loci_of(kit, [60, 75], [0.9, 0.8])
    assert len(loci) == 2
    assert all(l["chrom"] == "chrY" for l in loci)


def test_window_entry_falls_back_to_exact_hits():
    kit = fake_kit()
    r = rec([60, 61], [0.5, 0.4])  # no model hits (satellite-poor window)
    e = window_entry(kit, "tigA", 7, r, 0.7)
    assert e["hits"] == [[60, 0.5], [61, 0.4]]
    assert e["pos_mb"] == 0.7
    assert e["label"].startswith("tigA:0.7")


def _results(spec):
    """spec: {contig: [(w, ehit), ...]} -> results list."""
    return [(c, w, rec([b], [0.8])) for c, ws in spec.items()
            for w, b in ws]


def test_whole_chrom_picker_prefers_hits_times_span():
    kit = fake_kit()
    # tigWide: 16 hits walking 30 bins of chrY (score 16*3.0)
    # tigDense: 20 hits walking 5 bins (score 20*0.5) -> tigWide wins
    spec = {
        "tigWide": [(i, 50 + 2 * i) for i in range(16)],
        "tigDense": [(i, 55 + (i % 6)) for i in range(20)],
    }
    st = whole_chrom_set(kit, _results(spec), "S1 pat", "chrY")
    assert st is not None
    assert st["windows"][0]["label"].startswith("tigWide:")
    assert st["id"] == "y_s1"


def test_whole_chrom_picker_respects_exclusion_and_min_hits():
    kit = fake_kit()
    spec = {
        "tigWide": [(i, 50 + 2 * i) for i in range(16)],
        "tigNext": [(i, 60 + i) for i in range(15)],
        "tigScrap": [(0, 51), (1, 52)],  # below min_hits
    }
    st = whole_chrom_set(kit, _results(spec), "S1 pat", "chrY",
                         exclude={"tigWide"})
    assert st["windows"][0]["label"].startswith("tigNext:")
    st2 = whole_chrom_set(kit, _results({"tigScrap": spec["tigScrap"]}),
                          "S1 pat", "chrY")
    assert st2 is None


def test_whole_chrom_thinning_keeps_span_and_discloses():
    kit = fake_kit()
    spec = {"tig": [(i, 50 + i // 2) for i in range(40)]}
    st = whole_chrom_set(kit, _results(spec), "S1 pat", "chrY", cap=10)
    assert st["n_win_all"] == 40
    assert len(st["windows"]) == 10
    # uniform thinning keeps both ends of the contig span
    assert st["windows"][0]["pos_mb"] == 0.0
    assert st["windows"][-1]["pos_mb"] == 3.9


# ---------------------------------------------------------------- triage
import numpy as np

from kmer_clust.project import triage_rows, triage_summary


def _tri_setup():
    # 2 chroms x 50 bins; satellite = chrY bins 30..49
    chrom_idx = np.array([0] * 50 + [1] * 50)
    bin_mb = np.array([i * 0.1 + 0.05 for i in range(50)] * 2)
    sat = np.zeros(100, bool); sat[80:] = True
    return chrom_idx, bin_mb, sat


def test_triage_forward_and_reverse_orientation():
    chrom_idx, bin_mb, sat = _tri_setup()
    fwd = [("tigF", w, rec([10 + w], [0.8])) for w in range(10)]
    rev = [("tigR", w, rec([40 - w], [0.8])) for w in range(10)]
    rows = triage_rows(chrom_idx, bin_mb, sat, fwd + rev, BIN)
    by = {r["contig"]: r for r in rows}
    assert by["tigF"]["orient"] == "forward" and by["tigF"]["tau"] == 1.0
    assert by["tigR"]["orient"] == "reverse" and by["tigR"]["tau"] == -1.0
    assert by["tigF"]["jumps"] == 0


def test_triage_jump_and_dominant_chrom():
    chrom_idx, bin_mb, sat = _tri_setup()
    # 6 windows on chr0 then 3 on chr1: one chromosome jump, chr0 dominant
    res = [("tigJ", w, rec([5 + w], [0.9])) for w in range(6)] + \
          [("tigJ", 6 + w, rec([60 + w], [0.9])) for w in range(3)]
    rows = triage_rows(chrom_idx, bin_mb, sat, res, BIN)
    r = rows[0]
    assert r["dom_chrom"] == 0 and r["jumps"] == 1
    assert r["end5"] == "non-sat"


def test_triage_novelty_run_and_sat_end():
    chrom_idx, bin_mb, sat = _tri_setup()
    res = []
    for w in range(12):
        cover = 0.5 if 4 <= w <= 8 else 0.99
        res.append(("tigN", w, rec([85 + (w % 3)], [0.8], cover=cover)))
    rows = triage_rows(chrom_idx, bin_mb, sat, res, BIN)
    r = rows[0]
    assert r["novel_run_mb0"] == 0.4 and r["novel_run_mb1"] == 0.9
    assert r["novel_run_mincov"] == 0.5
    assert r["end5"] == "satellite" and r["end3"] == "satellite"


def test_triage_summary_counts():
    chrom_idx, bin_mb, sat = _tri_setup()
    res = [("tigF", w, rec([10 + w], [0.8])) for w in range(10)]
    rows = triage_rows(chrom_idx, bin_mb, sat, res, BIN)
    s = triage_summary(rows, res, 100, BIN)
    assert s["n_windows"] == 10 and s["placed_confident"] == 1.0
    assert s["orientation_census"] == {"forward": 1}
    assert s["ends_in_satellite"] == 0.0
