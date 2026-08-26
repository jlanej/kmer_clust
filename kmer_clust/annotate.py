"""Stage: per-bin annotation from T2T tracks (censat, segdups, RepeatMasker, telomere).

Annotation is used only to ASSESS the map, never to build it. Censat class
normalization follows kmer_dust's vocabulary, including its live-HOR rule:
T2T v2.x spells an active alpha-satellite array as hor(...L) -- a suffix
family token ending in L.
"""

import re

import numpy as np
import pandas as pd

from .config import DATA, OUT, Params

_SUFFIX = re.compile(r"(?:_(?:\d+|[XYMxym]))+$")

CENSAT_CLASSES = [
    "asat_hor_live", "asat_hor", "asat_dhor", "asat_mon",
    "hsat1A", "hsat1B", "hsat2", "hsat3",
    "bsat", "gsat", "other_sat", "rDNA", "ct",
]


def censat_class(name: str) -> str:
    head, _, rest = name.partition("(")
    token = rest[:-1] if rest.endswith(")") else rest
    head = _SUFFIX.sub("", head)
    if head == "hor":
        return "asat_hor_live" if token.endswith("L") else "asat_hor"
    if head == "dhor":
        return "asat_dhor"
    if head == "mon":
        return "asat_mon"
    if head == "censat":
        return "other_sat"
    if head in {"hsat1A", "hsat1B", "hsat2", "hsat3", "bsat", "gsat", "rDNA", "ct"}:
        return head
    return ""


def load_bed(path, usecols, names):
    # skip a UCSC "track ..." header line if present; never use pandas comment=
    # (it truncates every line at the comment char wherever it appears)
    with open(path) as fh:
        skip = 1 if fh.readline().startswith(("track", "#")) else 0
    return pd.read_csv(
        path, sep="\t", header=None, usecols=usecols, names=names, skiprows=skip
    ).dropna()


def bin_offsets(bins: pd.DataFrame) -> dict:
    """chrom -> (first bin index, chrom length) assuming bins are genome-ordered."""
    out = {}
    for chrom, g in bins.groupby("chrom", sort=False):
        out[chrom] = (int(g.index[0]), int(g["end"].max()))
    return out


def coverage(bins, offsets, bin_bp, chroms, starts, ends, groups, group_names):
    """Fraction of each bin covered per group. Intervals may span bins.

    Vectorized for intervals inside one bin (the vast majority); spanning
    intervals fall back to a small python loop.
    """
    n_groups = len(group_names)
    cov = np.zeros((n_groups, len(bins)), np.float64)
    gidx = {g: i for i, g in enumerate(group_names)}

    chroms = np.asarray(chroms, dtype=object)
    s = np.asarray(starts, dtype=np.int64)
    e = np.asarray(ends, dtype=np.int64)
    gid = np.array([gidx.get(g, -1) for g in np.asarray(groups, dtype=object)])
    base = np.array([offsets.get(c, (-1, 0))[0] for c in chroms], np.int64)
    clen = np.array([offsets.get(c, (-1, 0))[1] for c in chroms], np.int64)
    s = np.clip(s, 0, None)
    e = np.minimum(e, clen)
    ok = (gid >= 0) & (base >= 0) & (e > s)
    s, e, gid, base = s[ok], e[ok], gid[ok], base[ok]

    b0 = s // bin_bp
    b1 = (e - 1) // bin_bp
    same = b0 == b1
    flat = gid[same] * len(bins) + base[same] + b0[same]
    np.add.at(cov.reshape(-1), flat, (e - s)[same].astype(np.float64))
    for si, ei, g, ba, x0, x1 in zip(
        s[~same], e[~same], gid[~same], base[~same], b0[~same], b1[~same]
    ):
        for b in range(x0, x1 + 1):
            lo = max(si, b * bin_bp)
            hi = min(ei, (b + 1) * bin_bp)
            cov[g, ba + b] += hi - lo
    widths = (bins["end"] - bins["start"]).to_numpy(np.float64)
    return cov / widths[None, :]


def rm_group(rclass: str, family: str) -> str:
    if family.startswith("Alu"):
        return "alu"
    if family.startswith("L1"):
        return "l1"
    if rclass == "Retroposon" or family.startswith("SVA"):
        return "sva"
    if rclass == "LINE":
        return "line_other"
    if rclass == "SINE":
        return "sine_other"
    if rclass.startswith("LTR"):
        return "ltr"
    if rclass.startswith("DNA") or rclass == "RC":
        return "dna"
    if rclass == "Satellite":
        return "sat_rm"
    if rclass in {"Simple_repeat", "Low_complexity"}:
        return "simple"
    return "other_rm"


RM_GROUPS = ["alu", "l1", "sva", "line_other", "sine_other", "ltr", "dna",
             "sat_rm", "simple", "other_rm"]


def run(params: Params, bins: pd.DataFrame | None = None) -> pd.DataFrame:
    if bins is None:
        bins = pd.read_parquet(params.bins_parquet)
    bins = bins.reset_index(drop=True)
    offsets = bin_offsets(bins)
    bin_bp = params.bin_bp

    cen = load_bed(params.censat, [0, 1, 2, 3], ["chrom", "start", "end", "name"])
    cen["cls"] = cen["name"].map(censat_class)
    cen = cen[cen["cls"] != ""]
    cen_cov = coverage(
        bins, offsets, bin_bp,
        cen["chrom"], cen["start"], cen["end"], cen["cls"], CENSAT_CLASSES,
    )
    # dominant satellite class per bin (ct = transition doesn't count as satellite)
    sat_rows = [i for i, c in enumerate(CENSAT_CLASSES) if c != "ct"]
    sat_cov = cen_cov[sat_rows]
    best = sat_cov.argmax(axis=0)
    best_frac = sat_cov.max(axis=0)
    label = np.where(
        best_frac >= 0.3, np.array([CENSAT_CLASSES[sat_rows[i]] for i in best]), ""
    )
    annot = pd.DataFrame({
        "censat_class": label,
        "censat_frac": best_frac.astype(np.float32),
        "ct_frac": cen_cov[CENSAT_CLASSES.index("ct")].astype(np.float32),
    })
    for i, c in enumerate(CENSAT_CLASSES):
        annot[f"cov_{c}"] = cen_cov[i].astype(np.float32)

    sd_path = DATA / "chm13v2.0_SD.bed"
    if sd_path.exists():
        sd = load_bed(sd_path, [0, 1, 2], ["chrom", "start", "end"])
        merged = []
        for chrom, g in sd.sort_values(["chrom", "start"]).groupby("chrom"):
            s = g["start"].to_numpy()
            e = g["end"].to_numpy()
            keep_s, keep_e = [s[0]], [e[0]]
            for a, b in zip(s[1:], e[1:]):
                if a <= keep_e[-1]:
                    keep_e[-1] = max(keep_e[-1], b)
                else:
                    keep_s.append(a)
                    keep_e.append(b)
            merged.append(pd.DataFrame({"chrom": chrom, "start": keep_s, "end": keep_e}))
        sdm = pd.concat(merged)
        annot["sd_frac"] = coverage(
            bins, offsets, bin_bp, sdm["chrom"], sdm["start"], sdm["end"],
            np.repeat("sd", len(sdm)), ["sd"],
        )[0].astype(np.float32)
    else:
        annot["sd_frac"] = np.float32(0)

    telo = DATA / "chm13v2.0_telomere.bed"
    if telo.exists():
        t = load_bed(telo, [0, 1, 2], ["chrom", "start", "end"])
        annot["telo"] = (
            coverage(bins, offsets, bin_bp, t["chrom"], t["start"], t["end"],
                     np.repeat("t", len(t)), ["t"])[0] > 0
        )
    else:
        annot["telo"] = False

    if params.repeatmasker.exists():
        rm = pd.read_csv(
            params.repeatmasker, sep="\t", header=None, usecols=[0, 1, 2, 6, 7],
            names=["chrom", "start", "end", "rclass", "family"],
            dtype={"chrom": str, "rclass": str, "family": str},
        )
        rm["grp"] = [rm_group(c, f) for c, f in zip(rm["rclass"], rm["family"])]
        rm_cov = coverage(
            bins, offsets, bin_bp, rm["chrom"], rm["start"], rm["end"],
            rm["grp"], RM_GROUPS,
        )
        for i, g in enumerate(RM_GROUPS):
            annot[f"rm_{g}"] = np.clip(rm_cov[i], 0, 1).astype(np.float32)
        annot["rm_total"] = np.clip(rm_cov.sum(axis=0), 0, 1).astype(np.float32)
    else:
        print("  (RepeatMasker bed not found; skipping TE fractions)")

    # arm call from alpha-satellite extent per chromosome
    arm = np.full(len(bins), "", dtype=object)
    alpha = cen[cen["cls"].str.startswith(("asat_hor", "asat_dhor", "asat_mon"))]
    for chrom, g in bins.groupby("chrom", sort=False):
        a = alpha[alpha["chrom"] == chrom]
        idx = g.index.to_numpy()
        if len(a) == 0:
            arm[idx] = "q"
            continue
        cen_s, cen_e = a["start"].min(), a["end"].max()
        mid = (g["start"] + g["end"]) / 2
        arm[idx] = np.where(mid < cen_s, "p", np.where(mid > cen_e, "q", "cen"))
    annot["arm"] = arm
    annot["acro_p"] = bins["chrom"].isin(["chr13", "chr14", "chr15", "chr21", "chr22"]) & (
        annot["arm"] == "p"
    )

    out = OUT / f"annot_{bin_bp}.parquet"
    OUT.mkdir(exist_ok=True)
    annot.to_parquet(out, index=False)
    n_sat = int((annot["censat_class"] != "").sum())
    print(f"annotate: {n_sat} satellite-labeled bins of {len(bins)} -> {out.name}")
    return annot
