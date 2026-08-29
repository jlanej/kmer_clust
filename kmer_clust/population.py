"""Population phase (pilot): per-haplotype readouts across HPRC samples,
computed purely from cached whole-haplotype scans plus reference
annotation in its judging role.

A. Centromere divergence meter: per sample x chromosome, the median
   vocabulary coverage of query windows whose top-1 exact locus is a
   live-HOR bin. 1 - coverage = that individual's centromere novelty vs
   CHM13 — the Logsdon-style per-individual centromere divergence, with
   no HOR annotation of the query and no alignment.

B. Acro p-arm commons: per sample x p-arm, the fraction of windows whose
   top-1 lands on that acrocentric short arm while carrying a strong
   second locus (J2 >= 0.5 x J1) on a DIFFERENT acrocentric — per-window
   PHR-style promiscuity, comparable across haplotypes.

Run: python -m kmer_clust.population
Outputs: out/population.json, out/figs/pop_centromeres.png, pop_acro.png
"""

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import OUT, PARAMS

MIN_CELL = 5
ACROS = ["chr13", "chr14", "chr15", "chr21", "chr22"]
SURFACE, INK, MUTED = "#fcfcfb", "#171916", "#6f6d64"


def load_samples():
    out = {}
    for p in sorted(OUT.glob("hapscan_*.parquet")):
        tag = p.stem.replace("hapscan_", "")
        parts = tag.split(".")
        sample = parts[0] + (" mat" if "maternal" in parts else
                             " pat" if "paternal" in parts else "")
        if sample in out:
            raise RuntimeError(f"duplicate sample key {sample!r} from {p.name}")
        df = pd.read_parquet(p)
        df["top"] = df["ehits"].str[0]
        df["j1"] = df["ejacc"].str[0]
        out[sample] = df
    return out


def run(params=PARAMS):
    bins = pd.read_parquet(OUT / "bins_embedded.parquet")
    chroms = list(dict.fromkeys(bins["chrom"]))
    chrom_of = np.array([chroms.index(c) for c in bins["chrom"]])
    annot = pd.read_parquet(OUT / f"annot_{params.bin_bp}.parquet")
    hor_live = (annot["censat_class"] == "asat_hor_live").to_numpy()
    acro_p = annot["acro_p"].to_numpy().astype(bool)
    acro_idx = [chroms.index(c) for c in ACROS]

    samples = load_samples()
    print(f"{len(samples)} scanned haplotypes: {', '.join(samples)}")

    # ---- A. centromere divergence meter
    cen = {}
    for s, df in samples.items():
        hl = df[hor_live[df["top"]]]
        row = {}
        for ci, c in enumerate(chroms):
            sub = hl[chrom_of[hl["top"]] == ci]
            if len(sub) >= MIN_CELL:
                row[c] = round(float(sub["cover"].median()), 3)
        cen[s] = row
    # ---- B. acro commons promiscuity
    acro = {}
    for s, df in samples.items():
        ap = df[acro_p[df["top"]]].copy()
        row = {}
        for c in ACROS:
            ci = chroms.index(c)
            sub = ap[chrom_of[ap["top"]] == ci]
            if len(sub) < MIN_CELL:
                continue
            n_prom = n_censored = 0
            for r in sub.itertuples():
                j1 = r.j1
                hit = False
                for b, j in zip(r.ehits[1:], r.ejacc[1:]):
                    ci2 = chrom_of[b]
                    if ci2 != ci and ci2 in acro_idx and j >= 0.5 * j1:
                        hit = True
                        break
                if hit:
                    n_prom += 1
                elif all(chrom_of[b] == ci for b in r.ehits) \
                        and r.ejacc[-1] >= 0.5 * j1:
                    # every stored hit is same-chromosome and the hit list
                    # is still above the promiscuity bar at its truncation
                    # depth — a qualifying alternative below rank 8 cannot
                    # be excluded
                    n_censored += 1
            row[c] = {"n": len(sub),
                      "promiscuous": round(n_prom / len(sub), 3),
                      "censored": round(n_censored / len(sub), 3)}
        acro[s] = row

    res = {"samples": list(samples), "centromere_cover": cen, "acro_commons": acro}
    with open(OUT / "population.json", "w") as fh:
        json.dump(res, fh, indent=1)

    # ---- figures
    slist = list(samples)
    show_chroms = [c for c in chroms
                   if sum(c in cen[s] for s in slist) == len(slist)]
    M = np.array([[1 - cen[s][c] for c in show_chroms] for s in slist])
    fig, ax = plt.subplots(figsize=(12.5, 0.65 * len(slist) + 1.6), dpi=115)
    fig.patch.set_facecolor(SURFACE)
    im = ax.imshow(M, cmap="Purples", vmin=0, vmax=max(M.max(), 0.3),
                   aspect="auto")
    ax.set_xticks(range(len(show_chroms)),
                  [c.replace("chr", "") for c in show_chroms], fontsize=8)
    ax.set_yticks(range(len(slist)), slist, fontsize=9)
    for i in range(len(slist)):
        for j in range(len(show_chroms)):
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center",
                    fontsize=6.5,
                    color="white" if M[i, j] > 0.18 else INK)
    ax.set_title("personal centromeres: live-HOR novelty vs CHM13 "
                 "(1 − median vocabulary coverage of centromere-landing "
                 "windows)", fontsize=10, color=INK)
    fig.colorbar(im, ax=ax, shrink=0.8, label="novel fraction")
    fig.tight_layout()
    fig.savefig(OUT / "figs" / "pop_centromeres.png", bbox_inches="tight",
                facecolor=SURFACE)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 0.65 * len(slist) + 1.8), dpi=115)
    fig.patch.set_facecolor(SURFACE)
    A = np.full((len(slist), len(ACROS)), np.nan)
    for i, s in enumerate(slist):
        for j, c in enumerate(ACROS):
            if c in acro[s]:
                A[i, j] = acro[s][c]["promiscuous"]
    im = ax.imshow(A, cmap="Oranges", vmin=0, vmax=np.nanmax(A), aspect="auto")
    ax.set_xticks(range(len(ACROS)), [c.replace("chr", "") + "p" for c in ACROS],
                  fontsize=9)
    ax.set_yticks(range(len(slist)), slist, fontsize=9)
    for i in range(len(slist)):
        for j in range(len(ACROS)):
            if A[i, j] == A[i, j]:
                cell = acro[slist[i]][ACROS[j]]
                n, cfrac = cell["n"], cell.get("censored", 0)
                txt = f"{A[i, j]:.0%}\nn={n}" + (f"\n(+{cfrac:.0%}?)" if cfrac >= 0.05 else "")
                ax.text(j, i, txt, ha="center", va="center",
                        fontsize=7,
                        color="white" if A[i, j] > 0.55 * np.nanmax(A) else INK)
    ax.set_title("the acrocentric commons, per haplotype: windows landing on\n"
                 "each p-arm that carry a strong second home on another acrocentric\n"
                 "(+N%? = censored by the top-8 hit list: alternatives below "
                 "rank 8 cannot be excluded)", fontsize=9.5, color=INK)
    fig.colorbar(im, ax=ax, shrink=0.8, label="promiscuous fraction")
    fig.tight_layout()
    fig.savefig(OUT / "figs" / "pop_acro.png", bbox_inches="tight",
                facecolor=SURFACE)
    plt.close(fig)
    print("figures: out/figs/pop_centromeres.png, pop_acro.png")

    for s in slist:
        cc = cen[s]
        worst = max(cc, key=lambda c: 1 - cc[c]) if cc else "-"
        print(f"  {s}: centromere novelty median "
              f"{np.median([1-v for v in cc.values()]):.2f} "
              f"(most novel {worst} {1-cc.get(worst, 1):.2f})")
    return res


if __name__ == "__main__":
    run()
