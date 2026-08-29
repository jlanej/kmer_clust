"""Render shareable GIFs straight from the data (no browser capture):

- k_sweep.gif           the map morphing through vocabularies k=15..25
- tour_<set>.gif        the projection tour: assembly -> word-space -> loci
                        -> fine placement (composite excised axis, J bars)

Outputs land in docs/media/ (tracked, for the README) and out/figs/.
Run: python -m kmer_clust.gifs
"""

import colorsys
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from .config import DOCS, OUT, PARAMS
from .figures import censat_palette

ACC = "#3987e5"
SURFACE = "#fcfcfb"
INK = "#171916"
MUTED = "#8b897f"
GREY = "#c9c8c2"
MEDIA = DOCS / "media"


def ease(t):
    return 2 * t * t if t < 0.5 else 1 - (-2 * t + 2) ** 2 / 2


def frame(draw_fn, w=6.4, h=4.7):
    fig = plt.figure(figsize=(w, h), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    draw_fn(ax)
    fig.canvas.draw()
    img = Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[:, :, :3])
    plt.close(fig)
    return img


def save_gif(frames, path, duration=85):
    path.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(path, save_all=True, append_images=frames[1:],
                   duration=duration, loop=0, optimize=True)
    print(f"  {path} ({path.stat().st_size/1e6:.1f} MB, {len(frames)} frames)")


def norm_layouts(store, keys):
    pts = np.concatenate([store[k] for k in keys])
    lo, hi = pts.min(0), pts.max(0)
    return {k: (store[k] - lo) / np.maximum(hi - lo, 1e-9) for k in keys}


def gif_k_sweep(out_path):
    store = np.load(OUT / "kladder.npz")
    kl = json.loads((OUT / "kladder.json").read_text())
    ks = kl["ks"]
    keys = [f"k{k}" for k in ks]
    L = norm_layouts(store, keys)
    annot = pd.read_parquet(OUT / f"annot_{PARAMS.bin_bp}.parquet")
    pal = censat_palette(PARAMS)
    colors = np.array([pal.get(c if c else "non_sat", GREY)
                       for c in annot["censat_class"]])
    sat = (annot["censat_class"] != "").to_numpy()
    order = np.argsort(sat)  # satellites drawn on top

    def draw(xy, label):
        def fn(ax):
            p = xy[order]
            ax.scatter(p[:, 0] * 0.9 + 0.05, p[:, 1] * 0.82 + 0.10,
                       s=1.0, c=colors[order], linewidths=0, rasterized=True)
            ax.text(0.03, 0.955, f"word length {label}", fontsize=15,
                    color=INK, family="monospace", weight="bold")
            ax.text(0.03, 0.915, "the same 31,185 bins, re-sketched — "
                    "colors are censat annotation (judge, not input)",
                    fontsize=8, color=MUTED)
        return fn

    frames = []
    seq = keys + [keys[0]]  # loop back to k15
    for a, b in zip(seq[:-1], seq[1:]):
        ka = a[1:]
        for _ in range(7):
            frames.append(frame(draw(L[a], f"k = {ka}")))
        n_tr = 14 if b == keys[0] else 11
        for f in range(1, n_tr):
            t = ease(f / n_tr)
            frames.append(frame(draw(L[a] * (1 - t) + L[b] * t, f"k = {ka}")))
    save_gif(frames, out_path)


def _fine_axis(windows, chroms, chrom_nbins, bin_bp, x0=0.05, x1=0.95):
    """Composite excised axis in figure coords (mirrors the atlas panel)."""
    per = {}
    for w in windows:
        f = w["_fine"]
        lo, hi = per.get(f["ci"], (np.inf, -np.inf))
        per[f["ci"]] = (min(lo, f["mb"]), max(hi, f["mb"]))
    segs = []
    for ci in sorted(per):
        lo, hi = per[ci]
        clen = chrom_nbins[ci] * bin_bp / 1e6
        segs.append({"ci": ci, "mb0": max(0, lo - 2), "mb1": min(clen, hi + 2)})
    total = sum(g["mb1"] - g["mb0"] for g in segs) or 1
    gap = 0.02
    usable = (x1 - x0) - gap * (len(segs) - 1)
    per_mb = usable / total
    x = x0
    for g in segs:
        g["x0"] = x
        g["x1"] = x + (g["mb1"] - g["mb0"]) * per_mb
        x = g["x1"] + gap
    return segs, per_mb, total


def gif_tour(set_entry, chroms, chrom_nbins, chrom_off, n_bins, ghost_xy, out_path,
             sat=None):
    bin_bp = PARAMS.bin_bp
    windows = [w for w in set_entry["windows"] if w.get("hits") and w.get("loci")]
    for w in windows:
        l = w["loci"][0]
        ci = chroms.index(l["chrom"])
        bj = l.get("bins_j") or []
        sw = sum(j for _, j in bj)
        mb = (sum(((b - chrom_off[ci]) * bin_bp / 1e6 + 0.05) * j for b, j in bj) / sw
              if sw else (l["start_mb"] + l["end_mb"]) / 2)
        w["_fine"] = {"ci": ci, "mb": mb, "j": l["sim"]}
    lo_mb = min(w["pos_mb"] for w in windows)
    max_mb = max(w["pos_mb"] for w in windows) - lo_mb + 0.1
    segs, per_mb, total = _fine_axis(windows, chroms, chrom_nbins, bin_bp)
    zoom_x = (n_bins * bin_bp / 1e6) / total

    def fine_x(ci, mb):
        for g in segs:
            if g["ci"] == ci:
                return g["x0"] + (min(max(mb, g["mb0"]), g["mb1"]) - g["mb0"]) * per_mb
        return 0.05

    A = np.array([[0.05 + (w["pos_mb"] - lo_mb) / max_mb * 0.9, 0.90]
                  for w in windows])
    B = []
    for w in windows:
        wt = np.array([1 / (1.0001 - s) for _, s in w["hits"]])
        pts = ghost_xy[[h for h, _ in w["hits"]]]
        B.append((pts * wt[:, None]).sum(0) / wt.sum())
    B = np.array(B) * [0.9, 0.42] + [0.05, 0.33]
    C = np.array([[0.05 + (chrom_off[chroms.index(w["loci"][0]["chrom"])]
                           + (w["loci"][0]["start_mb"] + w["loci"][0]["end_mb"]) / 2
                           * 1e6 / bin_bp) / n_bins * 0.9,
                   0.245 + (i % 3) * 0.022] for i, w in enumerate(windows)])
    F = np.array([[fine_x(w["_fine"]["ci"], w["_fine"]["mb"]), 0.10] for w in windows])
    STAGES = [A, B, C, F]
    nq = len(windows)
    big = nq > 60          # whole-chromosome sets: smaller marks, fainter lines
    DOT_S, DOT_EW = (13, 0.4) if big else (26, 0.7)
    # index paint: hue sweep first->last (blue -> purple -> orange, no green)
    DCOL = [colorsys.hls_to_rgb(((215 + 165 * (i / (nq - 1) if nq > 1 else 0))
                                 % 360) / 360, 0.50, 0.68) for i in range(nq)]
    # Kendall tau: assembly order vs fine-axis position (ties count 0)
    tie_fig = 0.01 * per_mb  # placements within 10 kb count as tied
    tau_s = sum(np.sign(F[j][0] - F[i][0]) if abs(F[j][0] - F[i][0]) > tie_fig
                else 0 for i in range(nq) for j in range(i + 1, nq))
    tau = tau_s / (nq * (nq - 1) / 2) if nq > 1 else 0.0
    tauv = ("collinear" if tau >= 0.85 else "inverted" if tau <= -0.85 else
            "mostly ordered" if abs(tau) >= 0.5 else "scrambled")
    LABELS = ["1 · assembly coordinates", "2 · word-space (T2T ghosted)",
              "3 · loci on T2T", "4 · fine placement", "5 · world-lines",
              "6 · assembly ↔ placement"]

    gxy = ghost_xy * [0.9, 0.42] + [0.05, 0.33]

    blurb = set_entry.get("blurb", "")
    if len(blurb) > 118:
        blurb = blurb[:115] + "…"

    def chrome(ax, stage, salpha, direct=0.0):
        ax.text(0.03, 0.955, set_entry["label"], fontsize=11, color=INK, weight="bold")
        if blurb:
            ax.text(0.03, 0.925, blurb, fontsize=6.8, color=MUTED)
        ax.text(0.62, 0.955, LABELS[stage], fontsize=9.5, color=MUTED, family="monospace")
        ax.plot([0.05, 0.95], [0.90, 0.90], color=MUTED, lw=0.8, alpha=0.5)
        if direct < 0.995:
            ax.scatter(gxy[:, 0], gxy[:, 1], s=0.5, c=GREY,
                       alpha=0.5 * (1 - direct), linewidths=0, rasterized=True)
            ax.plot([0.05, 0.95], [0.22, 0.22], color=MUTED, lw=0.8,
                    alpha=0.4 * (1 - direct))
            ax.text(0.05, 0.196, "T2T loci (full genome)", fontsize=6.5,
                    color=MUTED, alpha=1 - direct)
        for g in segs:
            if sat is not None:
                b0 = chrom_off[g["ci"]] + int(g["mb0"] * 1e6 / bin_bp)
                b1 = chrom_off[g["ci"]] + int(g["mb1"] * 1e6 / bin_bp)
                run0 = -1
                for b in range(b0, b1 + 1):
                    on = b < b1 and bool(sat[b])
                    if on and run0 < 0:
                        run0 = b
                    elif not on and run0 >= 0:
                        xa = g["x0"] + (run0 - b0) * bin_bp / 1e6 * per_mb
                        xb = g["x0"] + (b - b0) * bin_bp / 1e6 * per_mb
                        ax.fill_between([xa, xb], 0.094, 0.106,
                                        color="#c9b8d8", alpha=0.5,
                                        linewidth=0, zorder=1)
                        run0 = -1
            ax.plot([g["x0"], g["x1"]], [0.10, 0.10], color=MUTED, lw=1.0, alpha=0.65)
            ax.text((g["x0"] + g["x1"]) / 2, 0.062,
                    chroms[g["ci"]].replace("chr", ""), fontsize=7.5,
                    color=MUTED, ha="center")
        for i in range(1, len(segs)):
            xm = (segs[i - 1]["x1"] + segs[i]["x0"]) / 2
            ax.plot([xm - 0.005, xm + 0.001], [0.108, 0.092], color=MUTED, lw=0.8)
            ax.plot([xm + 0.001, xm + 0.007], [0.108, 0.092], color=MUTED, lw=0.8)
        if sat is not None:
            ax.text(0.95, 0.062, "shaded = satellite reference", fontsize=6,
                    color=MUTED, ha="right")
        ax.text(0.05, 0.028, f"fine placement — {total:.0f} Mb of "
                f"{n_bins*bin_bp/1e6:.0f} Mb shown ({zoom_x:.0f}× zoom), "
                "bar height = exact Jaccard · assembly order "
                f"τ {tau:+.2f} ({tauv})", fontsize=6.5, color=MUTED)
        ax.text(A[0][0] - 0.008, 0.90, "1", fontsize=6.5, color=MUTED,
                ha="right", va="center")
        ax.text(A[-1][0] + 0.008, 0.90, str(nq), fontsize=6.5, color=MUTED,
                ha="left", va="center")
        if set_entry.get("n_win_all", 0) > nq:
            kind = ("best contiguous run of" if set_entry.get("trim_kind")
                    == "contiguous-run" else "uniformly thinned from")
            ax.text(0.05, 0.877, f"showing {nq} windows — {kind} "
                    f"{set_entry['n_win_all']} spanning the region",
                    fontsize=6, color=MUTED)
        if stage >= 2 and salpha > 0.3 and direct < 0.98:
            for i, w in enumerate(windows):
                for li, l in enumerate(w["loci"]):
                    lx = 0.05 + (chrom_off[chroms.index(l["chrom"])]
                                 + (l["start_mb"] + l["end_mb"]) / 2 * 1e6 / bin_bp
                                 ) / n_bins * 0.9
                    src = STAGES[min(stage, 2)][i] if stage == 2 else C[i]
                    ax.plot([src[0], lx], [src[1], 0.222],
                            color=DCOL[i], lw=(0.4 if li else 0.7) if big else
                            (0.5 if li else 0.9),
                            alpha=(0.14 if li else 0.30) * salpha * (1 - direct))
        if stage >= 3 and salpha > 0.3:
            for i, w in enumerate(windows):
                ax.plot([F[i][0], F[i][0]], [0.10, 0.10 + w["_fine"]["j"] * 0.09],
                        color=DCOL[i], lw=1.0 if big else 1.6,
                        alpha=0.45 * salpha)

    def draw(pts, stage, salpha, paths=0.0, direct=0.0):
        def fn(ax):
            chrome(ax, stage, salpha, direct)
            if paths > 0.01:
                for i in range(len(windows)):
                    p1 = B[i] + (A[i] + (F[i] - A[i]) / 3 - B[i]) * direct
                    p2 = C[i] + (A[i] + (F[i] - A[i]) * 2 / 3 - C[i]) * direct
                    ax.plot([A[i][0], p1[0], p2[0], F[i][0]],
                            [A[i][1], p1[1], p2[1], F[i][1]],
                            color=DCOL[i], lw=0.7 if big else 1.0,
                            alpha=(0.22 if big else 0.30) * paths
                            + 0.12 * direct, zorder=4)
                ax.scatter(A[:, 0], A[:, 1], s=4 if big else 7, c=DCOL,
                           alpha=0.32 * paths, linewidths=0, zorder=4)
                if direct < 0.98:
                    for S in (B, C):
                        ax.scatter(S[:, 0], S[:, 1], s=4 if big else 7, c=DCOL,
                                   alpha=0.32 * paths * (1 - direct),
                                   linewidths=0, zorder=4)
            ax.scatter(pts[:, 0], pts[:, 1], s=DOT_S, c=DCOL, zorder=5,
                       edgecolors=SURFACE, linewidths=DOT_EW)
        return fn

    frames = []
    order = STAGES + [STAGES[0]]
    for si in range(len(order) - 1):
        stage = si % 4
        for _ in range(9):
            frames.append(frame(draw(order[si], stage, 1.0)))
        finale = stage == 3
        if finale:  # world-lines: every path through all four stations at once
            for f in range(1, 7):
                frames.append(frame(draw(order[si], 4, 1.0, paths=ease(f / 6))))
            for _ in range(12):
                frames.append(frame(draw(order[si], 4, 1.0, paths=1.0)))
            # then the inner tracks dissolve and every path straightens into
            # a direct assembly <-> placement ribbon
            for f in range(1, 9):
                frames.append(frame(draw(order[si], 5, 1.0, paths=1.0,
                                         direct=ease(f / 8))))
            for _ in range(12):
                frames.append(frame(draw(order[si], 5, 1.0, paths=1.0,
                                         direct=1.0)))
        for f in range(1, 12):
            t = ease(f / 12)
            frames.append(frame(draw(order[si] * (1 - t) + order[si + 1] * t,
                                     5 if finale else stage, 1 - t,
                                     paths=(1 - t) if finale else 0.0,
                                     direct=(1 - t) if finale else 0.0)))
    save_gif(frames, out_path)


def run(params=PARAMS) -> None:
    gif_k_sweep(MEDIA / "k_sweep.gif")
    proj = json.loads((OUT / "projection.json").read_text())
    bins = pd.read_parquet(OUT / "bins_embedded.parquet")
    chroms = list(dict.fromkeys(bins["chrom"]))
    chrom_nbins = [int((bins["chrom"] == c).sum()) for c in chroms]
    chrom_off = np.concatenate(([0], np.cumsum(chrom_nbins)))[:-1].tolist()
    xy = bins[["x", "y"]].to_numpy(np.float64)
    ghost = (xy - xy.min(0)) / np.maximum(xy.max(0) - xy.min(0), 1e-9)
    annot = pd.read_parquet(OUT / f"annot_{PARAMS.bin_bp}.parquet")
    sat = (annot["censat_class"] != "").to_numpy()
    for s in proj["sets"]:
        gif_tour(s, chroms, chrom_nbins, chrom_off, len(bins), ghost,
                 MEDIA / f"tour_{s['id']}.gif", sat=sat)


if __name__ == "__main__":
    run()
