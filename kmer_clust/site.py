"""Stage: pack results into a single self-contained docs/index.html.

Everything is embedded: typed arrays as base64 (little-endian), the 1 Mb
similarity matrix as a grayscale PNG the page decodes back into numbers,
static robustness figures as data URIs. No CDN, no fetches (fonts aside).
"""

import base64
import io
import json

import numpy as np
import pandas as pd

from .config import DOCS, OUT, REPO, Params
from .annotate import CENSAT_CLASSES

TEMPLATE = REPO / "kmer_clust" / "template.html"


def b64(arr: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(arr).tobytes()).decode()


def png_b64(path) -> str:
    return base64.b64encode(path.read_bytes()).decode()


def quant16(v: np.ndarray):
    lo, hi = float(v.min()), float(v.max())
    q = np.round((v - lo) / max(hi - lo, 1e-9) * 65535).astype(np.uint16)
    return q, lo, hi


def u8(v, lo=0.0, hi=1.0):
    return np.clip((np.asarray(v, np.float64) - lo) / (hi - lo) * 255, 0, 255).astype(np.uint8)


def build_payload(params: Params) -> dict:
    bins = pd.read_parquet(OUT / "bins_embedded.parquet")
    annot = pd.read_parquet(OUT / f"annot_{params.bin_bp}.parquet")
    table = pd.read_parquet(OUT / "cluster_table.parquet")
    metrics = json.loads((OUT / "metrics.json").read_text())
    palette = json.loads((OUT / "palette.json").read_text())

    chroms = list(dict.fromkeys(bins["chrom"]))
    chrom_idx = np.array([chroms.index(c) for c in bins["chrom"]], np.uint8)
    chrom_nbins = [int((bins["chrom"] == c).sum()) for c in chroms]

    qx, x0, x1 = quant16(bins["x"].to_numpy())
    qy, y0, y1 = quant16(bins["y"].to_numpy())
    censat_all = CENSAT_CLASSES + ["non_sat"]
    cen_map = {c: i for i, c in enumerate(censat_all)}
    censat_i = np.array(
        [cen_map[c] if c else cen_map["non_sat"] for c in annot["censat_class"]], np.uint8
    )

    rm_cols = [c for c in ("rm_alu", "rm_l1", "rm_ltr", "rm_sva") if c in annot]
    payload = {
        "meta": {
            "n": len(bins),
            "bin_bp": params.bin_bp,
            "k": params.k,
            "scaled": params.embed_scaled,
            "base_scaled": params.base_scaled,
            "chroms": chroms,
            "chrom_nbins": chrom_nbins,
            "x_range": [x0, x1],
            "y_range": [y0, y1],
            "censat_classes": censat_all,
        },
        "arrays": {
            "qx": b64(qx),
            "qy": b64(qy),
            "chrom": b64(chrom_idx),
            "cluster": b64(bins["cluster"].to_numpy(np.int16)),
            "censat": b64(censat_i),
            "censat_frac": b64(u8(annot["censat_frac"])),
            "gc": b64(u8(bins["gc"], 0.25, 0.65)),
            "private": b64(u8(bins["private_frac"])),
            "sd": b64(u8(annot["sd_frac"])),
            "prob": b64(u8(bins["cluster_prob"])),
            "sketch": b64(
                np.clip(bins["sketch_size"].to_numpy(np.float64) / 40, 0, 255).astype(np.uint8)
            ),
        },
        "clusters": {
            # via to_json so numpy scalar types become plain JSON numbers
            "table": json.loads(table.to_json(orient="records")),
            "colors": palette["cluster"],
        },
        "censat_colors": palette["censat"],
        "metrics": metrics,
    }
    for c in rm_cols:
        payload["arrays"][c] = b64(u8(annot[c]))
        payload["meta"].setdefault("rm_cols", []).append(c)

    periods_path = OUT / f"periods_{params.bin_bp}.parquet"
    if periods_path.exists():
        per = pd.read_parquet(periods_path)
        payload["arrays"]["period"] = b64(
            np.clip(per["period_bp"].to_numpy(), 0, 65500).astype(np.uint16)
        )
        payload["arrays"]["pstrength"] = b64(u8(per["strength"]))

    acro_path = OUT / "acro.parquet"
    if acro_path.exists():
        from .analyze import ACROS

        ac = pd.read_parquet(acro_path)
        acro_idx = np.zeros(len(bins), np.uint8)
        apromisc = np.zeros(len(bins), np.uint8)
        a_ix = {c: i + 1 for i, c in enumerate(ACROS)}
        acro_idx[ac["bin"]] = [a_ix[c] for c in ac["chrom"]]
        apromisc[ac["bin"]] = np.clip(ac["promiscuity"] * 255, 0, 255).astype(np.uint8)
        payload["arrays"]["acro"] = b64(acro_idx)
        payload["arrays"]["apromisc"] = b64(apromisc)
        payload["meta"]["acros"] = ACROS

    # alternative embedding views: the k-ladder (Procrustes-aligned to k=21,
    # quantized in ONE shared frame so morphing shows structure, not scaling)
    # plus the dual-vocabulary view from the structure lab
    kl_json, kl_npz = OUT / "kladder.json", OUT / "kladder.npz"
    lab_json = OUT / "structure_lab.json"
    if kl_json.exists() and kl_npz.exists():
        kl = json.loads(kl_json.read_text())
        store = np.load(kl_npz)
        names = [f"k{k}" for k in kl["ks"]]
        names += [n for n in ("concat", "duo1521") if n in store]
        allxy = np.stack([store[n] for n in names])
        lo = allxy.reshape(-1, 2).min(axis=0)
        hi = allxy.reshape(-1, 2).max(axis=0)

        def qshared(v, ax):
            return np.round(
                (v - lo[ax]) / max(hi[ax] - lo[ax], 1e-9) * 65535
            ).astype(np.uint16)

        concat_metrics = duo_metrics = None
        if lab_json.exists():
            lab = json.loads(lab_json.read_text())
            concat_metrics = next(
                (r for r in lab["results"] if r["tag"] == "concat"), None
            )
        mk_json = OUT / "multik_lab.json"
        if mk_json.exists():
            mk = json.loads(mk_json.read_text())
            duo_metrics = next((r for r in mk if r["tag"] == "k15+k21"), None)

        TIPS = {
            "concat": "consensus vocabulary k17⊕k21 — cosine is the mean of two "
                      "adjacent horizons; their agreement sharpens clusters (~2% noise)",
            "duo1521": "information vocabulary k15⊕k21 — composition + identity, the "
                       "best-organized mainland; the horizons' disagreement lowers "
                       "flat-cluster confidence",
            "k15": "single vocabulary, k=15 — at the composition end: 4^15 is about "
                   "the genome's own size, so sharing drifts from identity to style",
        }
        views = []
        for n in names:
            xy = store[n]
            if n == "concat":
                label, met = "consensus k17⊕k21", concat_metrics
            elif n == "duo1521":
                label, met = "info k15⊕k21", duo_metrics
            else:
                k = int(n[1:])
                label, met = f"k={k}", kl["metrics"].get(str(k))
            views.append({
                "id": n, "label": label,
                "tip": TIPS.get(n, f"single vocabulary, k={n[1:]} — a full re-sketch "
                                   "of the genome at this word length"),
                "qx": b64(qshared(xy[:, 0], 0)), "qy": b64(qshared(xy[:, 1], 1)),
                "metrics": {
                    key: met[key]
                    for key in ("mainland_score", "sat_sem_2d", "alpha_chrom")
                    if met and key in met
                } if met else None,
            })
        payload["views"] = views
        payload["meta"]["slider_ids"] = [f"k{k}" for k in kl["ks"]]
        # the k=21 ladder layout IS the baseline (same seed, same path); use it
        # for the primary arrays so slider morphs start from a shared frame
        payload["arrays"]["qx"] = b64(qshared(store["k21"][:, 0], 0))
        payload["arrays"]["qy"] = b64(qshared(store["k21"][:, 1], 1))

    # ---- 1 Mb pairwise heatmap as grayscale PNG ----------------------------
    from PIL import Image

    pw = np.load(OUT / "pairwise.npz")
    parents = pd.read_parquet(OUT / "pairwise_bins.parquet")
    # display exact Jaccard, cube-root-stretched so the low-J structure is
    # visible (the cANI transform compresses everything into ~[0.75, 1]); the
    # page cubes the decoded value back to true Jaccard for the hover readout
    jacc = 1.0 - pw["d_jacc"].astype(np.float64)
    q = np.clip(np.clip(jacc, 0, 1) ** (1 / 3) * 255, 0, 255).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(q, mode="L").save(buf, format="PNG", optimize=True)
    p_chrom_idx = np.array([chroms.index(c) for c in parents["chrom"]], np.uint8)
    payload["heatmap"] = {
        "png": base64.b64encode(buf.getvalue()).decode(),
        "n": len(parents),
        "encoding": "cbrt-jaccard",
        "leaf_order": b64(pw["leaf_order"].astype(np.int16)),
        "chrom": b64(p_chrom_idx),
        "cluster": b64(parents["cluster"].to_numpy(np.int16)),
        "start_mb": b64((parents["start"].to_numpy(np.int64) // 1_000_000).astype(np.uint16)),
        "bin_mb": params.pairwise_bin_bp // 1_000_000,
    }

    payload["figs"] = {
        name: png_b64(OUT / "figs" / f"{name}.png")
        for name in ("confusions", "sweep", "ab_agreement", "seeds")
        if (OUT / "figs" / f"{name}.png").exists()
    }
    return payload


PAGES_PREFIX = (
    '<!doctype html>\n<meta charset="utf-8">\n'
    '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
)


def run(params: Params) -> None:
    payload = build_payload(params)
    html = TEMPLATE.read_text()
    blob = json.dumps(payload, separators=(",", ":"))
    html = html.replace("/*__PAYLOAD__*/null", blob)
    # optional modular 3-D panel: injected only if the fragment exists;
    # deleting kmer_clust/atlas3d.html removes the feature entirely
    frag = TEMPLATE.parent / "atlas3d.html"
    html = html.replace(
        "<!--__ATLAS3D__-->", frag.read_text() if frag.exists() else ""
    )
    DOCS.mkdir(exist_ok=True)
    out = DOCS / "index.html"
    tmp = out.with_suffix(".tmp")
    tmp.write_text(PAGES_PREFIX + html)  # standards mode for GitHub Pages
    tmp.replace(out)
    (DOCS / ".nojekyll").touch()
    # bare variant (no doctype) for publishing surfaces that wrap the content
    (OUT / "atlas_bare.html").write_text(html)
    print(f"site: {out} ({out.stat().st_size/1e6:.1f} MB)")
