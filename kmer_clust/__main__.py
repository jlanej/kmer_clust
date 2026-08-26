"""CLI: python -m kmer_clust <stage> [overrides]."""

import argparse
import time

from .config import PARAMS

STAGES = ["sketch", "matrix", "embed", "pairwise", "annotate", "analyze",
          "periods", "figures", "site"]


def main() -> None:
    ap = argparse.ArgumentParser(prog="kmer-clust")
    ap.add_argument("stage", choices=STAGES + ["all"])
    ap.add_argument("--k", type=int)
    ap.add_argument("--bin-bp", type=int)
    ap.add_argument("--base-scaled", type=int)
    ap.add_argument("--embed-scaled", type=int)
    args = ap.parse_args()

    p = PARAMS
    for name in ("k", "bin_bp", "base_scaled", "embed_scaled"):
        v = getattr(args, name)
        if v is not None:
            setattr(p, name, v)

    stages = STAGES if args.stage == "all" else [args.stage]
    t0 = time.time()
    for stage in stages:
        print(f"== {stage} ==")
        if stage == "sketch":
            from . import sketch_run

            sketch_run.run(p)
        elif stage == "matrix":
            from . import matrix

            matrix.run(p)
        elif stage == "embed":
            from . import embed_run

            embed_run.run(p)
        elif stage == "pairwise":
            from . import pairwise

            pairwise.run(p)
        elif stage == "annotate":
            from . import annotate

            annotate.run(p)
        elif stage == "analyze":
            from . import analyze

            analyze.run(p)
        elif stage == "periods":
            from . import periodicity

            periodicity.run(p)
        elif stage == "figures":
            from . import figures

            figures.run(p)
        elif stage == "site":
            from . import site

            site.run(p)
    print(f"== done ({time.time()-t0:.0f}s) ==")


if __name__ == "__main__":
    main()
