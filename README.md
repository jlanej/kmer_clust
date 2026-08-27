# kmer_clust

**A map of the complete human genome in which position is decided purely by k-mer
vocabulary.** No aligner, no reference coordinate, no orthology. Every dot is a
100 kb bin of T2T-CHM13v2.0; bins land next to each other because they use the
same words.

**→ The interactive atlas: [`docs/index.html`](docs/index.html)** (deployed to
GitHub Pages by [`pages.yml`](.github/workflows/pages.yml)).

This repo is the deliberately-narrow successor of
[kmer_dust](https://github.com/jlanej/kmer_dust): one genome, one sketch engine,
two analysis tracks, one self-contained report. Where kmer_dust spanned 463 HPRC
assemblies with 10 kb bins at scaled=200 (~50 hashes per bin — its own
representation study found satellite bins vocabulary-starved at ~27), kmer_clust
inverts the trade: **100 kb bins at scaled=20**, roughly 250× more hashes per
bin-pair comparison, deep enough that pairwise distances stop being estimates.

## The pipeline

```
chm13v2.0.fa.gz
   │  fasta.py       stream → 2-bit codes, cached per chromosome
   ▼
sketch (fracminhash.py, numba)          ~160 s for 3.1 Gbp on an M1 Max
   │  canonical 21-mers → MurmurHash3-x64-128(seed 42) → keep h ≤ 2⁶⁴/scaled
   │  bit-identical to sourmash (tested against it), with per-bin multiplicities
   ▼
sketch store  data/sketch_k21_s20_bin100000.npz
   │  (bin → sorted unique hashes + counts; every coarser scaled and larger
   │   bin size is derived from this store without touching sequence again)
   │
   ├── Track A: the model ──────────────────────────────────────────────
   │     matrix.py   bins × shared-hashes (scaled=50, df≥2), log1p(count)·IDF, L2
   │     gram_rsvd   randomized subspace SVD on the implicit Gram operator
   │     embed_run   UMAP (cosine) 2-d display + 12-d clustering space
   │                 HDBSCAN → clusters, named after the fact by annotation
   │
   └── Track B: the ground truth ──────────────────────────────────────
         pairwise.py  bins boosted to 1 Mb (union of children); one sparse
                      matmul → exact Jaccard & max-containment for all pairs;
                      cANI = C^(1/k); average-linkage + optimal leaf ordering;
                      HDBSCAN(metric=precomputed); UMAP(precomputed)

annotate.py   censat v2.1 (+ segdup, RepeatMasker, telomere) → per-bin fractions
analyze.py    metrics, the two-way acrocentric test, robustness sweeps, A/B accord
figures.py    static analytical panels
site.py       everything → one self-contained docs/index.html (no CDN)
```

Run it:

```bash
python3.11 -m venv .venv && .venv/bin/pip install -e .[dev]
make test     # sketch engine is verified bit-identical to sourmash
make all      # sketch → matrix → embed → pairwise → annotate → analyze → figures → site
```

Everything runs on one laptop. Data lands in `data/` (~2 GB incl. the genome),
results in `out/`, the site in `docs/`.

## Design decisions (and why)

- **FracMinHash, sourmash-compatible.** The engine is ~200 lines of numba, but
  the hash function, canonicalization, and keep-rule are bit-identical to
  sourmash (enforced by `tests/test_fracminhash.py`), so sourmash's
  Jaccard/containment/ANI theory applies unchanged and sketches are
  interchangeable.
- **One dense store, many views.** The base sketch (scaled=20, 100 kb bins,
  with multiplicities) is written once; scaled=50/100/200 and 1 Mb bins are
  subsets/unions of it. FracMinHash's containment property makes downsampling a
  threshold cut.
- **Private vocabulary is a feature, not a column.** Hashes seen in only one
  bin (most of euchromatin) carry no between-bin signal; they are dropped from
  the model, and each bin's private fraction is kept as a diagnostic — the map
  can be colored by it.
- **Abundance kept, IDF kept mild.** Satellite arrays express identity through
  multiplicity of few words (kmer_dust's study: within-bin dedup discards ~74%
  of live-HOR signal, and reproducibility *rises* with copy number), so weights
  are log1p(multiplicity) × log1p(N/df).
- **Clusters come from a 12-d UMAP space, display from 2-d.** And a 4×3
  HDBSCAN parameter sweep is part of the report, so "how many clusters" is
  presented as the parameter-dependent quantity it is.
- **Annotation judges, never builds.** censat/RepeatMasker/segdup tracks color
  and score the finished map; they contribute nothing upstream.

## What the map should show — and does

- **The two-way acrocentric test** (design borrowed from kmer_dust): alpha-HOR
  bins should be *identifiable by chromosome* from vocabulary alone
  (HOR arrays are chromosome-specific), while rDNA bins should be
  *chromosome-confused* (the arrays recombine across the five acrocentrics).
  A method that secretly encodes position passes the first and fails the
  second; a method blind to fine vocabulary fails the first. kmer_clust passes
  both — and the alpha-satellite confusions that do occur are chr1↔5↔19,
  the S1C1/5/19 HOR family those chromosomes genuinely share.
- Satellite families (HSat1A/1B/2/3, alpha live/divergent/monomeric, beta,
  gamma, rDNA) reassemble from across the genome into coherent islands.
- The euchromatic mainland organizes by repeat dialect and GC, with segdup-rich
  bins forming their own reefs; Yq12's HSat1B/HSat3 continent dominates the
  satellite hemisphere.
- Track A (cosine in the SVD model) and Track B (exact Jaccard at 1 Mb) rank
  pairs the same way — measured, not assumed.

Headline numbers from the shipped run (full details in `out/metrics.json` and
on the atlas page):

| what | result |
|---|---|
| bins / clusters / noise | 31,185 · 133 · 33.3% |
| satellite-cluster purity (size-weighted) / recovery | 90.7% / 90.4% |
| αSat HOR bin → chromosome, kNN CV (chance) | **95.7%** (7.9%) |
| rDNA bin → chromosome — the predicted weak side (chance) | 74.5% (35.7%) |
| satellite family recovery, kNN CV (chance) | **97.0%** (29.0%) |
| satellite semantic accuracy across 4 UMAP seeds | 94.3–94.4% every seed |
| satellite semantic accuracy at half / quarter sketch depth | 96.3% / 96.4% |
| HDBSCAN agreement at comparable granularity (full 10× grid) | ARI 0.61 (0.34) |
| model vs exact distance, 1 Mb pairs (Spearman) | ρ 0.54 |
| live-HOR bins: tandem period from hash spacing alone | **1,364 bp = 7.99 × the 171 bp monomer** (83% periodic) |
| rDNA bins: tandem period from hash spacing alone | **44.8 kb** (the ~45 kb unit) |
| mainland dialect R² (Alu/L1/SD/GC): k=21 → k=17 vocabulary | 0.465 → **0.530** (satellites intact) |
| dual vocabulary k21⊕k17: dialect R² / HDBSCAN noise | 0.501 / **2.2%** (vs 33.3% baseline) |

The raw k=15 neighbor-overlap numbers under reseeding/subsampling are low
(0.11–0.24) — and that is the honest reading of a dense euchromatic mass whose
neighbors are legitimately interchangeable; the semantic quantities above are
the stable object. Runtime on the M1 Max: sketch 157 s, matrix+SVD 33 s,
UMAP+HDBSCAN 76 s, exact pairwise 117 s, full analysis 282 s.

## Finding more structure (annotation-free levers)

Two additions keep the build annotation-agnostic while articulating the map:

- **Tandem periodicity** (`periodicity.py`, `make periods`): within one bin, the
  same kept hash recurring every P bases is the signature of a tandem array
  with unit ~P. The dominant gap per bin becomes a color mode and a tooltip
  stat. Judged after the fact: live alpha-HOR bins land at 7.99× the 171 bp
  monomer and rDNA at its 45 kb unit — from spacing statistics alone.
- **The structure lab** (`structure_lab.py`): a whitening sweep over SVD
  component scaling plus a second k=17 vocabulary (shorter words let diverged
  repeat copies share again). Verdict: whitening at α≈0.35 collapses HDBSCAN
  noise (33%→2%) without adding information; **k=17 adds real euchromatic
  signal** (+14% dialect R², satellites improve); the k21⊕k17 concatenation
  gets both. The atlas ships the winners as morphable embedding views —
  satellite-health-gated, baseline always default.
- **The acrocentric commons** (`analyze.acro_analysis`, atlas mode
  "acro p-arms"): for every p-arm bin of chr13/14/15/21/22, the fraction of
  its nearest vocabulary neighbors (own-array adjacency excluded) that come
  from a *different* acrocentric. Mean cross-acro mixing is 57% and
  neighbor-vote chromosome assignment only 48% — the assembly difficulty of
  these arms, quantified bin by bin. The structure is uneven and specific:
  non-satellite p-arm sequence is 88% interchangeable (the pseudo-homologous
  region), hsat1B 92%, while hsat3 (35%) and rDNA (43%) retain identity;
  13p and 15p carry large chromosome-knowing blocks while 14p/21p/22p are
  mostly commons. The atlas colors arms by chromosome, desaturated by mixing.
- **The k-ladder** (`kladder.py`): one full re-sketch and embedding per word
  length k ∈ {15,17,19,21,23,25}, each layout Procrustes-aligned to k=21 and
  quantized in a shared frame, so the atlas can expose **k as a slider** that
  morphs the map through vocabularies (with per-k judge metrics displayed
  live). k=15 is included deliberately — 4¹⁵ is about the genome's own size,
  so the slider lets you watch the random-collision floor arrive.

## Repo layout

| path | what |
|---|---|
| `kmer_clust/fracminhash.py` | numba FracMinHash engine (murmur3, canonical k-mers, per-bin stats) |
| `kmer_clust/sketch_run.py` | genome → sketch store |
| `kmer_clust/matrix.py` | store → weighted sparse matrix → low-memory randomized SVD |
| `kmer_clust/embed_run.py` | UMAP + HDBSCAN |
| `kmer_clust/pairwise.py` | 1 Mb exact pairwise distances, linkage, precomputed clustering |
| `kmer_clust/annotate.py` | censat/segdup/RepeatMasker/telomere → per-bin fractions |
| `kmer_clust/analyze.py` | metrics, two-way test, robustness sweeps, A/B accord |
| `kmer_clust/periodicity.py` | per-bin tandem period from kept-hash spacing |
| `kmer_clust/structure_lab.py` · `kladder.py` | annotation-free structure experiments; k-slider views |
| `kmer_clust/figures.py` | static panels |
| `kmer_clust/site.py` + `template.html` | the self-contained interactive atlas |
| `tests/` | sourmash bit-parity + toy-scale math checks |

## Data

- Genome: [T2T-CHM13v2.0 analysis set](https://s3-us-west-2.amazonaws.com/human-pangenomics/T2T/CHM13/assemblies/analysis_set/chm13v2.0.fa.gz) (downloaded by `make all` if absent)
- censat v2.1, segdup, telomere BEDs: cached in `data/` (originals from the
  [T2T CHM13 annotation bucket](https://github.com/marbl/CHM13))
- RepeatMasker 4.1.2p1 BED: read from a sibling `kmer_dust/data/cache/` checkout
  if present; optional (TE fractions are skipped without it)

## Deliberately out of scope

HPRC assemblies (that's kmer_dust's job), gene annotation, alignment baselines,
multi-k sweeps, Snakemake/HPC. Future fun that the store already supports:
order-aware readouts (hash positions recover HOR periodicity), multiple k from
re-sketch, cross-assembly projection.
