# kmer_clust

**A map of the complete human genome in which position is decided purely by k-mer
vocabulary.** No aligner, no reference coordinate, no orthology. Every dot is a
100 kb bin of T2T-CHM13v2.0; bins land next to each other because they use the
same words.

**→ The interactive atlas: [`docs/index.html`](docs/index.html)** (deployed to
GitHub Pages by [`pages.yml`](.github/workflows/pages.yml)).

![The map morphing through word lengths k=15..25](docs/media/k_sweep.gif)

*The same 31,185 bins re-sketched at every word length from k=15 to k=25 —
word length is a composition↔homology dial, and the satellites hold station
throughout. Colors are censat annotation, which judges the map and never
builds it.*

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
- The exact-distance matrix is explorable: annotation margins (chromosome +
  satellite class), zoom/pan to single-pair resolution, and click-a-pair to
  light both loci up on the map and territory (e.g. one click on an
  off-diagonal speck reads: chr13 rDNA × chr21 rDNA, Jaccard 0.256).

Headline numbers from the shipped run (full details in `out/metrics.json` and
on the atlas page):

| what | result |
|---|---|
| bins / clusters / noise | 31,185 · 133 · 33.3% |
| satellite-cluster purity (size-weighted) / recovery | 90.7% / 90.4% |
| αSat HOR bin → chromosome, leakage-free¹ (chance) | **92.0%** (7.9%) |
| rDNA bin → chromosome — the predicted weak side¹ (chance) | 70.4% (35.7%) |
| satellite family recovery, leakage-free¹ (chance) | **96.8%** (29.0%) |
| satellite semantic accuracy across 4 UMAP seeds | 94.3–94.4% every seed |
| satellite semantic accuracy at half / quarter sketch depth | 96.3% / 96.4% |
| HDBSCAN agreement at comparable granularity (full 10× grid) | ARI 0.61 (0.34) |
| model vs exact distance, 1 Mb pairs (Spearman) | ρ 0.54 |
| live-HOR bins: tandem period from hash spacing alone | **1,364 bp = 7.99 × the 171 bp monomer** (85% periodic) |
| rDNA bins: tandem period from hash spacing alone | **44.8 kb** (the ~45 kb unit) |
| mainland dialect R² (unified 12-D protocol): k=21 → k=17 | 0.465 → **0.525** (satellites intact) |
| consensus k17⊕k21: dialect R² / HDBSCAN noise | 0.512 / **2.2%** (vs 33.3% baseline) |
| info k15⊕k21 / k=15 alone: dialect R² | 0.585 / 0.661 (composition end) |

¹ classification is a similarity-weighted vote among cosine neighbors with
same-chromosome bins within ±5 bins **excluded**, so a bin's own array cannot
vote for it — no adjacency leakage. (The naive CV formulation scored 95.7% on
αSat; removing the leakage costs under 4 points — the signal is cross-array,
not positional.)

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
- **The thread through word-space** (`atlas3d.html`, prototype): a 3-D panel
  where the genome runs left→right and the current embedding plane rotates
  around it — chromosome threads dive from the mainland into their satellite
  islands, and a ▶ fly mode walks the genome with the map and territory
  tracking in sync. Genome-wide or single-chromosome. Deliberately modular:
  one fragment file injected at a template marker, reading the page's live
  state (so it inherits every color mode, selection, and the k slider);
  deleting `kmer_clust/atlas3d.html` removes the feature entirely.
- **The k-ladder** (`kladder.py`): one full re-sketch and embedding per word
  length k ∈ {15,17,19,21,23,25}, each layout Procrustes-aligned to k=21 and
  quantized in a shared frame, so the atlas can expose **k as a slider** that
  morphs the map through vocabularies (with per-k judge metrics displayed
  live). k=15 is included deliberately — 4¹⁵ is about the genome's own size,
  so the slider lets you watch the random-collision floor arrive.
- **Multi-k vocabularies** (`multik_lab.py`): paired views concatenate two
  per-k models, each block L2-normalized so the pair's cosine is exactly the
  mean of the two vocabularies' cosines. Measured verdict: adjacent horizons
  agree and sharpen clusters (**consensus k17⊕k21**: dialect R² 0.51, 2.4%
  HDBSCAN noise); complementary horizons maximize information at the cost of
  flat-cluster confidence (**info k15⊕k21**: R² 0.59, best-organized
  euchromatin); concatenating all six k's is dominated by both — redundant
  middle horizons average the ends away. Both winners ship as atlas views.

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

## Projection (prototype)

`project.py` (`make project`) freezes the model — shared-hash universe, IDF,
SVD basis — and drops any assembly's 100 kb windows onto the map with **two
signals on purpose**: model cosine in the shared-vocabulary space places a
window in word-space (its dialect neighborhood), while **exact sketch Jaccard
against the full store** (private words included) decides its loci — because
euchromatic locus identity lives precisely in the private vocabulary the
model excludes. Self-projection of 50 kb-offset T2T windows: **97% top-1 /
99% top-3** on euchromatin and 70% / 92% on satellite+acrocentric (residue =
within-array ties), with cosine alone managing only 30% — the gap is the
point. Projected 4 Mb chr21-acrocentric slices of HG00097 (both haplotypes)
and NA19909: the CHM13 control maps 40/40 windows to chr21, while the HPRC
haplotypes send 20–70% of their windows to chr13/chr14 as best exact home —
the pan-acrocentric commons, measured per window, with runner-up loci and
vocabulary coverage (95–98%) carrying the ambiguity. The atlas gains a
four-stage morphing panel: assembly coordinates → word-space placement →
T2T loci (dots lifted into lanes, solid stem to the best locus, thin stems
to runner-ups) → fine placement on an auto-zooming excised axis: only the
matched chromosomes remain, spanning the placements' min–max (+2 Mb), at
one uniform px-per-Mb scale with break marks, a scale bar, an "N× zoom"
readout, and a per-window exact-Jaccard bar. Click pins a window
(world-line through all four stations, loci lit in the main views); a
"paths" toggle draws every world-line at once, and "zoom locus" opens
per-locus J-bar detail for the pinned window. Modular as `project.html`,
delete to remove.

![Projection tour, NA19909 hap2](docs/media/tour_na19909_h2.gif)

*NA19909's chr21-acrocentric slice touring the four stations, closing on
the world-line finale and the direct assembly↔placement ribbon. Its fine
placements land on an excised chr13 ∥ chr14 ∥ chr21 axis at 158× zoom —
most of this haplotype's "chr21" slice finds its best exact home on
CHM13's chr13.* (Control: [tour_chm13_slice.gif](docs/media/tour_chm13_slice.gif),
one contiguous chr21 segment at 395×.)

### Showcase: famous loci, fished out of a whole haplotype

To stress the projector beyond one slice, the entire HG002 paternal
assembly (HPRC year-1, 29,337 windows of 100 kb) was projected in ~5 min,
and windows whose best **exact** locus falls in a famous T2T region were
pulled out — no alignment, no assembly annotation, the projector locating
landmarks by vocabulary alone. The regions that made the cut read as a
divergence dial across locus biology (exact-Jaccard median, high to low):

| set | windows → top chrom | median J | order τ¹ | what it shows |
|---|---|---|---|---|
| **Yq12 heterochromatin** | 44/44 → chrY | 0.88 | +0.90 | CHM13's chrY *is* HG002's — a near-self control; runner-up loci are other spots inside the same satellite ocean, yet the windows still project *in order*: vocabulary resolves a gradient along the array |
| **IGH** (chr14) | 13/13 → chr14 | 0.87 | +1.00 | germline-variable between people; window J spans 0.25–0.95 |
| **SMN1/SMN2 · 5q13** | 8/8 → chr5 | 0.79 | +0.54 | the spinal muscular atrophy locus: 7 of 8 windows carry *two* strong homes ~0.9 Mb apart — the near-identical twin blocks — and assembly order shuffles between them |
| **22q11.2 · DiGeorge/VCFS** | 35/35 → chr22 | 0.79 | −0.91 | the most common microdeletion syndrome region; LCR22 segdups multi-map, and the contig is reverse-stored (another ribbon X) |
| **KIR / LRC · 19q13.4** | 9/9 → chr19 | 0.46 | −1.00 | the NK-cell immunity complex: low J throughout, and the KIR window itself is 20% *novel* to CHM13 (coverage 0.80) — KIR haplotypes differ in gene content; the contig is reverse-stored, perfectly (a flawless ribbon X) |
| **MHC / HLA** (chr6) | 44/44 → chr6 | 0.63 | +1.00 | the genome's most polymorphic region — divergent in sequence, colinear in structure, so every window still lands home (one hypervariable window drops to J = 0.09 and *still* places on chr6) |
| **8p23.1 defensins** | 44/44 → chr8 | 0.56 | +0.67 | inversion flanked by copy-number-variable segdup clusters — the one showcase locus whose window *order* shuffles |
| **MAPT / 17q21.31** | 15/15 → chr17 | 0.35 | +1.00 | the H1/H2 inversion polymorphism — most word-divergent locus here, yet placed exactly and in order |
| **chr13 centromere entry** | 29/30 → chr13, 1 → chr21 | 0.53 | −0.76 | a *data-driven* discovery: the assembly's lowest-coverage contiguous megabase turned out to be a contig descending into the chr13 live αSat array — coverage crashes 1.00 → 0.55 as HOR variants CHM13 lacks appear (a personal centromere), one window lands on the chr13/21-shared HOR family, and the contig ends inside the array. The contig is [JAHKSE010000070.1](https://www.ncbi.nlm.nih.gov/nuccore/JAHKSE010000070.1) — 97.5 Mb spanning the whole chr13 q-arm, opening on telomere repeat and dying in the centromere; the set shows its final 3 Mb (positions 94.4–97.4 Mb) |

The centromere walk was not hand-picked: scanning all 29,337 windows for
the contiguous run CHM13's vocabulary knows least surfaced it — the
coverage meter doubling as a novelty detector.

¹ Kendall correlation between a window's index along the assembly segment
and its fine placement along the excised axis — a synteny readout with no
alignment anywhere. The atlas paints every dot by assembly order
(blue → orange), so collinearity is visible as a smooth gradient at fine
placement; τ is printed on the axis. A **direct** toggle (and each GIF's
closing beat) morphs the two inner stations away and straightens every
world-line into an assembly ↔ placement ribbon — a classic synteny
ribbon plot, derived without an aligner: parallel ribbons = collinear,
one full crossing = a reverse-oriented contig, a weave = the commons. The CHM13 self-control scores a
perfect +1.00, and the three chr21-acro haplotype slices form a
scrambling dial: HG00097 h1 +0.95, h2 +0.71, NA19909 h2 **−0.19** — the
acrocentric commons, now measurable as lost ordering.

Each set is one *contiguous* assembly segment (windows between the first and
last region hit all stay in), so the assembly axis has no artificial holes.

### chrY, one contig per sample

The year-1 HiFi drafts have no continuous chrY — palindromes and satellite
arrays break it into pieces (HG002's Y arrives as 34 contigs totalling
~51 Mb against the reference's 62.5; the missing ~11 Mb is collapsed
satellite). So each sample shows its single most informative Y contig,
picked by a fixed rule: most windows × widest reference span walked,
skipping contigs already featured in another set.

| set | contig | windows → top | median J | order τ |
|---|---|---|---|---|
| **HG002 pat** (the reference Y *is* HG002's) | 13 Mb walk, chrY 6–19 | 88/89 → chrY | 0.36 | **−0.98** |
| **HG005 pat** (a different Y lineage) | 36 Mb walk, chrY 24–60 | 47/47 → chrY | 0.64 | +0.97 |

![HG002 chrY contig tour](docs/media/tour_y_hg002.gif)

Each row is a one-line finding. **HG002's contig walks the reference
almost perfectly backward** (τ −0.98): the draft stored this ampliconic
contig in reverse orientation — contig strand is arbitrary — and because
sketches use canonical k-mers, placement is strand-blind and lands every
window correctly anyway; the *order* readout is what exposes the flip,
as a reversed color gradient. Its low J (0.36) is the ampliconic terrain:
window-grid offset caps J for unique sequence and the draft's amplicon
copies add real damage (satellite loci elsewhere on the same Y sit at
~0.8, offset-immune). One window even three-way ties across a duplicon
family shared with the acrocentric p-arms (chr14:2.9 / chr15:4.8 /
chrY:11.3 Mb at J 0.35–0.40) — the acrocentric commons, seen from the Y. **HG005's contig walks 24–60 Mb of a
different Y lineage in order** (τ +0.97) at lower overlap — *divergence
without disorder*. A multi-fragment variant that scaffolds *all* of a sample's Y
contigs by median placement gave τ +0.89 for both samples (and flagged
the X-transposed region via windows landing at chrX ~90 Mb); the
one-contig view is what ships, for clarity. Reproduce with
`python -m kmer_clust.project ychrom <assembly.fa.gz> "<label>"`.

Each set ships in the atlas dropdown with a one-line blurb, and every
example has a shareable tour GIF in [docs/media/](docs/media/)
(e.g. [tour_mhc.gif](docs/media/tour_mhc.gif),
[tour_yq12.gif](docs/media/tour_yq12.gif)). Reproduce with
`python -m kmer_clust.project showcase <assembly.fa.gz> "<label>"`.

## Deliberately out of scope

Clustering HPRC assemblies themselves (that's kmer_dust's job — here they
only *project* onto the frozen CHM13 map), gene annotation, alignment
baselines, Snakemake/HPC. Future fun that the store already supports:
more order-aware readouts beyond periodicity, projection at finer window
sizes, panels of many haplotypes per locus.
