# Where this tool can push the state of the art

A literature-grounded map (August 2026) from the field's documented pain
points in hard-region assembly and assembly-to-reference comparison to
concrete uses of this repository's machinery. Companion to the README's
[Methods, precisely](README.md#methods-precisely).

## What the field says is broken

**Alignment collapses exactly where assemblies are hardest.**
[Logsdon et al. 2024](https://www.nature.com/articles/s41586-024-07278-3)
(*Nature*), completing all centromeres of a second human genome, report
that **45.8% of centromeric sequence cannot be reliably aligned using
standard methods** — new α-satellite HORs emerge per individual, with
≥4.1× the SNV rate of unique flanks and up to 3× size differences; a
[2026 follow-up across many genomes](https://www.nature.com/articles/s41586-026-10841-9)
extends this population-wide.
[Winnowmap2](https://pubmed.ncbi.nlm.nih.gov/35365778/) improved repeat
mapping, but active HOR arrays only become mappable at 40–60 kb reads and
diverged/monomeric regions [remain largely unmappable below ~200 kb
reads](https://genome.cshlp.org/content/early/2022/11/15/gr.276871.122.full.pdf).
The response in the evaluation literature is a turn *away from alignment
itself*: a 2026 framework from the Giunta lab
([arXiv:2606.11276](https://arxiv.org/abs/2606.11276)) scores centromere
assemblies by comparing inter-motif distance *distributions* (KL
divergence) precisely because "conventional benchmarking relies on
sequence alignment, which becomes problematic in regions of high
homogeneity and divergence."

**Assembly QC has a stated blind spot for balanced events.**
[Merqury](https://genomebiology.biomedcentral.com/articles/10.1186/s13059-020-02134-9)
gives k-mer QV and completeness but no placement, order, or orientation.
[HMM-Flagger](https://www.researchgate.net/publication/401477515_Evaluating_genome_assemblies_with_HMM-Flagger)
detects structural errors from mapped-read coverage, and its authors state
it is **"unlikely to detect balanced events … like inversions"** unless
coverage happens to ripple. Long-read misassembly detectors (e.g.
[LRMD, 2025](https://www.biorxiv.org/content/10.1101/2025.11.07.686952.full.pdf))
still start from alignments.

**Hard-region read recruitment is bespoke.**
[centroFlye](https://www.nature.com/articles/s41587-020-0582-4) recruits
centromeric ultralong reads per chromosome via HOR fitting-alignment plus
curated unique/rare 19-mer markers — powerful, but hand-built per target
and annotation-heavy. There is no general "which locus does this read's
vocabulary belong to" service.

**The acrocentric commons is an active frontier.**
[Guarracino et al. 2023](https://www.nature.com/articles/s41586-023-05976-y)
(*Nature*) defined pseudo-homologous regions (PHRs) where the five
acrocentric short arms recombine as a community — with Robertsonian
breakpoints inside them — and a
[Dec 2025 follow-up](https://www.biorxiv.org/content/10.64898/2025.12.16.694519v1.full)
measures de novo mutation and recombination there.

**The closest alignment-free relatives do adjacent, different jobs.**
[ModDotPlot](https://academic.oup.com/bioinformatics/article/40/8/btae493/7729118)
(modimizer sketches) draws self-identity heatmaps of repeats — visualization,
not cross-genome placement.
[MashMap3](https://github.com/marbl/MashMap)/[wfmash](https://github.com/waveygang/wfmash)
produce approximate mappings from minmer sketches — mapping coordinates,
without novelty accounting or an order/orientation diagnostic, and with the
base-level stage bounded by alignment cost.
[ntSynt](https://link.springer.com/article/10.1186/s12915-025-02455-w)
builds minimizer-graph synteny blocks between whole genomes — block-level
synteny, not per-window placement, novelty, or satellite-interior behavior.

## What this repository already demonstrates against those gaps

- **Placement *inside* satellite DNA** without alignment: αSat HOR bins
  identify their chromosome at 92.0% (chance 7.9%) under an
  adjacency-excluded protocol; a whole Yq12 contig walks the reference in
  order (τ +0.90) at J ≈ 0.88.
- **Balanced-event detection** — the Flagger blind spot — for free: a
  reverse-stored contig reads τ −0.98/−1.00 as a full ribbon crossing;
  order shuffling at 8p23.1 and SMN reads as intermediate τ.
- **Per-window novelty vs the reference** with an unbiasedness argument
  (the sketch lottery is deterministic in the k-mer): the chr13
  centromere entry — coverage crashing 1.00 → 0.55 as personal HOR
  variants appear — is Logsdon's per-individual centromere divergence,
  measured in seconds and with no HOR annotation (the arXiv framework
  above requires motif calls; this does not).
- **The commons, per window**: the acro slices quantify PHR-style
  promiscuity as a scrambling dial (τ +0.95 → −0.19) on single 100 kb
  windows.
- **Speed**: ~10 ms/window; a whole haplotype in ~5 min on a laptop,
  cached.

## Proposed real-world use cases, ranked by (impact × feasibility here)

1. **`triage` — a one-command draft-assembly compass.** ✅ *Shipped — see
   [README § Triage](README.md#triage-the-assembly-compass).* Per contig, from
   one scan: best chromosome and span walked, orientation (sign of τ),
   fraction of windows confidently placed, novelty profile (runs of
   low-coverage windows = candidate personal/unrepresented sequence or
   artifact), satellite context, and *contig-end terrain* (what fraction
   of contig ends die in satellite — "death by centromere" as a
   statistic). Complements Merqury (needs no reads) and Flagger (needs no
   alignment; catches inversions). All machinery exists in
   `project.py`; output = TSV + a ribbon summary panel. Demo on the
   HG002/HG005 drafts already scanned.
2. **Ultralong-read locus recruitment.** Reads ≥100 kb are the same size
   as our windows: place raw ONT UL reads by vocabulary (locus + J +
   coverage + runner-ups), benchmark against Winnowmap2 on reads
   simulated from T2T including centromeres. Generalizes centroFlye's
   hand-built marker recruitment into a reference-wide, annotation-free
   service; useful as a pre-assembly binner and a post-assembly
   validation stream.
3. **Misassembly detection, quantified.** Inject synthetic misjoins,
   inversions, and translocations into T2T windows; measure ROC of
   τ-discordance and locus-discordance detection — including inside
   satellite arrays where alignment-based detectors cannot operate.
   Cheap; turns the QC claims into numbers.
4. **PHR cartography across the pangenome.** Run the acro machinery over
   many HPRC haplotypes: per-window commons promiscuity and τ per
   haplotype = a population-scale map of the recombining community,
   directly comparable to Guarracino's PHRs and Robertsonian breakpoint
   zones.
5. **A personal-centromere divergence meter.** The centromere-entry
   coverage profile, computed per sample per chromosome, is a fast
   annotation-free divergence index for exactly the variation Logsdon
   describes — a triage statistic for "how different is this individual's
   centromere from the reference" before anyone attempts alignment.
6. **Immunogenetic haplotype screening (KIR/LRC, MHC, IGH).** The
   per-window novelty + J profile over these loci fingerprints haplotype
   gene content (shown here: the KIR window at 20% novel vocabulary).
   Sub-window resolution would need smaller bins — the store supports
   re-binning without re-sketching.

## How others can use it today

- Any assembly → this atlas: `python -m kmer_clust.project showcase
  <fa.gz> "<label>"` (regions), `ychrom` (chromosome walks); scans cache
  as parquet.
- Any *reference* (not just CHM13): the pipeline is reference-agnostic —
  `make all` against another FASTA rebuilds the store, model, and
  projection kit.
- The sketch store is sourmash-compatible by construction (bit-parity
  tested), so sketches interoperate with the sourmash ecosystem.

## Sources

Logsdon et al. 2024, Nature — https://www.nature.com/articles/s41586-024-07278-3 ·
2026 follow-up — https://www.nature.com/articles/s41586-026-10841-9 ·
Winnowmap2 — https://pubmed.ncbi.nlm.nih.gov/35365778/ ·
Centromere-aware evaluation — https://arxiv.org/abs/2606.11276 ·
Merqury — https://genomebiology.biomedcentral.com/articles/10.1186/s13059-020-02134-9 ·
HMM-Flagger — https://www.researchgate.net/publication/401477515_Evaluating_genome_assemblies_with_HMM-Flagger ·
LRMD — https://www.biorxiv.org/content/10.1101/2025.11.07.686952.full.pdf ·
centroFlye — https://www.nature.com/articles/s41587-020-0582-4 ·
Guarracino et al. 2023 — https://www.nature.com/articles/s41586-023-05976-y ·
Acro de novo 2025 — https://www.biorxiv.org/content/10.64898/2025.12.16.694519v1.full ·
ModDotPlot — https://academic.oup.com/bioinformatics/article/40/8/btae493/7729118 ·
MashMap3 — https://github.com/marbl/MashMap ·
wfmash — https://github.com/waveygang/wfmash ·
ntSynt — https://link.springer.com/article/10.1186/s12915-025-02455-w
