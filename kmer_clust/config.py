"""Single source of truth for parameters and paths."""

from dataclasses import dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "data"
OUT = REPO / "out"
DOCS = REPO / "docs"

GENOME_URL = (
    "https://s3-us-west-2.amazonaws.com/human-pangenomics/T2T/CHM13/"
    "assemblies/analysis_set/chm13v2.0.fa.gz"
)

# censat (required, auto-fetched by the annotate stage); SD/telomere/RepeatMasker
# BEDs are optional — read from data/ or a sibling kmer_dust checkout when present.
KMER_DUST_CACHE = REPO.parent / "kmer_dust" / "data" / "cache"
CENSAT_URL = (
    "https://s3-us-west-2.amazonaws.com/human-pangenomics/T2T/CHM13/"
    "assemblies/annotation/chm13v2.0_censat_v2.1.bed"
)


@dataclass
class Params:
    k: int = 21
    base_scaled: int = 20        # densest sketch kept on disk; coarser scaleds are subsets
    embed_scaled: int = 50       # Track A (bin x hash matrix -> SVD -> UMAP)
    bin_bp: int = 100_000        # base bin size (Track A)
    pairwise_bin_bp: int = 1_000_000  # boosted bins for exact pairwise distances (Track B)
    min_df: int = 2              # drop hashes private to a single bin from the model
    svd_dims: int = 128
    umap_neighbors: int = 30
    umap_min_dist: float = 0.08
    hdbscan_min_cluster_size: int = 25
    hdbscan_min_samples: int = 10
    seed: int = 42
    exclude_chroms: tuple = ("chrM",)
    genome: Path = DATA / "chm13v2.0.fa.gz"
    censat: Path = DATA / "chm13v2.0_censat_v2.1.bed"
    repeatmasker: Path = field(
        default=KMER_DUST_CACHE / "chm13v2.0_RepeatMasker_4.1.2p1.2022Apr14.bed"
    )

    @property
    def sketch_npz(self) -> Path:
        return DATA / f"sketch_k{self.k}_s{self.base_scaled}_bin{self.bin_bp}.npz"

    def svd_npz(self, scaled: int | None = None) -> Path:
        scaled = scaled or self.embed_scaled
        return OUT / f"svd_k{self.k}_s{scaled}_bin{self.bin_bp}.npz"

    @property
    def bins_parquet(self) -> Path:
        return DATA / f"bins_{self.bin_bp}.parquet"


PARAMS = Params()
