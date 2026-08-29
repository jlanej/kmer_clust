"""Streaming FASTA -> 2-bit code arrays, with an on-disk per-chromosome cache."""

import gzip
import json
import time
import urllib.request
from pathlib import Path

import numpy as np

from .config import DATA, GENOME_URL, Params
from .fracminhash import encode_sequence

CODES_DIR = DATA / "codes"
MANIFEST = CODES_DIR / "manifest.json"


def fetch_genome(params: Params) -> Path:
    """Download the genome if it is not already on disk."""
    path = params.genome
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"downloading {GENOME_URL} -> {path}")
    urllib.request.urlretrieve(GENOME_URL, path)
    return path


def iter_fasta_codes(path: Path):
    """Yield (name, uint8 codes) per record, streaming; handles .gz."""
    opener = gzip.open if str(path).endswith(".gz") else open
    name = None
    chunks: list[bytes] = []
    with opener(path, "rb") as fh:
        for line in fh:
            if line.startswith(b">"):
                if name is not None:
                    yield name, encode_sequence(b"".join(chunks))
                name = line[1:].split()[0].decode()
                chunks = []
            else:
                chunks.append(line.rstrip())
    if name is not None:
        yield name, encode_sequence(b"".join(chunks))


def iter_chrom_codes(params: Params, use_cache: bool = True):
    """Yield (name, codes) for analysis chromosomes, caching codes as .npy."""
    exclude = set(params.exclude_chroms)
    gid = f"{params.genome.name}:{params.genome.stat().st_size}" \
        if params.genome.exists() else ""
    if use_cache and MANIFEST.exists():
        manifest = json.loads(MANIFEST.read_text())
        if manifest.get("genome", gid) != gid:
            print("  codes cache was built from a different genome; rebuilding")
            manifest = None
    else:
        manifest = None
    if manifest is not None:
        for name in manifest["chroms"]:
            if name in exclude:
                continue
            yield name, np.load(CODES_DIR / f"{name}.npy")
        return

    CODES_DIR.mkdir(parents=True, exist_ok=True)
    chroms = []
    t0 = time.time()
    for name, codes in iter_fasta_codes(fetch_genome(params)):
        chroms.append(name)
        if use_cache:
            np.save(CODES_DIR / f"{name}.npy", codes)
        print(f"  parsed {name} ({codes.size/1e6:.1f} Mb, t={time.time()-t0:.0f}s)")
        if name not in exclude:
            yield name, codes
    if use_cache:
        MANIFEST.write_text(json.dumps({"chroms": chroms, "genome": gid}))
