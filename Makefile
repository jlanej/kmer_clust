PY := .venv/bin/python

.PHONY: all sketch matrix embed pairwise annotate analyze periods project figures site test clean

all:
	$(PY) -m kmer_clust all

sketch matrix embed pairwise annotate analyze periods project figures site:
	$(PY) -m kmer_clust $@

test:
	$(PY) -m pytest -q

clean:
	rm -rf out
