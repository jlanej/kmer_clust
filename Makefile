PY := .venv/bin/python

.PHONY: all sketch matrix embed pairwise annotate analyze figures site test clean

all:
	$(PY) -m kmer_clust all

sketch matrix embed pairwise annotate analyze periods figures site:
	$(PY) -m kmer_clust $@

test:
	$(PY) -m pytest -q

clean:
	rm -rf out
