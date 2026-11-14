"""The growth programs: the MSigDB Hallmark sets E2F targets, G2M checkpoint and mitotic spindle (release 2025.1), on a
Tahoe-100M fold's 2,000-gene panel. Their union is the growth genes.
"""
from functools import cache

import numpy as np
import pandas as pd

from ..origin_resolution import ROOT

__all__ = ["GROWTH", "HALLMARK", "MIN_GENES", "gene_sets", "panel", "masks", "growth"]

GROWTH = ('HALLMARK_E2F_TARGETS', 'HALLMARK_G2M_CHECKPOINT', 'HALLMARK_MITOTIC_SPINDLE')
HALLMARK = ROOT / 'msigdb-hallmark/h.all.v2025.1.Hs.symbols.gmt'
MIN_GENES = 10   # on Parse PBMC and the genetic screens, a program is read when it has this many genes on the panel
PANELS = ROOT / 'tahoe-100m-gene-panels'


def gene_sets(text):
    """Set name -> its gene symbols, from the text of a GMT file (name, description, genes; tab-separated)."""
    return {line.split('\t')[0]: set(line.split('\t')[2:]) for line in text.splitlines()}


def panel(fold):
    """The fold's gene symbols, in the order of the arrays' gene axis."""
    return pd.read_csv(PANELS / f'tahoe_{fold}.csv').gene_id.astype(str).to_numpy()


def masks(genes, sets):
    """Per growth program, which of ``genes`` it holds; and their union (the growth genes)."""
    found = [np.isin(genes, sorted(sets[name])) for name in GROWTH]
    return found, np.logical_or.reduce(found)


@cache
def growth(fold):
    """The growth programs on the fold's panel (one mask per program) and the growth genes."""
    return masks(panel(fold), gene_sets(HALLMARK.read_text()))
