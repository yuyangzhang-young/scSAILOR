"""Read a localization map folder: the run folders that ``scsailor.localization.starts`` writes.

A map folder holds one screen and fold. Each run folder has ``RUN_RECORD.json`` (its ``row`` names the model, half,
fold, seed, case, cut, arm and rate), ``EVAL_PARENT.json`` (the original model, read before training) and, once the run
has finished, ``COMPLETE.json`` with ``EVAL.json`` (the read-out at the end) or ``DIVERGED.json``. A folder with neither
has not finished and is left out, so its cell counts as not run.

``read_map`` returns only the validation and seen panels, the panels the light version selects on; a diverged run
keeps its original model's reading and has no end values. The held-out panel is read by ``heldout_q``, cell by cell,
only for cells already chosen on the validation panel.
"""
import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from ..readings import TARGETS as BITS

PANELS = ('validation', 'seen')


def read_map(root, dataset):
    """One row per finished run: identity, status, end and original Q of each bit and the raw error, per panel."""
    rows = []
    for run in sorted(p for p in Path(root).iterdir() if p.is_dir()):
        complete, diverged = (run / 'COMPLETE.json').is_file(), (run / 'DIVERGED.json').is_file()
        if not (complete or diverged):
            continue
        row = json.loads((run / 'RUN_RECORD.json').read_text())['row']
        parent = json.loads((run / 'EVAL_PARENT.json').read_text())['supports']
        end = json.loads((run / 'EVAL.json').read_text())['after']['supports'] if complete else None
        record = dict(dataset=dataset, fold=row['fold'], run=str(run), model=row['model'], half=row['half'],
                      seed=int(row['seed']), case=row.get('case', 'parent'), cut=row['cut'], arm=row['arm'],
                      rate=float(row['rate_multiple']), status='complete' if complete else 'diverged')
        for panel in PANELS:
            for bit in BITS:
                record[f'{panel}_Q{bit}'] = end[panel]['bits'][bit]['Q'] if complete else np.nan
                record[f'parent_{panel}_Q{bit}'] = parent[panel]['bits'][bit]['Q']
            record[f'{panel}_raw'] = end[panel]['mse']['raw_O50'] if complete else np.nan
            record[f'parent_{panel}_raw'] = parent[panel]['mse']['raw_O50']
        rows.append(record)
    return pd.DataFrame(rows)


@lru_cache(maxsize=None)
def _heldout_bits(run):
    return json.loads((Path(run) / 'EVAL.json').read_text())['after']['supports']['heldout']['bits']


def heldout_q(run, target):
    """Held-out recovery of one target at the end of one completed run."""
    return float(_heldout_bits(str(run))[str(target)]['Q'])
