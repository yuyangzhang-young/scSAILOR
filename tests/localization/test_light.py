"""The light version on small hand-written run tables: no study data, no torch.

    python -m pytest tests/localization/test_light.py
"""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from scsailor.localization import readings
from scsailor.localization.light import maps, menus, plan, rules, selection

START = {'0': .6, '1': .1, '5': .6, '6': .1}   # the original model reads both common targets, not the selective mean


def run(cut, arm, rate, seed=31002, q1=.1, half='synthetic', raw=1., **end):
    """One finished map run of the summary-statistics model: validation Q per bit and raw error, with its original
    model's reading START."""
    q = START | {'1': q1} | end
    row = dict(dataset='d', fold='F1', run='', model='summary', half=half, seed=seed, case='parent', cut=cut, arm=arm,
               rate=float(rate), status='complete', validation_raw=raw, parent_validation_raw=1.)
    for bit in START:
        row[f'validation_Q{bit}'], row[f'parent_validation_Q{bit}'] = q[bit], START[bit]
    return row


REFERENCES = [run('input', 'a', 10, seed=seed) for seed in rules.MAP_SEEDS]   # both map seeds, as the light map runs them


def selective_mean_case(rows):
    """The light reading of the selective-mean case of the synthetic-task model; the common targets are solved."""
    runs = pd.DataFrame(rows + REFERENCES)
    cases = selection.light_cases(runs, rules.cells(runs))
    return runs, cases, cases[(cases.half == 'synthetic') & (cases.target == '1')].iloc[0]


class Menus(unittest.TestCase):
    def test_light_maps_have_the_run_counts_of_the_six_menus(self):
        counts = {model: len(menus.runs(model)) for model in menus.LIGHT_MENUS}
        self.assertEqual(counts, {'summary': 22, 'deepsets': 16, 'attention_model': 20, 'state': 22, 'cpa': 16,
                                  'prepr': 16})

    def test_a_registered_architecture_is_read_like_the_one_it_copies(self):
        menus.register('renamed', menus.LIGHT_MENUS['deepsets'], 'phi')
        try:
            self.assertEqual(menus.runs('renamed'), menus.runs('deepsets'))
            self.assertEqual(menus.DIAGNOSTIC['renamed'], 'phi/a/x1')
        finally:
            for table in (menus.LIGHT_MENUS, menus.JUNCTION, menus.DIAGNOSTIC):
                table.pop('renamed')


class Margins(unittest.TestCase):
    def test_margins_hold_the_raw_error_and_on_the_synthetic_task_the_learned_targets(self):
        rows = pd.DataFrame([run('projections', 'a', 1, raw=2.5), run('projections', 'f', 1, **{'0': .4}),
                             run('projections', 'e', 1, half='natural', **{'0': .4}), run('projections', 'a', 30)])
        kept = rules.cells(rows).query("target == '1'").margins.tolist()
        self.assertEqual(kept, [False, False, True, True])

    def test_the_table_margins_are_those_of_the_full_version_run_by_run(self):
        rng = np.random.default_rng(0)
        rows = []
        for half in ('synthetic', 'natural'):
            for _ in range(200):
                row = run('projections', 'a', 1, half=half, raw=float(rng.uniform(0, 4)),
                          **{bit: float(q) for bit, q in zip(START, rng.uniform(0, 1, 4))})
                rows.append(row | {f'parent_validation_Q{bit}': float(q) for bit, q in zip(START, rng.uniform(0, 1, 4))})
        table = rules.cells(pd.DataFrame(rows))

        def support(row, prefix):
            return {'mse': {'raw_O50': row[f'{prefix}validation_raw']},
                    'bits': {bit: {'Q': row[f'{prefix}validation_Q{bit}']} for bit in START}}

        expected = [readings.margins(support(r, 'parent_'), support(r, ''), r['half'], target)
                    for target in rules.TARGETS for r in rows]
        self.assertEqual(table.margins.tolist(), expected)
        self.assertGreater(sum(expected), 0)
        self.assertLess(sum(expected), len(expected))


class Selection(unittest.TestCase):
    CANDIDATES = [run('projections', 'e', 30, q1=.50), run('projections', 'e', 30, seed=31003, q1=.30),
                  run('projections', 'f', 1, q1=.49), run('projections', 'f', 1, seed=31003, q1=.45),
                  run('projections', 'a', 1, q1=.30)]   # outside the 0.02 window of the best

    def test_the_confirmed_cell_with_the_best_two_seed_mean_is_handed_on(self):
        _, cases, case = selective_mean_case(self.CANDIDATES)
        self.assertEqual((case.kind, case.cell), ('qualified repair', 'projections/f/x1'))
        self.assertEqual(sorted(case.checked.split(';')), ['projections/e/x30', 'projections/f/x1'])
        self.assertTrue((cases[cases.target != '1'].kind == 'solved at the start').all())

    def test_a_missing_second_seed_run_leaves_the_case_unconfirmed_and_is_listed(self):
        runs, cases, case = selective_mean_case(self.CANDIDATES[:3] + self.CANDIDATES[4:])
        self.assertEqual((case.kind, case.behaviour), ('unconfirmed', 'waiting for the check'))
        self.assertEqual([r['name'] for r in plan.check_rows(cases, runs)], ['map-summary-synthetic-31003-projections-f-x1'])

    def test_a_failed_check_or_no_candidate_hands_on_the_fixed_intervention(self):
        failed = [dict(r, validation_Q1=.1) if r['seed'] == 31003 else r for r in self.CANDIDATES]
        self.assertEqual(selective_mean_case(failed)[2][['kind', 'cell']].tolist(),
                         ['diagnostic intervention (check failed)', 'projections/a/x1'])
        none = [dict(r, validation_Q1=.1) for r in self.CANDIDATES]
        self.assertEqual(selective_mean_case(none)[2][['kind', 'cell']].tolist(),
                         ['diagnostic intervention', 'projections/a/x1'])


class Plan(unittest.TestCase):
    def test_origin_resolution_lists_every_fit_one_handed_cell_needs(self):
        runs = pd.DataFrame(Selection.CANDIDATES + REFERENCES)
        need = plan.origin_needs('d', 'summary', 'F1', ['projections/f/x1'], runs)
        self.assertEqual(need.item.value_counts().to_dict(),
                         {'natural-task run of the cell': 6, 'ordinary continuation at equal dose': 6,
                          'map run for the cards': 4, 'nested parent': 3,
                          'synthetic +Var parent for the cards (training only)': 1})
        self.assertNotIn('map-summary-synthetic-31002-projections-f-x1', set(need.name))


class ReadMap(unittest.TestCase):
    def test_finished_and_diverged_runs_are_read_and_unfinished_ones_left_out(self):
        support = {'bits': {bit: {'Q': .3} for bit in START}, 'mse': {'raw_O50': 1.}}
        evaluation = {panel: support for panel in ('validation', 'seen', 'heldout')}
        with tempfile.TemporaryDirectory() as root:
            for name, end in (('a-done', 'COMPLETE.json'), ('b-diverged', 'DIVERGED.json'), ('c-running', None)):
                folder = Path(root, name)
                folder.mkdir()
                row = dict(model='summary', half='natural', fold='F1', seed='31002', cut='phi', arm='a', rate_multiple=1)
                (folder / 'RUN_RECORD.json').write_text(json.dumps({'row': row}))
                (folder / 'EVAL_PARENT.json').write_text(json.dumps({'supports': evaluation}))
                (folder / 'EVAL.json').write_text(json.dumps({'after': {'supports': evaluation}}))
                if end:
                    (folder / end).write_text('{}')
            table = maps.read_map(root, 'd')
            self.assertEqual(table.status.tolist(), ['complete', 'diverged'])
            self.assertEqual(table.seed.tolist(), [31002, 31002])
            self.assertEqual(maps.heldout_q(table.run.iloc[0], 1), .3)


if __name__ == "__main__":
    unittest.main()
