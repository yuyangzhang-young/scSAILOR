"""The light menus of the six models, and the rule that writes the menu of a new architecture.

One map seed; both original models (trained on the synthetic task and on real responses); the full map length. For
each architecture: the junction arms at the rates where its restoring changes and learners sit on Tahoe-100M; the site
of its restoring change where that lies outside the junction; fine-tuning at its own rates; and the whole network as
the reference, at both map seeds (``REFERENCE_SEEDS``).
"""
from . import rules
from .rules import FRESH, JUNCTION, MAP_SEEDS

PRIMARY_SEED, SECOND_SEED = MAP_SEEDS
REFERENCE_SEEDS = MAP_SEEDS
WINDOW = 0.02

LIGHT_MENUS = {  # model -> [(cut, arms, rates)]
    'summary': [('projections', ('a', 'f', 'e', 'e_mean'), (1, 30)), ('projections', ('a',), (3,)),
                ('input', ('a',), (10,))],
    'deepsets': [('phi', ('a', 'f', 'e'), (1, 10)), ('input', ('a',), (3,))],
    'attention_model': [('head', ('a', 'f', 'e'), (1, 30)), ('block3', ('a', 'f'), (1,)),
                        ('input', ('a',), (1,))],
    'state': [('head', ('a', 'f', 'e'), (1, 30)), ('junction', ('a', 'e', 'f'), (1,)),
              ('input', ('a',), (1,))],
    'cpa': [('decoder', ('a', 'f', 'e'), (1, 30)), ('input', ('a',), (1,))],
    'prepr': [('prediction', ('a', 'f', 'e'), (1, 30)), ('input', ('a',), (1,))],
}
DIAGNOSTIC = {model: f'{cut}/a/x1' for model, cut in JUNCTION.items()}   # handed on when nothing is confirmed


def in_menu(model, cut, arm, rate):
    return any(cut == c and arm in arms and float(rate) in (float(r) for r in rates) for c, arms, rates in LIGHT_MENUS[model])


def is_reference(cut, arm):
    return cut == 'input' and arm in ('a', FRESH)


def runs(model):
    """The light map's runs for one fold: (half, seed, cut, arm, rate)."""
    out = []
    for cut, arms, rates in LIGHT_MENUS[model]:
        for arm in arms:
            for rate in rates:
                starts = ('synthetic',) if arm == FRESH else ('synthetic', 'natural')
                seeds = REFERENCE_SEEDS if is_reference(cut, arm) else (PRIMARY_SEED,)
                out += [(half, seed, cut, arm, float(rate)) for half in starts for seed in seeds]
    return out


def derive_menu(runs, model, junction):
    """The light menu of a new architecture, from its full maps on development data (validation panel only).

    A cell learns a case when it reaches LEVEL within the margins in both map seeds, in a case not solved at the start.
    1. The junction cut: arms a, f and e, and any other arm carried there, at x1 and the rates of the junction cells
       carried; with none carried, at x1 and the highest rate at which fine-tuning at the junction finished.
    2. Every other cut with a carried cell: arms a, f and e where they exist at that cut, at the carried rates.
    3. Fine-tuning (arm a) at the junction, at the rates where it learns a case beyond those of step 1.
    4. The whole network at the rate where it learns the most cases (ties: the lower rate);
       x1 when it learns none.
    On the Tahoe-100M full maps the rule gives the six menus of LIGHT_MENUS."""
    runs = runs[(runs.model == model) & (runs.case == 'parent')]
    cells = rules.cells(runs)
    start = rules.solved_at_start(runs)
    carried = set()
    for key, menu in cells.groupby(['dataset', 'model', 'fold', 'half', 'target']):
        if key in start:
            kind, cell = rules.carried(menu, model, start[key], junction=junction)
            if kind == 'qualified repair':
                cut, arm, rate = cell.split('/')
                carried.add((cut, arm, float(rate[1:])))
    open_cases = cells[[not start.get(k, True) for k in zip(cells.dataset, cells.model, cells.fold, cells.half, cells.target)]]
    open_cases = open_cases[open_cases.seed.isin(MAP_SEEDS)].assign(ok=lambda t: (t.Q >= rules.LEVEL) & t.margins)
    learned = open_cases.groupby(['dataset', 'fold', 'half', 'target', 'cut', 'arm', 'rate']).ok.agg(['size', 'all']).reset_index()
    learned = learned[(learned['size'] == len(MAP_SEEDS)) & learned['all']]
    at_junction = {(arm, rate) for cut, arm, rate in carried if cut == junction}
    finished = runs[(runs.cut == junction) & (runs.arm == 'a') & (runs.status == 'complete')].rate
    rates = sorted({1.0} | ({r for _, r in at_junction} or {finished.max()}))
    arms = ('a', 'f', 'e') + tuple(sorted({a for a, _ in at_junction} - {'a', 'f', 'e'}))
    menu = [(junction, arms, tuple(rates))]
    for cut in sorted({c for c, _, _ in carried} - {junction}):
        present = set(runs[runs.cut == cut].arm)
        menu.append((cut, tuple(a for a in ('a', 'f', 'e') if a in present), tuple(sorted({r for c, _, r in carried if c == cut}))))
    fine = sorted(set(learned[(learned.cut == junction) & (learned.arm == 'a')].rate) - set(rates))
    if fine:
        menu.append((junction, ('a',), tuple(fine)))
    whole = learned[(learned.cut == 'input') & (learned.arm == 'a')].groupby('rate').size()
    rate = float(whole[whole == whole.max()].index.min()) if len(whole) else 1.0
    return menu + [('input', ('a',), (rate,))]


def register(model, menu, junction):
    """Add a new architecture: its light menu (from derive_menu) and its junction cut, which also fixes its fixed
    intervention (junction, a, x1). Selection, run lists and readings then treat it as the six above."""
    LIGHT_MENUS[model] = menu
    JUNCTION[model] = junction
    DIAGNOSTIC[model] = f'{junction}/a/x1'


def cells_of(menu):
    """The (cut, arm, rate) cells of a menu."""
    return {(cut, arm, float(rate)) for cut, arms, rates in menu for arm in arms for rate in rates}
