"""Attribution's rows: one per candidate origin of a failure to transfer, and the headline's line 3.

Origins are coded by kind: M (the model: M5 its context-level correction, M7 its perturbation correction), D (the
data: D1 cell sampling of the scored targets, D2 the between-replicate discrepancy, D7 a systematic plate or campaign
difference, ...) and T (the task set-up: T2 the input bag's sampling, T3 the simple references, T7 unequal treatment,
...), with O for the evaluation itself. Rows with computed evidence take it from the attribution stage (``ev``); the
others carry a dataset's documented status (``static``: origin -> (status, evidence, next check)), which a binding
supplies. A documented property is never a lead: it is supported only by its own test.
"""
__all__ = ["ORIGINS", "contrast_row", "rows", "line_three"]

ORIGINS = ('M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8', 'D9',
           'T1', 'T2', 'T3', 'T4', 'T5', 'T6', 'T7', 'T8', 'O1', 'O2')


def contrast_row(origin, d, kind=None, share=None):
    """A model contrast's row from its test (comparison.detect), its alignment reading and its Shapley share."""
    if d['level'] == 'case-level':
        status = 'case-level'
    else:
        status = 'supported' if d['detected'] else 'not detected'
    row = {'origin': origin, 'status': status, 'contrast_total_pts': d['total_pts'],
           'contrast_by_seed_pts': ' / '.join(f'{v:+.2f}' for v in d['seed_totals_pts']), 'p_signflip': d.get('p'),
           'floor': d['floor'], 'seed_rule': d['seed_rule'], 'bootstrap_call': (d.get('bootstrap_call') or {}).get('call')}
    if share is not None:
        row['shapley_share_pts'] = share
    if kind is not None:
        row['evidence'] = kind
    return row


def rows(ev, static, fold1_only=(), seed_permutation='not tested (precondition unavailable)'):
    """All origin rows for one model and fold. ``ev`` holds the stage's computed evidence; ``static`` the dataset's documented
    rows; ``fold1_only`` the documented rows to mark as read on fold 1 (when another fold is read);
    ``seed_permutation`` what O1 says of the seed-permutation statistic."""
    out = []
    for origin in ORIGINS:
        if origin in ('M5', 'M7'):
            out.append(contrast_row(origin, ev['detections'][origin], ev['kind'][origin], ev['shares']['total'].get(origin, 0.)))
        elif origin == 'D1':
            f = ev['references'].get('D1')
            out.append({'origin': 'D1', 'status': 'reference' if f else 'not computed',
                        'evidence': (f"floor's share of the parent's error: total {f['total']:.3f}, common {f['common']:.3f}, "
                                     f"drug difference {f['differential']:.3f}") if f else 'no per-unit floor on this side',
                        'next_check': '' if f else 'the local D1 computation (references/)'})
        elif origin == 'D2':
            out.append({'origin': 'D2', **ev['d2']})
        elif origin == 'D4' and 'd4' in ev:
            out.append({'origin': 'D4', **ev['d4']})
        elif origin == 'D7':
            out.append({'origin': 'D7', **ev['d7']})
        elif origin == 'D8' and 'd8' in ev:
            out.append({'origin': 'D8', **ev['d8']})
        elif origin == 'T1' and 't1' in ev:
            out.append({'origin': 'T1', **ev['t1']})
        elif origin == 'T2':
            g = ev['references']['T2']
            out.append({'origin': 'T2', 'status': 'reference',
                        'evidence': f"bag-against-pool gap: {g['common']:.3f} of the parent's common error (common part only)"})
        elif origin == 'T3':
            r = ev['references']
            out.append({'origin': 'T3', 'status': 'reference',
                        'evidence': f"anchor {r['anchor']['total']:.3f} and no change {r['nochange']['total']:.3f} of the parent's total; "
                                    f"drug-difference part: anchor {r['anchor']['differential']:.3f}, no change {r['nochange']['differential']:.3f}; "
                                    "by gene group of across-line response variability for the parent, anchor and no change in WHERE"})
        elif origin == 'T7' and 't7' in ev:
            out.append({'origin': 'T7', **ev['t7']})
        elif origin == 'O1':
            out.append({'origin': 'O1', 'status': 'reported',
                        'evidence': f"outer units {ev['units']}; sign-flip floor {0.5 ** ev['units']:.3g}; seed spread of the parent's "
                                    f"outer total {ev['seed_spread_pct']:.1f}%; seed-permutation statistic: " + seed_permutation})
        elif origin in static:
            s, e, n = static[origin]
            if origin in fold1_only:
                e = 'fold 1: ' + e
            out.append({'origin': origin, 'status': s, 'evidence': e, 'next_check': n})
        else:
            out.append({'origin': origin, 'status': 'not tested', 'evidence': '', 'next_check': ''})
    return out


def line_three(ev, coincident):
    """Line 3: the statuses of the data and task rows that bear on the remainder, with the factors that coincide with
    identity by design (``coincident``, or None)."""
    parts = []
    for origin in ('D1', 'D2', 'D4', 'D7', 'T1', 'T7'):
        row = next((r for r in ev['rows'] if r['origin'] == origin), None)
        if row is not None:
            parts.append(f"{origin} {row['status']}")
    text = '; '.join(parts)
    if coincident:
        text += f'; confounded by design: {coincident}'
    return text
