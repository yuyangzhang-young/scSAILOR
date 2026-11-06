"""Origin resolution: what a restoring change costs on held-out lines, why, and whether any of it pays.

A model's prediction is its anchor (the control mean plus the transferred response) plus its correction. Information
localization hands on the restoring changes that recover a statistic the model missed. Origin resolution follows each
change through four steps, one group of modules each:

1. recovery and cost: ``il_cells``    the change trained on measured responses, beside its original model, continued
                                      training and training longer;
2. attribution: ``attribution``,      where the error sits and which origin it points to, with the references that add
   ``origins``, ``references``,       like the error (cell sampling of the targets, the between-plate discrepancy, the
   ``probes``                         input set's sampling) and the fixed-weight probes;
3. calibration (recommend or          what to keep, chosen on calibration lines no model of the nested chain has seen: one
   refuse): ``slopes``, ``policies``, slope per layer, the shrinks and kept fractions, the single-model menu, the
   ``card``, ``menu``, ``net_cost``,  comparisons and the recommendation rule against the standard references,
   ``planted``                        the calibrated original, the design-aware baseline and the model's fixed
                                      intervention, and planted changes with a known answer;
4. held-out check: ``comparison``     per held-out line, a sign flip over lines with the seed rule, and the paired line
                                      bootstrap beside it. Held-out lines take no part in a choice and report differences.

``stages`` holds the stages over one dataset's case, ``scoring`` the natural task's risk on the clipped scale and its
additive parts, and ``grids`` reads prediction grids. A dataset enters through its *case* (``stages``) and the *sides* of
``slopes``, ``net_cost`` and ``probes``: its units, its models' prediction grids, its input sets and its documented origin
rows. The modules take loaded objects and the dataset's facts (for example the names of its replicate plates) as
arguments; importing them has no side effects and loads no torch.

Code names and the paper's terms (saved files keep the code names and their original strings):

==============================================================  ==================================================
Code                                                            Paper
==============================================================  ==================================================
parent                                                          original model
nested parent, nested chain                                     nested model
calibration units or lines                                      calibration lines
points, ``*_pts``                                               % of the named total MSE
outer units or lines                                            held-out lines
input bag, bag mean ``b``                                       input set, control mean $S$
anchor ``a``                                                    anchor ($S+r_p$)
no change ``b``, ``nochange``                                   no-change prediction
plate-aware anchor ``A``, ``plate_baseline``                    plate-aware baseline
design baseline ``T`` (``net_cost``)                            design-aware baseline
correction ``d = p - a``; its context part ``P d``              correction; context-level correction
common / differential part                                      common / contrast component
M5, M7, N                                                       context-level correction, perturbation-level
                                                                correction, transferred response
layers 'response', 'line', 'perturbation' (``slopes``)          the transferred response, the context-level and the
                                                                perturbation-level correction
amount ``lam``, ``(lam_c, lam_d)``, ``kappa``                   kept fraction $\\lambda$, $(\\lambda_1, \\lambda_2)$,
                                                                $\\lambda_r$
``alpha``, ``size_fix``                                         kept fraction of a restoring change
``cheapest_fix``, ``cell_line_shrink_A``                        the context-level correction scaled (context-level
                                                                shrink), toward the anchor or toward A
'cell-line shrink', 'cell-line-level shrink' in saved strings   context-level shrink
cheap alternatives, ``ALTERNATIVES``, ``PLATE_ALTERNATIVES``    the original model's shrinks among the standard
                                                                references; with a plate-aware target, the plate-aware
                                                                baseline and the shrinks toward it
``global_parent_shrink``                                        uniform shrink toward the anchor
repair, IL cell, cell                                           restoring change
kind 'control', ``control``, ``*_own_control``                  fixed intervention (negative control)
continuation, ``continued_*``, ``input/a/x1``                   continued training
``training_longer``, ``longer_*``                               training longer
extra or second seed, ``extra_seed*``, ``second_design``        second training run
stages and sets 1, 2, 3 (``LADDER.csv``, ``min_gap_set*``)      the standard references; with the design-aware baseline
                                                                and the original calibrated toward it; with the second
                                                                training run
'field' calibration, ``*_field``                                calibration toward the baseline
gap, ``gap_pts``, ``*_worst_vs_*``                              a reference's error minus the kept change's (positive
                                                                when the kept change has the lower error); worst: over
                                                                references and seeds
comparisons 'standard', 'calibrated', 'design-aware'            the standard references; with the original calibrated
                                                                toward the baseline; the design-aware comparison
``recommend``, ``RULE_MARGIN``, ``RECOMMENDATIONS.csv``         recommend or refuse, at the margin of 0.1% of the
                                                                original model's total MSE on calibration lines
``planted``, ``READ_BACK.csv``                                  planted changes, the known-answer test
``*_standing_matched``                                          the design-aware comparison; standing: recommended
``card.branch``'s ``recommendation``                            recommend or refuse, for a model's action
ledger (``LEDGER.csv``)                                         the accounting, in both orders
``price``                                                       not in the paper: cost at full strength per unit the
                                                                best size returns
cosine, best scale below one                                    alignment; over-sized
``H``, ``dependence``, ``E``, ``Delta_E``                       benefit, dependence, added cost, what a probe removes of
                                                                it
``output_shrink``                                               matched shrink
``repair_increment`` energy                                     increment energy
``true``, ``train_avg``, ``other_plate``                        own controls, line-agnostic control mean, other-plate
                                                                mean
own, ``pooled``                                                 own and line-agnostic controls
model ``summary``, ``deepsets``, ``attention_model``            summary-statistics model, DeepSets model, attention
                                                                model
model ``state``                                                 State
saved ``attention``, and ``state`` in localization's runs       attention model
saved ``official_state``                                        State
``'about right'``, ``'too large'`` of the label ``d1c``        size matched; over-sized
card, ``cards``, ``CARD_*``, ``CARDS_*``                        the readings of one model, or of one change, on a fold
Q1-Q4 in the ``CARDS_V2`` columns                               recovery and cost (Q1, Q2), attribution (Q3), the
                                                                comparisons beside continued training (Q4)
'verified', 'not confirmed', ``verification`` in saved strings  the held-out check: the seed rule on held-out
                                                                differences, a reading and not a test of a choice
==============================================================  ==================================================
"""
__all__ = ["attribution", "card", "comparison", "grids", "il_cells", "menu", "net_cost", "origins", "planted", "policies",
           "probes", "references", "scoring", "slopes", "stages"]
