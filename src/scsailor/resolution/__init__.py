"""Origin resolution: whether a model's correction, or a restoring change, pays on held-out lines, and why not.

A model's prediction is its anchor (the input bag's mean plus the transferred response) plus its correction.
Origin resolution has four steps. Recovery and cost: a restoring change from information localization against the
original model, on planted and on measured responses. Attribution: the error assigned to the model, the data or the
task set-up. Calibration: on lines that no model of the nested chain has seen, a kept fraction is recommended, or the
candidates are refused. The held-out check: the margins of that choice on the held-out lines.

Steps, one module each:

1. ``policies``, ``card``, ``menu``  the candidates and their kept fractions, chosen on calibration lines: the
                                    uniform and cell-line-level shrinks, the removals of the model's context-level and
                                    perturbation-level corrections and of the transferred response, the single-model
                                    menu (toward the plate-aware baseline where the screen has replicate plates), and
                                    the recommendation condition;
2. ``comparison``                   the held-out test: per held-out line, a sign flip over lines with the seed rule,
                                    and the paired line bootstrap beside it;
3. ``attribution``, ``origins``     where the error sits and which origin it points to, with the references that add
                                    like the error (cell sampling of the targets, the between-plate discrepancy, the
                                    input bag's sampling);
4. ``stages``                       the stages over one dataset's case (selection, calibration, attribution);
5. ``il_cells``                     recovery and cost of a cell that information localization hands on, against continued
                                    training and simple calibration.

``scoring`` holds the natural task's risk on the clipped scale and its additive parts, ``grids`` reads prediction
grids, and ``references`` computes the datasets' references from cells. A dataset enters through its *case*
(``stages``): its units, its models' prediction grids, its input bags and its documented origin rows. The modules take
loaded objects and the dataset's facts (for example the names of its replicate plates) as arguments; importing them
has no side effects and loads no torch.

Code names and the paper's terms (saved files keep the code names):

==============================================================  ==================================================
Code                                                            Paper
==============================================================  ==================================================
parent                                                          original model
nested parent, nested chain                                     nested model
calibration units or lines                                      calibration lines
outer units or lines                                            held-out lines
anchor ``a``                                                    anchor ($S+r_p$)
no change ``b``, ``nochange``                                   no-change prediction
plate-aware anchor ``A``, ``plate_baseline``                    plate-aware baseline
correction ``d = p - a``; its context part ``P d``              correction; context-level correction
common / differential part                                      common / contrast component
M5, M7, N                                                       context-level correction, perturbation-level
                                                                correction, transferred response
amount ``lam``, ``(lam_c, lam_d)``, ``kappa``                   kept fraction $\\lambda$, $(\\lambda_1, \\lambda_2)$,
                                                                $\\lambda_r$
``cheapest_fix``                                                the context-level correction scaled (cell-line-level
                                                                shrink)
``global_parent_shrink``                                        uniform shrink toward the anchor
repair, IL cell, ``size_fix``                                   restoring change; its kept fraction
continuation                                                    continued training
cosine, best scale below one                                    alignment; over-sized
model ``summary``, ``deepsets``, ``attention_model``            summary-statistics model, DeepSets model, attention
                                                                model
model ``state``                                                 State
module ``card``, files ``CARD_*``                                attribution and calibration
``card.branch``, recommendation                                 recommend or refuse
verification, "verified"                                        held-out check
``il_cells`` Q1, Q2                                             recovery and cost
``il_cells`` Q3                                                 attribution of the change
``il_cells`` Q4                                                 continued training and calibration
==============================================================  ==================================================
"""
__all__ = ["attribution", "card", "comparison", "grids", "il_cells", "menu", "origins", "policies", "references",
           "scoring", "stages"]
