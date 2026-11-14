"""The synthetic-answer assay in the paper: its settings and its binding to the built models (``study``), and the
tables of Results 3, Fig. 3 and Supplementary Note S6, computed on a CPU from the saved per-fit scores and the
deposited instances (``tables``, README.md).

``scores`` reads the per-fit scores; ``recovery`` makes the per-fit table, the seed and fold means, the isolation
checks of the saved predictions and the positive control; ``whole_profile`` compares recovery with the whole-profile
scores; ``references`` computes the attainable reference and ``checks`` the carrier check from an instance;
``fold_selection`` reads which genetic-screen folds the assay uses; ``constructed`` scores constructed predictions
with known errors (Fig. 3d); ``tables`` writes them all.

Code names and the paper's terms. The tables print the paper's names; the saved files keep the names in the third
column, and the columns of ``GENETIC_FOLD_SELECTION.csv`` keep the saved word ``anchor`` for the baseline.

================================  ================================================  ==================================
Code (tables)                     Paper                                             Saved
================================  ================================================  ==================================
``planted_state`` 0-24            planted (binary) state, a test; in                ``module``, ``module_id``
                                  ``scsailor.assay`` a bit
``cue``                           cue: shift, spread, skew, depth trend,            ``edge``: mean, variance,
                                  covariation; in ``scsailor.assay`` a statistic    skewness, depth_coupling,
                                                                                    gene_joint
``response``                      shared, selective, weak shared, weak selective,   ``module_type``: main_common,
                                  null response; in ``scsailor.assay`` a role       main_conditional, weak_common,
                                                                                    weak_conditional, null
``level``                         Mean, +Var, +Dist, +DepCp, Cells                  L1, L2, L3, L3d, L4
context                           context; in ``scsailor.assay`` a background       -
model input                       model input; in ``scsailor.assay`` a carrier      -
``cue_in_input``                  the input at that level carries the cue           ``design.visible``
``true_energy``                   true squared response size V                      E
``squared_error``                 squared prediction error M                        M
``predicted_energy``              predicted squared size V-hat                      A
``recovery``                      recovery Q = 1 - M/V                              Q (computed here from E and M)
``projection_slope``              projection slope kappa = K/(2V)                   computed here from the saved K,
                                                                                    not read from ``kappa``
``fit_status``                    kept: a fit drawn in Fig. 3; excluded: training   -
                                  collapse: an attention-model fit collapsed in
                                  training (Methods, Fit exclusions)
``learning_rate``                 standard: the model's peak learning rate; 0.3x    run suffix ``_lr03``
                                  standard: 0.3 times it
``model`` attention model         attention model                                   ``attention_model``; ``state`` in
                                                                                    the pooled and whole-profile
                                                                                    tables
``model`` State                   State                                             ``state`` in the score files'
                                                                                    folder names; ``state_pub`` in the
                                                                                    tables
``model`` the other models        DeepSets model, summary-statistics model, CPA     ``deepsets``, ``summary``, ``cpa``
``model`` PrePR-CT                PrePR-CT                                          ``prepr``; ``PrePR-CT`` in the
                                                                                    pooled table
held-out                          held-out contexts                                 ``F1`` in every fold; ``scope``
                                                                                    ``F1``, ``support`` ``heldout``
``side``                          held-out or training contexts of the evaluation   ``F1``, ``seen``
``n_seeds``, ``n_folds``          seeds and folds averaged                          -
``largest_size_without_cue``      largest predicted squared size of a non-null      -
                                  test at a level without its cue
``largest_ratio_without_cue``     the same, divided by the true squared size        -
``largest_null_ratio``            a visible null test's predicted squared size      ``capability.leakage_ratio``
                                  over the mean true energy of the same cue's
                                  shared and selective responses: the largest
``median_null_ratio``             the same, the median                              ``capability.leakage_ratio``
``total_mse``                     total MSE                                         ``total.mse``
``pearson_delta``                 PearsonΔ, training-perturbation reference         ``systema.pearson_ref_train``
``interaction_alignment``         interaction alignment (r_gamma)                   ``mz.r_gamma``
``weighted_r2``                   adapted weighted R² (R²w)                         ``dbd.r2w_delta``
``centroid_accuracy``             centroid accuracy                                 ``systema.centroid_accuracy``
``shared_recovery``               recovery summed over the five shared tests        -
``selective_recovery``            recovery summed over the five selective tests     -
decoded-cue reference             the attainable reference that decoded the input   -
                                  state
``decoded``, ``decoded_wrong``    inputs decoded per state; states decoded wrongly  -
``labels`` exact, sampled         exact-label and sampled-label arms of the         -
                                  positive control
``case``                          Exact, Offset, Swapped, Halved, Absent,           -
                                  Reversed: the constructed predictions of Fig. 3d
``assessment``                    Whole profile, Paired (the half-differences)      -
R², R²row, R²w, R²w,full          global R², row-mean R², adapted weighted R²,      -
                                  full-response adapted weighted R²
``squared error / V``             squared error in units of the true squared        -
                                  response size V
``anchor_error``                  the baseline's contrast error (baseline:          ``heldout_anchor_raw_difference_er
                                  input-set mean plus average training response)    ror_mse``
``no_change_error``               no change's contrast error                        ``heldout_input_only_raw_differenc
                                                                                    e_error_mse``
``noise_floor``                   the contrast error of the measured responses'     ``difference_error_mse``
                                  sampling noise
``anchor_share_percent``          fraction (%) of above-noise                       -
                                  perturbation-specific signal explained by the
                                  average training response (S6.3)
================================  ================================================  ==================================
"""
