"""The information ladder (Results 1-2): every natural-task fit on the five screens with its held-out scores, the
ladder's fold points and fold means, the model-free references, the intervals, the plate-aware baseline and the
adapted weighted R2, from the saved per-fit scores and the fold data under the data root.

``fits`` reads the saved scores into the per-fit table and applies the fit rules; ``ladder`` averages seeds and folds;
``references`` scores the model-free references of Parse PBMC and Jiang et al.; ``intervals`` bootstraps the held-out
lines of Tahoe-100M; ``plate`` scores the plate-aware baseline; ``weighted_r2`` holds the adapted weighted R2;
``tables`` writes them all, in this order (README.md).

Code names and the paper's terms (the tables print the paper's names of screens, models and levels):

=========================================================  ==========================================================
Code                                                       Paper
=========================================================  ==========================================================
screen 'Tahoe-100M', 'GSE306429', 'Parse PBMC',            the five screens (the key is the printed name; en dash in
'Replogle–Nadig', 'Jiang et al.'                           Replogle–Nadig)
fit                                                        one model trained on one screen, fold, seed and level, read
                                                           at its fixed training length (``training_updates``)
``fit_id``                                                 screen|fold|model code|seed|level label|training updates
fold 'F1'...; 'G1'-'G3'                                    fold; Parse PBMC's donor groups 1-3
level 'Mean', '+Var', '+Dist', '+DepCp', 'Cells';          information ladder levels; the score files' labels
file label 'L1', 'L2', 'L3', 'L3d', 'L4'
prediction unit; ``units``, ``units_left_out``             prediction unit (one context and perturbation); their count
the files' ``unit`` (``held-out-unit-sums``)               a held-out cell line (Tahoe-100M) or donor (Parse PBMC)
baseline; the files' anchor                                baseline: control mean plus the average training response;
                                                           the anchor is the baseline a model adds its correction to,
                                                           stored before clipping and scored after clipping at zero
``model_mse_*``, ``baseline_mse_*`` (total, common,        total, common and contrast MSE of the model and of the
contrast); the files' ``difference_mse``,                  baseline (contrast: the files' ``difference``, ``d``,
``difference_error_mse``, ``d``, ``base_d``                ``base_d`` names)
``relative_mse``                                           relative MSE (model MSE divided by baseline MSE)
``delta_common_points``, ``delta_contrast_points``         model-minus-baseline common and contrast MSE, % of
                                                           baseline total MSE
``common_change_from_mean``,                               the level's fold point minus the Mean level's (Cells minus
``contrast_change_from_mean``                              Mean at Cells), % of baseline total MSE
``fit_status`` 'kept', 'excluded: ...'                     fit exclusions: training total MSE at least 0.99 of the
                                                           anchor's training total MSE; held-out relative MSE above
                                                           1.4 times the median over the fits of the same screen,
                                                           fold, model and level that passed the training rule,
                                                           itself included
fold point; 'fold mean'; ``seeds``                         fold marker (mean over the kept fits of a model, level and
                                                           fold); tick or bar (mean over folds, equally weighted); the
                                                           kept fits' seeds (their number in the plate comparison)
``relative_mse_outliers_reinstated``                       fold-mean relative MSE with the outliers kept
``change_from_mean``, ``low``, ``high``                    Mean-to-Cells change and its 90% paired bootstrap interval
plate-aware baseline                                       plate-aware baseline: control mean plus the training lines'
                                                           average response on the same plate
``common_reduction``, ``contrast_reduction``               decrease of common and contrast MSE from the baseline, % of
                                                           its total MSE
``lowest_*``, ``lowest_level_*``                           the lowest of the five ladder levels, with that level
``above_plate_aware``                                      model R2w above the plate-aware baseline's
no change (the files' ``input_only``); stratum mean;       model-free references: control mean; plus the average
design-aware baseline                                      training response within cell type or stimulus; that
                                                           response shrunk toward no change
``anchor_raw`` (``no_change.csv``)                         the baseline's stored prediction, scored after clipping
``lowest_model``, ``lowest_model_name``                    the lowest model relative MSE of a fold, and its model
``best_scale``                                             best scale (kappa), fitted on calibration lines
``strongest``; ``minus_strongest``                         the strongest model-free reference; model MSE minus its
                                                           MSE, % of baseline total MSE
``weighted_r2``, ``weighted_r2_contrast``; version         adapted weighted R2 (R2w), total and contrast versions
'Total', 'Contrast'
``difference`` (``WEIGHTED_R2.csv``)                       Delta R2w: model minus baseline adapted weighted R2
evaluation 'all twelve donors', 'ten donors'               Parse PBMC's primary evaluation; the sensitivity analysis
                                                           without Donor2 and Donor3
model ``summary``, ``deepsets``, ``attention_model``       summary-statistics model, DeepSets model, attention model
model ``state``, ``cpa``, ``prepr``                        State, CPA, PrePR-CT
=========================================================  ==========================================================
"""
