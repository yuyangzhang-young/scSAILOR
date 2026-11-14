"""The paper's Results 7 on Tahoe-100M (folds 1-3), computed on a CPU from the saved predictions: memorization of the
training lines' responses and control levels, and the readers of a held-out context's control cells that replace a
model's context-level correction.

``data`` reads a fold's units and the saved predictions; ``programs`` holds the growth programs on each fold's panel;
``readers`` fits the readers and builds the arms; ``calibration`` holds origin resolution's +Var cases and their
calibration; ``arms``, ``metrics``, ``selection``, ``ranking``, ``storage`` and ``growth_genes`` compute the tables;
``tables`` writes them all (README.md).

Code names and the paper's terms (the tables keep the code names, except their ``model`` column, which prints the
names of ``data.MODELS``):

=========================================================  ==========================================================
Code                                                       Paper
=========================================================  ==========================================================
context ('line|plate'); unit                               a cell line on one plate; one line, plate and drug
input set, its mean ``b``                                  input set (the 256 control cells), control mean S
control pool, pool mean ``C``                              full control pool, pool mean
drug average ``r``, ``r_double``                           average training response r_p
anchor ``b + r``, ``A0``                                   anchor (S + r_p)
``kappa``                                                  lambda_r
``b + kappa r_plate``, ``Th``, ``baseline``                design-aware baseline
offset (``offsets``, ``training_offsets``)                 source context response: the context-level correction
                                                           needed beyond the pool mean and average training response
correction, ``proj_ctx(P - A0)``                           context-level correction
gene-wise readers (``GeneReaders``)                        readers of each gene's own control mean
growth reader (``KernelReader``), ``GUT``                  growth reader: three program means and gut lineage
arms 'original', 'removal', 'constant', 'readers'          original, removal, the training lines' mean (constant
                                                           control, Const.), Readers
arms 'calibrated', 'calibrated + readers'                  calibrated (Calib.), calibration plus readers (Combined)
contrast 'A - B'; 'calibrated + readers - calibrated'      A vs B; Combined vs calibrated
``units`` 'all held-out', 'calibration-supported'          all held-out units; matched +Var support
``n_units``                                                the number of calibration-supported units
level 'Mean', '+Var'; file label ``L1``                    Mean inputs, +Var inputs; Mean inputs
``total_*``                                                total MSE
``common_*``, ``contrast_*``                               common component, contrast component
``*_points``, ``*_pct_of_design_aware``                    % of the design-aware baseline's total MSE
``*_pct_of_original``, ``*_pct_of_calibrated``             % of the original's, of the calibrated prediction's total
                                                           MSE
metric ``mse``                                             squared error (total MSE)
``weighted_r2``, ``weighted_r2_contrast``                  adapted weighted R^2, adapted weighted contrast R^2
``pearson_delta``, ``spearman_delta``                      Pearson Delta, Spearman Delta
``direction_top50``, ``cosine_pds``                        top-50 direction, PDS
``systema_pearson_delta``                                  Systema Pearson Delta
``systema_pearson_delta_observed``                         Systema Pearson Delta (observed)
growth programs (``programs.GROWTH``), growth genes        Hallmark E2F targets, G2M checkpoint and mitotic spindle;
                                                           their 177-182 panel genes
program response, growth response                          growth-program response
pairs (``selection``), reliability, level relation         training-selected drug-program pairs, between-plate
                                                           response correlation, mean within-plate
                                                           control-level-response correlation
``lines_spearman``; predictors 'original', 'readers'       line ranking; original, Readers
``PLATE_REFERENCE``, ``data``                              Plate against plate
``data_low``, ``data_high``                                its 95% line-bootstrap interval
pull, ``pull_slope`` (``beta``), ``level_gap``             pull: the plate-centred slope on the training-pool-minus-
                                                           input growth gap
``without_pull``, ``ranking_without_pull``                 Without pull
``growth_biology_median``, ``pair_median``                 pair-median (drug-program) growth relation
growth-gene correction, linear reader                      the separate growth-gene correction, per-drug linear
                                                           readers of all gene means
read-outs ``READOUTS`` (``Readouts``)                      the seven routine read-outs
``systema_pearson_delta_observed_unweighted``              Systema Pearson Delta (observed), read without weights
``pearson_delta_unweighted``                               Pearson Delta, read without weights
readable part ``R``, unreadable residual ``E``             readable and unreadable response
sampling difference ``s``                                  pool-minus-control sampling difference
``slope_unreadable``                                       storage (the coefficient on the unreadable response)
``train_program_slope_sampling``                           control-level memory (the sampling coefficient), on the
                                                           training lines, at program level
``step``                                                   training length (updates)
seed 'seed mean'                                           mean of the seeds, each scored alone
model ``summary``, ``deepsets``, ``attention_model``       summary-statistics model, DeepSets model, attention model
model ``state``, ``cpa``, ``prepr``                        State, CPA, PrePR-CT
=========================================================  ==========================================================
"""
