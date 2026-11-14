"""The paper's Results 7 on Tahoe-100M (folds 1-3), Parse PBMC and the genetic screens, computed on a CPU from the
saved predictions: memorization of the training contexts' responses and control levels, and the readers of a held-out
context's control cells that replace a model's context-level correction.

``data`` reads a fold's units and the saved predictions; ``programs`` holds the growth programs on each fold's panel;
``readers`` fits the readers and builds the arms; ``calibration`` holds origin resolution's +Var cases and their
calibration; ``arms``, ``metrics``, ``selection``, ``ranking``, ``storage`` and ``growth_genes`` compute the Tahoe-100M
tables; ``parse`` and ``parse_interferon`` those of Parse PBMC (readers and error; interferon readings) and
``genetic`` those of the genetic screens; ``tables`` writes them all (README.md).

Code names and the paper's terms (the tables keep the code names, except their ``model`` and ``screen`` columns, which
print the names of ``data.MODELS`` and ``genetic.NAMES``):

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
Parse PBMC (``parse``, ``parse_interferon``)
group 'G1', 'G2', 'G3'                                     donor groups 1-3
context ('donor|cell type'), cell type                     a donor's cell type: CD14 monocytes, CD4 memory and CD4
                                                           naive T cells
readers (``group``'s ``readers``)                          readers of each gene's control mean, cell-type centred,
                                                           leave-one-donor-out
arms 'calibrated', 'calibrated + readers'                  calibrated; readers after calibration
amounts (``amounts``; ``GRID``, ``TIES_TO_SMALLER``,       calibration's context and perturbation scales, chosen on
``ONE_PAIR_FROM_SEED_MEAN``)                               the calibration donors from the nested parents (a rule
                                                           unlike menu.pick_two's: parse's docstring)
``total_pct_of_original``                                  total MSE in % of the original model's
scope 'G1'-'G3', 'pooled'; ``donors``                      one donor group; pooled over all 12 donors
comparison 'A - B'; ``gain``                               A vs B; the change, signed so that positive favours A
                                                           (squared error in % of the original's total MSE)
``ci90_low``, ``ci90_high``                                its 90% paired-donor interval
``control_level``                                          control level relative to training donors
series 'measured'; reference wells                         measured, against control wells no model saw (H10-H12)
``value``; ``ratio``                                       interferon-alpha response; its ratio to the control level
``split``                                                  a cytokine partition (the same in every group)
``spearman`` (``INTERFERON_RANKING.csv``)                  interferon-alpha ranking over the 36 group-by-cell-type-
                                                           centred contexts
``pearson`` (``CONTROL_WELLS.csv``)                        correlation of input wells (H7-H9) and reference wells
``sd_between_donors``                                      between-donor SD of the program's control level
genetic screens (``genetic``)
screen 'Replogle-Nadig', 'Jiang et al.'; ``line``          Replogle-Nadig CRISPRi, Jiang et al.; the held-out line
anchor, 'baseline'                                         baseline: the model's anchor (input-set mean + average
                                                           training response for the built models; State and CPA
                                                           store their own)
side 'training lines', 'held-out line'; ``slope``          the slope of the context-level correction on the line
                                                           offset, on training lines and on the held-out line
``reader_r2``                                              the readers' (out-of-line) R² of the line offset
``COMPOSITE``                                              growth composite: E2F targets, G2M checkpoint, mitotic
                                                           spindle
``growth_level``                                           the line's control growth level (relative to the training
                                                           lines)
``growth_correction``, ``measured_growth_correction``      the predicted growth correction; the measured growth-
                                                           composite correction
``most_pulled``                                            the most-pulled program (growth or MYC pull: E2F targets,
                                                           G2M checkpoint, mitotic spindle, MYC targets V1 and V2)
prediction 'removal'; 'removal + readers'                  context removal; readers after context removal
prediction 'gene-wise shrink'                              the gene-wise extension: slopes restricted to [0, 1] after
                                                           context removal, knockdown-target genes unshrunk
prediction 'no change'                                     predicting no change (the input-set mean)
``UNIT_VALUES``                                            the nine metrics per unit, the target gene kept (weight
                                                           zero in the weighted R² only; unlike the shared
                                                           target_excluded_values)
comparison 'A - B'; ``change``                             A vs B; A minus B, the raw sign (as in Supplementary
                                                           Table 28; MSE in points of the baseline's total MSE)
``cases_improved``, ``median_change``                      cases improved (lower MSE, higher otherwise); median
                                                           paired change
``median_first``, ``median_second``                        the two predictions' median MSE (or other metric)
=========================================================  ==========================================================
"""
