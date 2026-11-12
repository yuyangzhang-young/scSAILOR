"""The light version of information localization: a fixed menu of cuts, starting points and learning rates for each
model, one seed for the parts, and a second seed for the best candidates only.

Every light run keeps the full version's retraining budget. The light version reads the same run folders and uses
the full version's recovery level, margins and choice of the change (``scsailor.localization.readings``), all on the
validation panel; the exit for a target the original model already recovers is also read there. It trains nothing
itself: it writes the run lists for the runner and reads the finished runs.

    runs = maps.read_map(folder, dataset)            # finished runs of one map folder; validation and seen panels
    cells = rules.cells(runs)                        # per run and target: validation Q and margins
    cases = selection.light_cases(runs, cells)       # the light reading of every case
    plan.check_rows(cases, runs)                     # second-seed runs still needed
    selection.handoff(cases)                         # the changes handed on per model and fold, merged
    plan.map_rows(model, fold); plan.assay_fits(model, fold); plan.origin_needs(dataset, model, fold, cells, runs)
    compare.cases(runs, cells)                       # light against full, where a full map exists

``rules`` holds the full version's reading rules on tables of runs, ``menus`` the menus of the six models and the rule
for a new architecture, ``selection`` the light selection and the descriptive reading, ``maps`` the reader of a map
folder, ``plan`` the run lists, ``assay`` the light synthetic-answer assay panel and ``compare`` the light version
against the full one. Importing a module has no side effects.

Code names and the paper's terms, beyond those of ``scsailor.localization`` (models, halves, cuts, starting points):

==========================================  =============================================================
Code                                        Paper
==========================================  =============================================================
target ``'1'`` / ``'0'`` / ``'5'``          selective shift / shared shift / shared spread (the labels
                                            written in ``target_name``: selective mean, common mean,
                                            common variance)
case                                        model x fold x original model (``half``) x target
``LEVEL``                                   recovery level
primary seed; check                         the seed of the parts; the second seed of the best candidates
==========================================  =============================================================

The study's saved runs name the attention model ``state`` and State ``official_state``; rename them to
``attention_model`` and ``state`` before reading them with these modules.
"""
__all__ = ["assay", "compare", "maps", "menus", "plan", "rules", "selection"]
