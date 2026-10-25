"""Information localization: where, inside a trained model, learning of a missed planted response is obstructed.

A model that misses a planted response of the synthetic-answer assay is retrained in part: a cut divides it into a
front and the part behind it, and the part behind the cut is trained on the planted responses from several
starting points. Which starting point learns the response says where learning is obstructed. Planted faults with a
known site set the reading rules. The change that restores the response is then applied unchanged to the model
trained on real responses, for origin resolution to judge.

Steps, one module each:

1. ``starts``         one starting point at one cut, trained on the planted responses (map, rate pilot, time axis);
2. ``readings``       the reading rules over finished runs: per-run outcomes, contrasts between starting points,
                      the change handed on (or the fixed intervention), planted-fault verdicts;
3. ``presence``       how strongly a planted state still moves each cut, without training;
4. ``training_time``  recovery along the original model's own training, read on its saved checkpoints;
5. ``real_task``      the handed-on change applied to a model trained on real responses.

``task`` reads a synthetic-answer instance as the steps draw from it, ``recovery`` scores recovery Q on flip-pair
panels, ``draws`` makes the panels' draws and the training batches, ``runtime`` sets threads and numerical precision
(the only place that does), ``constants`` holds the shared constants, and ``contract`` states what localization
adds to the assay's model contract (``scsailor.assay.contract``). Every module takes loaded objects and the study's
settings (seeds, roots, panels, planted faults) as arguments; importing a module has no side effects, and torch is
imported only inside the functions that run a model.

Code names and the paper's terms (saved files keep the code names):

=========================================================  ==========================================================
Code                                                       Paper
=========================================================  ==========================================================
cut                                                        cut
arm, start                                                 starting point
arm ``a``; ``a`` at the cut ``input``                      fine-tuning; whole network
arm ``f``, ``e``, ``d``                                    back reset, front reset, embedding reset
arm ``fresh``                                              from scratch
arms ``e_mean``, ``e_var``                                 mean-projection reset, variance-projection reset
arms ``interaction``, ``common``, ``interaction_e_mean``   one branch trained alone (summary-statistics model)
readings                                                   reading rules
known answer, planted-fault cases                          synthetic target; planted faults
carried cell, qualified repair                             restoring change ("the change")
diagnostic intervention                                    fixed intervention
bits 0, 1, 5, 6                                            shared shift, selective shift, shared spread, selective
                                                           spread tests
Q                                                          recovery
parent                                                     original model
half ``synthetic`` / ``natural``                           model trained on the synthetic task / on real responses
time axis                                                  training longer
model ``summary``, ``deepsets``, ``state``                 summary-statistics model, DeepSets model, attention model
model ``official_state``, ``cpa``, ``prepr``               State, CPA, PrePR-CT
drug embedding (``backbone.pert``)                         perturbation embedding
=========================================================  ==========================================================
"""
__all__ = ["constants", "contract", "draws", "presence", "readings", "real_task", "recovery", "runtime", "starts",
           "task", "training_time"]
