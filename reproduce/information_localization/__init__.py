"""The paper's information-localization runs of the built models on Tahoe-100M, and the tables they give.

``tahoe`` holds the settings, the parents and the run entry; ``planted_faults`` builds the planted-fault parents;
``time_and_presence`` makes cue profiles and scores later checkpoints (training longer); ``tables`` rebuilds the
tables from the saved run records and cue profiles. The method is ``scsailor.localization``, whose term table gives the
code names of maps, cuts, starting points, halves and bits.

What the saved files and these tables add:

==============================================  =============================================================
Code                                            Paper
==============================================  =============================================================
``state`` (saved runs, scores, profiles)        attention model (``attention_model`` in the public code)
``official_state``, ``official-state``          State
bit ``4``                                       null test of the shift cue (no response)
``Q1``                                          selective-shift recovery
cases ``solved``, ``ds-solved``, ``parent``     unfaulted controls (``parent``: State, with no planted fault)
cases ``dead-interaction``, ``ds-dead-back``    planted faults: disabled back
cases ``blocked-mean-cue``, ``ds-blocked-cue``  planted faults: blocked shift cue
case ``blocked-variance-cue``                   planted fault: blocked spread cue
lockstep group                                  runs that shared one GPU
==============================================  =============================================================
"""
