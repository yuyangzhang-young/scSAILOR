"""The attainable reference of the synthetic-answer assay (Supplementary Table 9), computed on a CPU from an instance.

The decoded-cue reference reads each planted state from the model's actual input, with a fixed sign rule
(``scsailor.assay.carrier.decode_bit``) at the first level that carries its cue and with the replay's technical
draw 0, and predicts the state's response as the training-average half-difference of that state and perturbation:
the signed half-difference of the synthetic target between the state's two values, averaged over the training
contexts' full control pools and four random base states each (keyed by ``TRAINING_ROOT``). Its training side thus
sees paired state responses, which the models' single-state labels do not give. The predicted half-differences are
written into profiles (``embed``) and scored like a model's (``scsailor.assay.score``) on the held-out contexts.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from scsailor.assay.carrier import carrier, decode_bit
from scsailor.assay.constants import N_BITS, N_GENES, REPLAY_DRAW_KEY
from scsailor.assay.design import STATISTICS
from scsailor.assay.generator import Background, truth_table
from scsailor.assay.instance import HELDOUT
from scsailor.assay.score import flip_pairs, pool_contexts

from .scores import describe

__all__ = ["TRAINING_ROOT", "BASES", "training_average", "decode_states", "embed", "decoded_cue_reference"]

TRAINING_ROOT = 61120     # root of the keys of the training contexts' base states
BASES = 4                 # base states per context, as in the evaluation support


def training_average(instance):
    """Each planted state's signed half-difference of the synthetic target, per perturbation (states x
    perturbations x genes), averaged over the training contexts' full pools and ``BASES`` base states each."""
    total = np.zeros((N_BITS, len(instance.generator.p_ids), N_GENES))
    for q, context in enumerate(instance.train_contexts):
        background = Background(instance.pool("train", context)[0])
        states, values = truth_table(background, instance.generator, TRAINING_ROOT, q, BASES)
        for base in range(BASES):
            for a in range(N_BITS):
                total[a] += states[base, 0, a] * (values[base, 0] - values[base, a + 1]) / 2
    return total / (BASES * len(instance.train_contexts))


def decode_states(instance, bag):
    """The states read from each evaluation input of one bag (``EvaluationBag``; bases x 26 x 25): each cue's five
    states at the cue's first level, from the input built with the replay's technical draw 0."""
    read = np.zeros_like(bag.states)
    for base in range(len(bag.states)):
        key = [bag.root, bag.context_index, base, 0, REPLAY_DRAW_KEY]
        for config in range(N_BITS + 1):
            for cue, (_, _, level) in enumerate(STATISTICS):
                rows, row_depth = carrier(bag.x, bag.depth, bag.states[base, config], level,
                                          np.random.default_rng(np.random.SeedSequence(key)), instance.params)
                for a in range(5 * cue, 5 * cue + 5):
                    read[base, config, a] = decode_bit(rows, row_depth, a)
    return read


def embed(differences, states):
    """Profiles (bases x 26 x perturbations x genes) whose paired half-differences equal ``differences`` (states x
    perturbations x genes) under ``states``; a common positive offset keeps clipping inactive."""
    base = np.einsum("ha,apg->hpg", states[:, 0], differences)
    prediction = np.repeat(base[:, None], N_BITS + 1, axis=1)
    for a in range(N_BITS):
        prediction[:, a + 1] += (states[:, a + 1, a] - states[:, 0, a])[:, None, None] * differences[a]
    offset = np.maximum(0, -prediction.min(axis=(1, 3))) + 1
    return np.maximum(prediction + offset[:, None, :, None], 0)


def decoded_cue_reference(instance):
    """Held-out recovery of the decoded-cue reference per planted state, pooled over the held-out contexts, with the
    number of decoded inputs and of wrongly decoded states."""
    average = training_average(instance)
    tables, decoded, wrong = [], 0, np.zeros(N_BITS, dtype=int)
    for bag in instance.evaluation_bags(HELDOUT):
        read = decode_states(instance, bag)
        decoded += bag.states.shape[0] * bag.states.shape[1]
        wrong += (read != bag.states).sum(axis=(0, 1))
        tables.append(flip_pairs(embed(average, read), instance.truth(HELDOUT, bag.context)[1], bag.states)[0])
    return pd.DataFrame([{**describe(row["module"]), "recovery": row["Q"], "decoded": decoded,
                          "decoded_wrong": int(wrong[row["module"]])} for row in pool_contexts(tables)])
