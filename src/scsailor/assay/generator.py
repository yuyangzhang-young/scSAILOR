"""The known answer: a real response field, moved by the planted state, applied to real control cells.

For perturbation ``p`` and planted state ``z`` (25 signs) the shift of every gene is

    shift(p, z) = r[p] + sum_a beta[a] * z[a] * (code[p] if bit a is perturbation-dependent else 1) * direction[a % K]

where ``K = min(5, rank)`` is the number of directions kept (five in every
instance, so bit a uses direction ``a % 5``), and the known answer on one background
is the mean over all its control cells of ``max(cell + shift, 0)``, computed exactly
(``Background.clipped_mean``). Each bit's weight in beta and its perturbation dependence
come from its role (``design.ROLES``).

``calibrate`` derives the generator from real training responses: ``r[p]`` is
the weighted mean response of perturbation p, the directions are the (up to five) leading SVD
directions of the perturbation-centred responses on the scored genes, ``code`` is the
perturbation index standardized under the perturbation weights, and beta is
scaled so that sqrt(sum_a beta[a]^2) is half the real response scale sigma.

The evaluation support: each evaluation background has four random base states, each with its 25
single-bit flips (``flip_set``, ``base_states``), and the known answers of all of them for every
perturbation (``truth_table``).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from .constants import BASE_STATE_KEY, N_BITS, N_CELLS, N_GENES, N_SCORED, SCORED
from .design import ROLES

RHO = np.tile(np.array([weight for _, weight, _ in ROLES]), 5)          # response weight of each bit (role a % 5)
PERTURBATION_DEPENDENT = [a for a in range(N_BITS) if ROLES[a % 5][2]]  # bits scaled by the perturbation code
GAMMA = .5            # planted response scale, in units of sigma
# Sum of RHO**2, written as the literal the saved generators were calibrated with: the same sum computed in
# floating point may differ in the last bit (0.2**2 is inexact), which would change beta.
RHO_ENERGY = 10.4

__all__ = ["Parameters", "Generator", "Background", "perturbation_weights", "response_field", "svd_directions",
           "calibrate", "known_answer", "flip_set", "base_states", "bag_indices", "truth_table"]


@dataclass(frozen=True)
class Parameters:
    """Generator variants. alpha: the variance bits set the two genes' spreads to .5 + alpha*b and
    .5 - alpha*b; I_mean: write the mean bits into the input; T_mean: let the mean bits move the answer
    (the target). The field names are saved keys of ``INSTANCE.json``, checkpoints and predictions."""
    alpha: float = .125
    I_mean: bool = True
    T_mean: bool = True

    def __post_init__(self):
        if not 0 < self.alpha < .5:
            raise ValueError("alpha must lie in (0, 0.5) so that both planted variances stay positive")

    def record(self):
        return asdict(self)


def perturbation_weights(pcode, weight, count):
    """Training weight of each perturbation (pi)."""
    return np.bincount(pcode, weights=weight, minlength=count)


def response_field(target, parent, pcode, weight, context_means, count):
    """r[p]: weighted mean, over float64 training profiles of perturbation p, of profile minus its background's control mean."""
    residual = target - context_means[parent]
    return np.stack([np.average(residual[pcode == p], axis=0, weights=weight[pcode == p]) for p in range(count)])


def svd_directions(r, pi):
    """Up to five leading directions of the pi-weighted, perturbation-centred responses on the scored genes.

    Each direction has unit mean square on the scored genes (zero on genes 0-49) and its largest
    entry positive. Singular values closer than a numerical tolerance form a tie cluster; a cluster's
    basis is then rebuilt in gene order (Gram-Schmidt of the projections of unit gene vectors), so the
    result does not depend on the SVD routine's arbitrary rotation within the cluster.
    """
    a = (r[:, SCORED] - np.sum(pi[:, None] * r[:, SCORED], axis=0)) * np.sqrt(pi[:, None] / N_SCORED)
    _, singular, vt = np.linalg.svd(a, full_matrices=False)
    rank = int(np.count_nonzero(singular > singular[0] * 1e-10))
    if rank == 0:
        raise ValueError("The centred response field has rank zero")
    eps = np.finfo(np.float64).eps
    tie_tol = 64 * eps * max(a.shape) * singular[0]
    residual_tol = 64 * eps * N_SCORED
    bases, clusters = [], []
    start = 0
    while start < rank:  # resolve each whole tie cluster, also one that crosses direction five
        end = start + 1
        while end < len(singular) and singular[start] - singular[end] <= tie_tol:
            end += 1
        block = vt[start:end]
        if end - start == 1:
            basis = block.copy()
        else:
            selected = []
            for gene in range(N_SCORED):
                residual = block.T @ block[:, gene]
                for _ in range(2):
                    for b in selected:
                        residual -= b * np.dot(b, residual)
                norm = np.linalg.norm(residual)
                if norm > residual_tol:
                    selected.append(residual / norm)
                if len(selected) == len(block):
                    break
            if len(selected) != len(block):
                raise ValueError("Could not build a canonical basis for a tie cluster")
            basis = np.asarray(selected)
        for direction in basis:
            if direction[np.argmax(abs(direction))] < 0:
                direction *= -1
        bases.extend(basis)
        clusters.append([start, end])
        start = end
    directions = np.zeros((min(5, rank), N_GENES), dtype=np.float64)
    directions[:, SCORED] = np.asarray(bases)[:len(directions)] * np.sqrt(N_SCORED)
    return directions, {"singular_values": singular.tolist(), "gaps": np.diff(singular).tolist(),
                        "numerical_rank": rank, "clusters": clusters, "tie_tolerance": float(tie_tol),
                        "tie_branch_used": any(b - a > 1 for a, b in clusters)}


def calibrate(r, pi, context_means):
    """Generator fields from the response field ``r`` (perturbations x genes), perturbation weights ``pi`` and the
    training backgrounds' control means; also returns the SVD spectrum and the computed sigma."""
    sigma = float(np.sqrt(np.sum(pi[:, None] * r[:, SCORED] ** 2) / N_SCORED))
    directions, spectrum = svd_directions(r, pi)
    code = np.arange(len(r), dtype=np.float64)
    code -= pi @ code
    code /= np.sqrt(pi @ (code ** 2))
    fields = {"r_gen": r, "pi": pi, "directions": directions, "code": code, "sigma": sigma,
              "beta": GAMMA * sigma * RHO / np.sqrt(RHO_ENERGY), "c_train": context_means.mean(axis=0)}
    return fields, spectrum | {"sigma_computed": sigma}


class Generator:
    """A calibrated generator (``generator.npz``) under one set of ``Parameters``."""

    def __init__(self, path, params=Parameters()):
        with np.load(path) as z:
            self.r = z["r_gen"].copy()
            self.pi = z["pi"].copy()
            self.directions = z["directions"].copy()
            self.code = z["code"].copy()
            self.beta = z["beta"].copy()
            self.c_train = z["c_train"].copy()
            self.intercept = z["intercept"].copy()
            self.p_ids = z["p_ids"].astype(str)
        self.params = params
        self.beta_effective = self.beta.copy()
        if not params.T_mean:
            self.beta_effective[:5] = 0
        self.bit_direction = np.arange(N_BITS) % len(self.directions)

    def shift(self, p, states):
        """Gene shifts for perturbations ``p`` (n) under planted states (n x 25)."""
        p = np.asarray(p, dtype=np.int64).reshape(-1)
        states = np.asarray(states, dtype=np.float64).reshape(-1, N_BITS)
        coefficients = states * self.beta_effective
        coefficients[:, PERTURBATION_DEPENDENT] *= self.code[p, None]
        shift = self.r[p].copy()
        for a in range(N_BITS):
            shift += coefficients[:, a, None] * self.directions[self.bit_direction[a]]
        return shift


class Background:
    """One background's control cells, sorted per gene, so that clipped means of shifted cells are exact."""

    def __init__(self, cells):
        self.sorted = np.sort(np.asarray(cells, dtype=np.float64), axis=0).T.copy()   # genes x cells
        self.count = self.sorted.shape[1]
        self.prefix = np.concatenate((np.zeros((len(self.sorted), 1)), self.sorted.cumsum(axis=1)), axis=1)

    def clipped_mean(self, shift):
        """Mean over cells of max(cell + shift, 0), for each row of ``shift`` (n x genes)."""
        shift = np.asarray(shift, dtype=np.float64).reshape(-1, len(self.sorted))
        # Binary search, per row and gene, for the number of cells clipped to zero (cell + shift <= 0).
        low = np.zeros(shift.shape, dtype=np.int32)
        high = np.full(shift.shape, self.count, dtype=np.int32)
        genes = np.arange(len(self.sorted))[None, :]
        while np.any(low < high):
            mid = (low + high) // 2
            values = self.sorted[genes, np.minimum(mid, self.count - 1)]
            below = (mid < self.count) & (values <= -shift)
            low = np.where(below, mid + 1, low)
            high = np.where(below, high, mid)
        # The unclipped cells contribute their sum plus one shift each.
        return (self.prefix[:, -1] - self.prefix[genes, low] + (self.count - low) * shift) / self.count


def known_answer(generator, background, p, states):
    """Known answers (n x genes) on one background for perturbations ``p`` under planted ``states``."""
    return background.clipped_mean(generator.shift(p, states))


def flip_set(base):
    """A state and its 25 single-bit flips (26 x 25, int8): row 0 is ``base``, row a + 1 flips bit a."""
    states = np.tile(base, (N_BITS + 1, 1)).astype(np.int8)
    for a in range(N_BITS):
        states[a + 1, a] *= -1
    return states


def base_states(root, context_index, base):
    """Evaluation base state ``base`` of one background, drawn at random, with its 25 flips (26 x 25, int8)."""
    state = np.random.default_rng(np.random.SeedSequence([root, context_index, base, BASE_STATE_KEY])).choice(
        [-1, 1], size=N_BITS)
    return flip_set(state)


def bag_indices(pool_size, bag_key):
    """A bag: 256 cells drawn without replacement from a background's pool, keyed by ``bag_key`` (the fixed
    evaluation bags, and the probe bags)."""
    return np.random.default_rng(np.random.SeedSequence(bag_key)).permutation(pool_size)[:N_CELLS]


def truth_table(background, generator, root, context_index, bases=4):
    """Base states with their flips (bases x 26 x 25) and known answers for every perturbation
    (bases x 26 x perturbations x genes)."""
    p = np.arange(len(generator.p_ids))
    states = np.stack([base_states(root, context_index, base) for base in range(bases)])
    values = np.empty((bases, N_BITS + 1, len(p), len(background.sorted)))
    for base in range(bases):
        for config in range(N_BITS + 1):
            values[base, config] = known_answer(generator, background, p, np.tile(states[base, config], (len(p), 1)))
    return states, values
