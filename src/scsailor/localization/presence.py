"""Presence profile: how strongly a planted state still moves each cut of a network.

The profile is read behind every cut before any map run: a planted cue that no longer moves a cut cannot be
learned behind it. For one model, one support and one planted bit:

* **Draws** (``draws``): a flip pair is two inputs with the same cells, drug and carrier draw that differ only in
  the bit; two resample bags share the minus input's state, drug and background but have other cells and carrier
  draws. The resample bags are disjoint from the flip pair, so effect and noise are estimated independently. A
  context is one background (cell line and plate); every pair draws its own drug.
* **Bag embeddings per cut** (``Embedder``): a set-level output as it is; a per-cell output (one vector per cell)
  as its cell mean, its per-feature second moment, and a kernel mean embedding (random Fourier features of the
  standardized cells, Gaussian kernel, median-distance bandwidth). Standardization and bandwidth are calibrated
  without labels on inputs pooled over both states and all bits; coordinates that do not vary are kept. Every
  statistic below describes these embedded bag laws, not every aspect of the cell-set laws.
* **Numerical indistinguishability** (not a certificate of removal): per coordinate, the largest
  |embedding(plus) - embedding(minus)| over the tested pairs against that coordinate's own round-off floor,
  measured on the same bags with their cells permuted (every network here is permutation-invariant). A cut is
  indistinguishable on the tested pairs when no coordinate exceeds its floor.
* **Continuous statistics** (``statistics``), each with 2,000 line resamples (both plates of a line move
  together), pooled over contexts and, as ``*_ctx``, conditional on the context (only pairs of the same
  background are crossed):
  - ``energy``: the paired-index U-statistic of the energy distance between the flipped and unflipped bag laws
    (never comparing a bag with its own flipped copy), and ``energy_coefficient`` = energy / (2 E|X - Y|);
    ``energy_null`` is the same statistic between the two resample sets (same state);
  - ``pair_energy``: the mean squared per-pair displacement |delta|^2 over the noise energy. A flip pair shares
    its cells, so delta is exact for that bag and free of the fluctuation between two independent bags (it still
    varies with the sampled cells); it needs no consistent direction across contexts or drugs;
  - ``euclid``: the cross-pair U-statistic of |E delta|^2 (unbiased for independent pairs) over the mean squared
    resample difference (= twice the trace of one bag's covariance); the ratio itself is not unbiased;
  - ``whitened``: the cross-pair statistic after whitening by the bag-noise covariance, on the coordinates whose
    resample differences exceed round-off (the positive-noise part). The leading principal components are fitted
    on half of the resample differences and the covariance on the other half; the inverse covariance is
    corrected by the Wishart factor (n - k - 2) / (n - 1), unbiased if the noise is Gaussian and the same in
    every context;
  - coordinates whose resample differences never exceed round-off are noiseless: a flip along them is reported
    separately (``noiseless_*``) and not whitened;
  - for drug-dependent bits, ``*_x`` versions of ``euclid`` and ``whitened`` weight each pair's effect by its drug
    code (the signal a state x drug response needs).
* **Drug factor** (``drug_factor``): for a drug-dependent target, the decodability of the planted code from the
  drug embedding, a property of the embedding computed once per model.

``profile`` gives every bit at every cut of one support; ``table`` turns saved profiles into the compact
``PRESENCE_TABLE_<support>.csv``. The cuts are read through a model's ``contract.PRESENCE_SITES``: module
outputs (``capture``), module inputs (``input_cuts``) and elementwise products of two readings (``products``), plus
the model input itself (``input_cells``). Capture runs on the CPU (the model input is read with ``.numpy()``).
"""
import csv

import numpy as np

from ..assay.constants import N_CELLS
from .constants import LEVEL
from .draws import CARRIER_SEED_BOUND, draw, make_endpoint, seeded

__all__ = ["draws", "capture", "Embedder", "statistics", "roundoff_floor", "drug_factor",
           "profile", "table", "write_table"]

RFF_FEATURES = 1024          # random Fourier features of the kernel mean embedding
CALIBRATION_PER_BIT = 2      # flip pairs per bit, both states, that calibrate standardization and bandwidth
LINE_RESAMPLES = 2000        # line-bootstrap resamples of every statistic
FLOOR_BAGS = 6               # bags whose cells are permuted to measure the round-off floor
EMBED_DTYPE = np.float32     # bag embeddings are kept in float32, as saved
EMBEDDER_SEED = 20261017     # the random Fourier features and the bandwidth sample
RIDGE_PENALTIES = (1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100)   # the drug factor's penalties, times the number of drugs


# The compact table: its cuts per model, and per bit the null bit and the learned common bit of the same statistic
# (mean: 4 and 0; variance: 9 and 5) that each coefficient is divided by.
KEY_CUTS = {"deepsets": ["input_cells:mean", "input_cells:second", "input_cells:kernel", "backbone.phi.0:kernel",
                         "pooled", "backbone.rho.0", "backbone.rho.1", "backbone.rho.2"],
            "summary": ["input_cells:mean", "input_cells:second", "input_cells:kernel", "backbone.sproj",
                        "backbone.hproj", "backbone.common_hidden", "backbone.inter_hidden", "backbone.w0", "backbone.w1"],
            "state": ["input_cells:mean", "input_cells:second", "input_cells:kernel", "backbone.input:kernel",
                      "backbone.blocks.0:kernel", "backbone.blocks.3:kernel", "backbone.blocks.7:kernel",
                      "backbone.head.input", "backbone.head.1", "backbone.head.2"]}
NULL = {0: 4, 1: 4, 3: 4, 4: 4, 5: 9, 6: 9, 8: 9, 9: 9}
LEARNED = {0: 0, 1: 0, 3: 0, 4: 0, 5: 5, 6: 5, 8: 5, 9: 5}


def draws(task, contexts, group, root, per_context, bit):
    """Per context, ``per_context`` rows: a flip pair (minus, plus) and two resample bags in the minus state."""
    rows = []
    for j, context in enumerate(contexts):
        pool = task.pool(group, context)
        for i in range(per_context):
            rng = seeded(root, j, i, bit + 1)
            p, state, (minus, plus) = draw(task, context, pool, rng, bit)
            resamples = []
            for _ in range(2):
                cells = rng.permutation(len(pool[0]))[:N_CELLS]
                resamples.append(make_endpoint(task, context, pool, minus.state, p, cells,
                                               int(rng.integers(CARRIER_SEED_BOUND))))
            rows.append({"context": context, "line": context.split("|")[0], "p": p,
                         "code": float(task.generator.code[p]), "ends": (minus, plus, *resamples)})
    return rows


def capture(model, sites, x, depth, p):
    """Raw per-cut values for one input: per-cell outputs as (cells, features), set-level ones as vectors."""
    import torch

    captured, handles = {}, []
    modules = dict(model.named_modules())

    def output_reader(key):
        def hook(module, inputs, output):
            captured[key] = output.detach()[0].double().cpu().numpy()
        return hook

    def input_reader(cut, width):
        def hook(module, inputs):
            value = inputs[0].detach()[0]
            captured[cut] = (value if width is None else value[..., :width]).double().cpu().numpy()
        return hook

    for cut, (module, width) in sites.input_cuts.items():
        handles.append(modules[module].register_forward_pre_hook(input_reader(cut, width)))
    handles += [modules[key].register_forward_hook(output_reader(key)) for key in sites.capture]
    try:
        with torch.inference_mode():
            model(x, depth, torch.tensor([p]), LEVEL, canonicalize=True)
    finally:
        for handle in handles:
            handle.remove()
    for cut, (left, right) in sites.products.items():
        captured[cut] = captured[left] * captured[right]
    captured.pop("backbone.pert", None)   # a property of the drug, read by products only
    captured["input_cells"] = x[0].double().numpy()
    return captured


def endpoint_capture(model, sites, endpoint, permutation=None):
    x, depth = endpoint.x, endpoint.depth
    if permutation is not None:
        x, depth = x[:, permutation], depth[:, permutation]
    return capture(model, sites, x, depth, endpoint.p)


class Embedder:
    """Per cut: a set-level value as it is; per-cell values as cell mean, second moment and kernel mean embedding."""

    def __init__(self, calibration, seed=EMBEDDER_SEED):
        self.maps = {}
        rng = np.random.default_rng(seed)
        for cut in calibration[0]:
            values = [c[cut] for c in calibration]
            if values[0].ndim == 1:
                self.maps[cut] = None
                continue
            cells = np.concatenate(values)
            mean, sd = cells.mean(0), cells.std(0)
            positive = sd[sd > 0]
            fill = np.median(positive) if len(positive) else 1.0
            sd = np.where(sd > 1e-8 * fill, sd, fill)   # constant coordinates stay, scaled by the median spread
            z = (cells - mean) / sd
            sample = z[rng.choice(len(z), size=min(2000, len(z)), replace=False)]
            norms = (sample ** 2).sum(1)
            squared = np.maximum(norms[:, None] + norms[None, :] - 2 * sample @ sample.T, 0)
            bandwidth = float(np.sqrt(np.median(squared[np.triu_indices(len(sample), 1)])))
            weights = rng.normal(0, 1 / bandwidth, size=(z.shape[1], RFF_FEATURES))
            phases = rng.uniform(0, 2 * np.pi, size=RFF_FEATURES)
            self.maps[cut] = (mean, sd, weights, phases)

    def __call__(self, captured):
        out = {}
        for cut, value in captured.items():
            spec = self.maps.get(cut)
            if spec is None:
                out[cut] = value.astype(EMBED_DTYPE)
                continue
            mean, sd, weights, phases = spec
            z = (value - mean) / sd
            out[cut + ":mean"] = value.mean(0).astype(EMBED_DTYPE)
            out[cut + ":second"] = (value ** 2).mean(0).astype(EMBED_DTYPE)
            out[cut + ":kernel"] = (np.sqrt(2 / RFF_FEATURES) * np.cos(z @ weights + phases).mean(0)).astype(EMBED_DTYPE)
        return out


def _distances(a, b):
    sq = np.maximum((a ** 2).sum(1)[:, None] + (b ** 2).sum(1)[None, :] - 2 * a @ b.T, 0)
    return np.sqrt(sq)


class LineBlocks:
    """Line-block sums of a pair-by-pair matrix, so that a line resample is a weighted sum over lines.

    For a resample that draws line l w_l times, the mean over entries whose original rows differ is
    (w' B w - (w**2)' D) / ((w' n)**2 - (w**2)' n), with B the line-by-line block sums, D the per-line sums of
    the diagonal and n the pairs per line; a pair drawn twice never meets itself.
    """

    def __init__(self, lines):
        self.names, inverse = np.unique(lines, return_inverse=True)
        self.indicator = np.eye(len(self.names))[inverse]          # pairs x lines
        self.sizes = self.indicator.sum(0)

    def blocks(self, m):
        return self.indicator.T @ m @ self.indicator, self.indicator.T @ np.diag(m)

    def mean(self, blocks, w):
        b, d = blocks
        count = (w @ self.sizes) ** 2 - (w ** 2) @ self.sizes
        return float((w @ b @ w - (w ** 2) @ d) / count)


def _whitened(delta, noise):
    """delta' S^-1 delta between pairs, on the leading noise components; basis and S from alternate noise rows."""
    half_a, half_b = noise[0::2], noise[1::2]
    _, singular, vt = np.linalg.svd(half_a - half_a.mean(0), full_matrices=False)
    n = len(half_b)
    k = max(1, min(n // 2, int(np.sum(singular > singular[0] * 1e-8)) if singular[0] > 0 else 1))
    basis = vt[:k]
    noise_k = (half_b - half_b.mean(0)) @ basis.T
    covariance = noise_k.T @ noise_k / (n - 1) / 2 + 1e-12 * np.eye(k)   # one bag = half a resample difference
    wishart = (n - k - 2) / (n - 1)
    delta_k = delta @ basis.T
    return wishart * delta_k @ np.linalg.solve(covariance, delta_k.T), delta_k, k


def statistics(rows, embeddings, product, floors, rng):
    """Indistinguishability counts and the continuous statistics at every embedded cut, with line-resample intervals."""
    lines = LineBlocks(np.array([row["line"] for row in rows]))
    contexts = np.array([row["context"] for row in rows])
    same = (contexts[:, None] == contexts[None, :]).astype(float)
    codes = np.array([row["code"] for row in rows])
    weights = codes / np.mean(codes ** 2)
    count = len(lines.names)
    draws_ = [np.bincount(rng.integers(count, size=count), minlength=count).astype(float) for _ in range(LINE_RESAMPLES)]
    ones = np.ones(count)
    result = {}
    for cut in embeddings[0][0]:
        minus, plus, res_a, res_b = (np.stack([e[k][cut] for e in embeddings]).astype(np.float64) for k in range(4))
        delta, noise = plus - minus, res_a - res_b
        floor = floors[cut]
        noiseless = np.abs(noise).max(0) <= floor
        positive = ~noiseless
        noise_energy = float((noise ** 2).sum(1).mean())
        cross = _distances(plus, minus)
        null_cross = _distances(res_a, res_b)
        matrices = {"energy": cross + cross.T - _distances(plus, plus) - _distances(minus, minus),
                    "energy_null": null_cross + null_cross.T - _distances(res_a, res_a) - _distances(res_b, res_b),
                    "cross": cross, "null_cross": null_cross,
                    "euclid": delta @ delta.T / noise_energy if noise_energy > 0 else delta @ delta.T * np.inf}
        k = 0
        delta_k = np.zeros((len(delta), 0))
        if positive.any():
            matrices["whitened"], delta_k, k = _whitened(delta[:, positive], noise[:, positive])
        if product:
            outer = np.outer(weights, weights)
            matrices["euclid_x"] = matrices["euclid"] * outer
            if "whitened" in matrices:
                matrices["whitened_x"] = matrices["whitened"] * outer
        blocks = {name: lines.blocks(m) for name, m in matrices.items()}
        conditional = [name for name in ("energy", "energy_null", "cross", "null_cross", "euclid", "whitened")
                       if name in matrices]
        blocks.update({name + "_ctx": lines.blocks(matrices[name] * same) for name in conditional})
        blocks["same"] = lines.blocks(same)
        squared = (delta ** 2).sum(1)
        pair_energy = squared / noise_energy if noise_energy > 0 else squared * np.inf
        pair_line = lines.indicator.T @ pair_energy

        def compute(w):
            out = {name: lines.mean(blocks[name], w) for name in matrices if "cross" not in name}
            same_w = lines.mean(blocks["same"], w)
            for name in conditional:
                if "cross" not in name:
                    out[name + "_ctx"] = lines.mean(blocks[name + "_ctx"], w) / same_w
            for suffix in ("", "_ctx"):
                scale = same_w if suffix else 1.0
                denominator = 2 * lines.mean(blocks["cross" + suffix], w) / scale
                out["energy_coefficient" + suffix] = out["energy" + suffix] / denominator if denominator > 0 else 0.0
                null_denominator = 2 * lines.mean(blocks["null_cross" + suffix], w) / scale
                out["energy_null_coefficient" + suffix] = (out["energy_null" + suffix] / null_denominator
                                                           if null_denominator > 0 else 0.0)
            out["pair_energy"] = float(w @ pair_line / (w @ lines.sizes))
            return out

        point = compute(ones)
        boot = [compute(w) for w in draws_]
        result[cut] = {key: {"value": point[key],
                             "interval": [float(np.quantile([b[key] for b in boot], 0.025)),
                                          float(np.quantile([b[key] for b in boot], 0.975))]}
                       for key in point}
        total = float(squared.mean())
        silent = float((delta[:, noiseless] ** 2).sum(1).mean())
        inside = float((delta_k ** 2).sum(1).mean())
        result[cut]["noiseless_coordinates"] = int(noiseless.sum())
        result[cut]["noiseless_flip_coordinates"] = int((np.abs(delta[:, noiseless]).max(0) > floor[noiseless]).sum())
        result[cut]["noiseless_share"] = silent / total if total > 0 else 0.0
        # The rest of the flip's energy lies on noisy coordinates outside the leading noise components.
        result[cut]["beyond_components_share"] = 1 - (silent + inside) / total if total > 0 else 0.0
        result[cut]["coordinates_above_floor"] = int((np.abs(delta).max(0) > floor).sum())
        result[cut]["max_abs_flip"] = float(np.abs(delta).max())
        result[cut]["features"] = int(delta.shape[1])
        result[cut]["whitened_components"] = int(k)
    return result


def roundoff_floor(model, sites, embed, rows, rng):
    """Per cut and coordinate, the largest change of the bag embedding when the same bag's cells are permuted."""
    import torch

    floors = {}
    for row in rows[:FLOOR_BAGS]:
        endpoint = row["ends"][0]
        permutation = torch.from_numpy(rng.permutation(endpoint.x.shape[1]))
        a = embed(endpoint_capture(model, sites, endpoint))
        b = embed(endpoint_capture(model, sites, endpoint, permutation))
        for cut in a:
            change = np.abs(a[cut].astype(np.float64) - b[cut])
            floors[cut] = np.maximum(floors[cut], change) if cut in floors else change
    return floors


def drug_factor(model, task):
    """Leave-one-out ridge R^2 of the planted drug code from the drug embedding, over the drug menu."""
    embedding = model.backbone.pert.weight.detach().double().cpu().numpy()
    code = np.asarray(task.generator.code, dtype=np.float64)[: len(embedding)]
    x = embedding - embedding.mean(0)
    y = code - code.mean()
    best = None
    for penalty in RIDGE_PENALTIES:
        hat = x @ np.linalg.solve(x.T @ x + penalty * len(x) * np.eye(x.shape[1]), x.T)
        residual = (y - hat @ y) / np.maximum(1 - np.diag(hat), 1e-12)
        r2 = 1 - np.mean(residual ** 2) / np.mean(y ** 2)
        if best is None or r2 > best[0]:
            best = (float(r2), penalty)
    return {"loo_r2": best[0], "penalty": best[1], "drugs": int(len(embedding))}


def profile(model, sites, task, contexts, group, root, per_context, bits, product_bits, seed=0):
    """Presence statistics for every bit at every cut, with exact-invariance floors; one embedder per support.

    Inputs are drawn one bit at a time and dropped once embedded: a bag's carrier holds 512 cells x 2,000 genes,
    so keeping every bit's draws at once would need tens of gigabytes.
    """
    calibration = []
    for bit in bits:   # the first pairs of every bit, both states, so that every planted cue varies
        for row in draws(task, contexts[:1], group, root, CALIBRATION_PER_BIT, bit):
            calibration += [endpoint_capture(model, sites, end) for end in row["ends"][:2]]
    embed = Embedder(calibration)
    del calibration
    rng = np.random.default_rng(seed)
    floors = roundoff_floor(model, sites, embed, draws(task, contexts[:1], group, root, FLOOR_BAGS, bits[0]), rng)
    out = {"roundoff_floor_max": {cut: float(f.max()) for cut, f in floors.items()}, "bits": {}}
    for bit in bits:
        rows = draws(task, contexts, group, root, per_context, bit)
        embeddings = []
        for row in rows:
            embeddings.append(tuple(embed(endpoint_capture(model, sites, end)) for end in row["ends"]))
            row["ends"] = None   # free the carrier tensors
        stats = statistics(rows, embeddings, bit in product_bits, floors, rng)
        # Indistinguishable at a cut on the tested pairs: no coordinate of any of its embeddings exceeds its floor.
        indistinguishable = {}
        for key, values in stats.items():
            base = key.split(":")[0]
            indistinguishable[base] = indistinguishable.get(base, True) and values["coordinates_above_floor"] == 0
        out["bits"][bit] = {"pairs": len(rows), "lines": len({row["line"] for row in rows}), "cuts": stats,
                            "indistinguishable_at_cut": indistinguishable}
        del embeddings, rows
    return out


def table(profiles, support="seen"):
    """Compact presence rows of ``profiles``, pairs of a case and a saved profile (case ``parent`` for the original
    model's own, the planted fault's case for a planted-fault parent), in the order given: per half, model, seed,
    case, bit and key cut, the energy coefficient (pooled and conditional on the background) with its interval, its
    ratio to the null bit and to the learned common bit at the same cut, the per-pair displacement, the Euclidean
    and whitened readings, the noiseless share, whether the cut is numerically indistinguishable on the tested
    pairs, and the drug factor."""
    rows = []
    for case, result in profiles:
        bits = result["supports"][support]["bits"]
        for bit, values in bits.items():
            b = int(bit)
            for cut in KEY_CUTS[result["model"]]:
                if cut not in values["cuts"]:
                    continue
                st = values["cuts"][cut]
                coefficient = st["energy_coefficient"]["value"]
                null = bits[str(NULL[b])]["cuts"][cut]["energy_coefficient"]["value"]
                learned = bits[str(LEARNED[b])]["cuts"][cut]["energy_coefficient"]["value"]
                value = lambda key: f"{st[key]['value']:.3g}" if key in st else ""
                rows.append({"half": result["half"], "model": result["model"], "seed": result["seed"], "case": case, "bit": b,
                             "cut": cut, "energy_coefficient": f"{coefficient:.3g}",
                             "interval": f"[{st['energy_coefficient']['interval'][0]:.3g}, {st['energy_coefficient']['interval'][1]:.3g}]",
                             "ratio_to_null": f"{coefficient / null:.3g}" if null > 0 else "",
                             "ratio_to_learned": f"{coefficient / learned:.3g}" if learned > 0 else "",
                             "energy_coefficient_ctx": value("energy_coefficient_ctx"),
                             "pair_energy": value("pair_energy"), "euclid": value("euclid"),
                             "euclid_ctx": value("euclid_ctx"), "whitened": value("whitened"),
                             "whitened_ctx": value("whitened_ctx"),
                             "noiseless_share": f"{st['noiseless_share']:.3g}",
                             "indistinguishable": values["indistinguishable_at_cut"][cut.split(":")[0]],
                             "drug_factor_loo_r2": f"{result['drug_factor']['loo_r2']:.3g}"})
    return rows


def write_table(path, rows):
    """Write the compact table (``PRESENCE_TABLE_<support>.csv``)."""
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
