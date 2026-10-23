"""The natural task's training stream and learning-rate schedule.

Each update draws eight training units (``[seed, step, 0]``), for each unit m cells from its control pool
(``[seed, step, slot, 1]``) and the carrier's technical draws (``[seed, step, slot, 2]``); the batch content
does not depend on the number of worker threads. ``lr_at`` warms up linearly over 1,000 updates and decays
by a cosine to 10% of the peak at ``TOTAL`` updates (the built models' schedule).
"""
from concurrent.futures import ThreadPoolExecutor
import math
import time
import h5py
import numpy as np
from ..carrier import make
TOTAL=40000

class NaturalData:
    def __init__(self, directory):
        with h5py.File(directory / "controls.h5") as h:
            self.pools = {int(key): (h[key]["input/x"][:], h[key]["input/depth"][:]) for key in h}
        z = np.load(directory / "train_targets.npz")
        self.target = z["target"]
        self.parent = z["parent"]
        self.pcode = z["p_code"]
        self.prob = z["weight"]
        self.intercept = z["intercept"]
        self.p_count = len(z["p_ids"])

    def batch(self, step, m, level, seed):
        row_rng = np.random.default_rng(np.random.SeedSequence([seed, step, 0]))
        rows = row_rng.choice(len(self.target), size=8, replace=True, p=self.prob)
        x = np.zeros((8, 512, 2000), dtype=np.float32) if level == "L0" else np.empty((8, 512, 2000), dtype=np.float32)
        depth = np.empty((8, 512), dtype=np.int64)
        for slot, row in enumerate(rows):
            pool, d = self.pools[int(self.parent[row])]
            atoms = np.random.default_rng(np.random.SeedSequence([seed, step, slot, 1])).permutation(len(pool))[:m]
            if level == "L0":
                depth[slot] = np.repeat(np.sort(d[atoms], kind="stable"), 512 // m)
            else:
                rng = np.random.default_rng(np.random.SeedSequence([seed, step, slot, 2]))
                x[slot], depth[slot] = make(pool[atoms], d[atoms], level, rng)
        return x, depth, self.pcode[rows].astype(np.int64), self.target[rows]

def lr_at(step, peak):
    if step <= 1000:
        return peak * (.01 + .99 * (step - 1) / 999)
    return peak * (.1 + .45 * (1 + math.cos(math.pi * (step - 1000) / (TOTAL - 1000))))

def ordered_batches(data, start, stop, m, level, seed, workers):
    with ThreadPoolExecutor(max_workers=workers) as executor:
        pending = {}
        next_step = start
        def fill():
            nonlocal next_step
            while next_step <= stop and len(pending) < 2 * workers:
                pending[next_step] = executor.submit(data.batch, next_step, m, level, seed)
                next_step += 1
        fill()
        for step in range(start, stop + 1):
            before = time.perf_counter()
            batch = pending.pop(step).result()
            waited = time.perf_counter() - before
            fill()
            yield step, batch, waited
