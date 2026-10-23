"""A built model's prediction: the baseline (anchor) plus the backbone's correction.

The baseline is the input set's mean (at L0, the mean of the training controls) plus the perturbation's
mean offset ``r_pool[p]``. Rows are sorted by sequencing depth before the backbone sees them.
``set_runtime`` fixes every seed and disables reduced-precision matrix products.
"""
import random
import numpy as np
import torch
from torch import nn
from .backbones import build


def set_runtime(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    torch.set_num_threads(1)


class RawModel(nn.Module):
    def __init__(self, name, intercept, p_count, r_input, c_train):
        super().__init__()
        self.backbone = build(name, intercept)
        self.backbone.pert = nn.Embedding(p_count, 128)
        nn.init.normal_(self.backbone.pert.weight, std=.02)
        self.register_buffer('r_pool', torch.as_tensor(r_input, dtype=torch.float32))
        self.register_buffer('c_train', torch.as_tensor(c_train, dtype=torch.float32))
        heads = ([self.backbone.head[-1]] if name == 'state' else
                 [self.backbone.rho[-1]] if name == 'deepsets' else
                 [self.backbone.w0, self.backbone.w1])
        for head in heads:
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    @staticmethod
    def canonical_rows(x, depth):
        order = torch.argsort(depth, dim=1, stable=True)
        return x.gather(1, order[..., None].expand_as(x)), depth.gather(1, order)

    def anchor(self, x, p, level):
        s = self.c_train.expand(len(p), -1) if level == 'L0' else x.mean(dim=1)
        return s + self.r_pool[p]

    def forward(self, x, depth, p, level, canonicalize=True):
        if canonicalize:
            x, depth = self.canonical_rows(x, depth)
        # The backbone adds its intercept and it is subtracted here; kept as trained, for identical outputs.
        delta = self.backbone(x, depth, p, 256, level) - self.backbone.intercept
        return self.anchor(x, p, level) + delta


def raw_mean_then_clip(values):
    raw = np.mean(np.asarray(values, dtype=np.float64), axis=0)
    return raw, np.maximum(raw, 0)
