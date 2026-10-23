"""Official State for the paper's tasks.

The model is the State transition model, loaded unchanged from ``third_party/arc-state-9bbfe78``; two
imports it does not use at run time are stubbed (``FinetuneVCICountsDecoder`` and ``anndata``).
``AnchoredState`` changes its output only: the output activation is the identity instead of ReLU, the
output projection starts at zero, and each output cell is the input control cell plus the perturbation's
mean offset ``r[p]`` plus the model's correction. The hyperparameters are those released with the
ST-HVG-Tahoe model (``official_config.yaml``), with this task's sizes.

Training uses the model's own loss (energy distance between the predicted and the treated 256-cell sets;
in the synthetic task on genes 50 onward), AdamW without weight decay, a learning rate that warms up over
1,000 updates to 3e-5 and decays by a cosine to 10% at 160,000 updates, and gradient-norm clipping at 1.
The model is read out after ``STEPS`` updates.
"""
from copy import deepcopy
import importlib
from importlib.machinery import ModuleSpec
import math
from pathlib import Path
import sys
import types

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
import yaml

from scsailor.models import set_runtime

UPSTREAM = Path(__file__).resolve().parents[3] / "third_party" / "arc-state-9bbfe78"
STEPS = 5000          # read-out
TOTAL = 160000        # length of the learning-rate schedule
PEAK = 3e-5
REPLAY_CHUNK = 93     # perturbation sets per forward in synthetic replay


def lr_at(step):
    if step <= 1000:
        return PEAK * (.01 + .99 * (step - 1) / 999)
    return PEAK * (.1 + .45 * (1 + math.cos(math.pi * (step - 1000) / (TOTAL - 1000))))


def model_kwargs(p_count):
    kwargs = deepcopy(yaml.safe_load((UPSTREAM / "official_config.yaml").read_text())["model"]["kwargs"])
    kwargs.update(input_dim=2000, output_dim=2000, gene_dim=2000, pert_dim=p_count, batch_dim=None,
                  embed_key="X_hvg", output_space="gene", basal_mapping_strategy="random", control_pert="DMSO_TF")
    return kwargs


def _model_class():
    """The upstream model class, imported from the pinned files without the package's initializers."""
    package = "state_pinned"
    if package not in sys.modules:
        module = types.ModuleType(package)
        module.__path__ = [str(UPSTREAM / "src" / "state")]
        module.__spec__ = ModuleSpec(package, loader=None, is_package=True)
        sys.modules[package] = module
        decoders = types.ModuleType(package + ".tx.models.decoders")
        decoders.__spec__ = ModuleSpec(decoders.__name__, loader=None)

        class FinetuneVCICountsDecoder:
            def __init__(self, *args, **kwargs):
                raise RuntimeError("FinetuneVCICountsDecoder is not used by these runs")

        decoders.FinetuneVCICountsDecoder = FinetuneVCICountsDecoder
        sys.modules[decoders.__name__] = decoders
        anndata = types.ModuleType("anndata")
        anndata.__spec__ = ModuleSpec("anndata", loader=None)
        sys.modules["anndata"] = anndata
    return importlib.import_module(package + ".tx.models.state_transition").StateTransitionPerturbationModel


class AnchoredState(_model_class()):
    def __init__(self, r_pool, **kwargs):
        super().__init__(**kwargs)
        self.register_buffer("r_pool", torch.tensor(r_pool, dtype=torch.float32))
        self.relu = torch.nn.Identity()
        torch.nn.init.zeros_(self.project_out[0].weight)
        torch.nn.init.zeros_(self.project_out[0].bias)

    def forward(self, batch, padded=True):
        delta = super().forward(batch, padded)
        control = batch["ctrl_cell_emb"].reshape(-1, 2000)
        codes = batch["pert_emb"].reshape(-1, self.r_pool.shape[0]).argmax(-1)
        return control + self.r_pool[codes] + delta


class SyntheticState(AnchoredState):
    """The synthetic-answer assay scores genes 50 onward; the loss covers the same genes."""

    def _compute_distribution_loss(self, pred, target):
        return super()._compute_distribution_loss(pred[..., 50:], target[..., 50:])


class Adapter:
    """State on one task: ``task`` is "natural" or "synthetic"; ``r`` holds the perturbations' mean offsets."""

    steps = STEPS

    def __init__(self, task, seed, r, device):
        self.task, self.seed, self.device = task, seed, device
        self.r = np.array(r, dtype=np.float32, copy=True)
        self.p_count = len(self.r)

    def build(self):
        kwargs = model_kwargs(self.p_count)
        set_runtime(self.seed)
        model_type = SyntheticState if self.task == "synthetic" else AnchoredState
        return model_type(self.r, **kwargs).to(self.device)

    def load(self, checkpoint):
        model = self.build()
        model.load_state_dict(torch.load(checkpoint, map_location=self.device, weights_only=False)["model"])
        return model.eval()

    def optimizers(self, model):
        return {"optimizer": torch.optim.AdamW(list(model.parameters()), lr=lr_at(1), betas=(0.9, 0.999), eps=1e-8,
                                               weight_decay=0, fused=False)}

    def _one_hot(self, codes):
        codes = torch.as_tensor(codes, device=self.device)
        return torch.nn.functional.one_hot(codes, self.p_count).float().repeat_interleave(256, dim=0)

    def update(self, model, optimizers, batch, step):
        optimizer = optimizers["optimizer"]
        tensors = dict(ctrl_cell_emb=torch.from_numpy(batch["atoms"]).reshape(2048, 2000).to(self.device),
                       pert_cell_emb=torch.from_numpy(batch["target"]).reshape(2048, 2000).to(self.device),
                       pert_emb=self._one_hot(batch["codes"]))
        optimizer.zero_grad(set_to_none=True)
        for group in optimizer.param_groups:
            group["lr"] = lr_at(step)
        loss = model.training_step(tensors, step)
        loss.backward()
        clip_grad_norm_(list(model.parameters()), 1.0, error_if_nonfinite=True)
        optimizer.step()
        return loss.detach()

    # natural task: per evaluation draw, ``prepare`` once, then ``predict_chunk`` per group of perturbations
    def prepare(self, cells):
        return torch.as_tensor(cells, dtype=torch.float32, device=self.device)

    def predict_chunk(self, model, x, codes):
        prediction = model({"ctrl_cell_emb": x.repeat(len(codes), 1), "pert_emb": self._one_hot(codes)})
        return prediction.cpu().numpy().reshape(len(codes), 256, 2000)

    # synthetic task
    def predictor(self, model, codes=None):
        """``cells -> (raw, baseline)``: mean predicted profiles and baselines (codes x genes, float64)."""
        codes = list(range(self.p_count)) if codes is None else list(codes)
        response = torch.as_tensor(self.r, dtype=torch.float32, device=self.device)

        def predict(cells):
            x = torch.as_tensor(cells, dtype=torch.float32, device=self.device)
            raw, baseline = [], []
            for start in range(0, len(codes), REPLAY_CHUNK):
                selected = codes[start:start + REPLAY_CHUNK]
                output = model({"ctrl_cell_emb": x.repeat(len(selected), 1), "pert_emb": self._one_hot(selected)})
                raw.append(output.reshape(len(selected), 256, 2000).cpu().numpy().mean(1, dtype=np.float64))
                # the baseline is summed per cell in float32, then averaged in float64
                baseline.append((x[None] + response[selected, None]).cpu().numpy().mean(1, dtype=np.float64))
            return np.concatenate(raw), np.concatenate(baseline)

        return predict
