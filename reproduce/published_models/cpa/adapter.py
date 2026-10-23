"""CPA, as implemented in the State repository, for the paper's tasks.

The model code is the CPA implementation of ``third_party/arc-state-9bbfe78``. It is copied to a temporary
directory and changed there by the two one-line substitutions of ``patches/cpa.patch`` (a model with a
single batch has no batch embedding); the files in ``third_party`` stay unchanged. The logging-only
``torch_scatter`` import is stubbed. Hyperparameters are the repository's CPA configuration.

Changes for this task: the decoder's mean layer starts at zero, and the Gaussian mean of every cell is the
mean of its 256-cell input set plus the perturbation's mean offset ``r[p]`` plus the decoder output. The
adversarial discriminator is not trained. Training uses Adam from the model's own ``configure_optimizers``
(learning rate 5e-4) with its step decay applied every 25,000 updates; the loss is the model's own
(natural task) or the Gaussian negative log-likelihood on genes 50 onward (synthetic task). The model is
read out after ``STEPS`` updates.
"""
import importlib
from importlib.machinery import ModuleSpec
from pathlib import Path
import shutil
import sys
import tempfile
import types

import numpy as np
import torch
import yaml

UPSTREAM = Path(__file__).resolve().parents[3] / "third_party" / "arc-state-9bbfe78"
STEPS = 140000
# The two changes of patches/cpa.patch, applied to the temporary copy.
PATCHES = {
    "tx/models/cpa/_module.py": (
        "        z_batch = self.batch_embeddings(batch_ids.long())",
        '        z_batch = self.batch_embeddings(batch_ids.long()) if hasattr(self, "batch_embeddings") else 0.0'),
    "tx/models/cpa/_model.py": (
        "            + list(filter(lambda p: p.requires_grad, self.module.batch_embeddings.parameters()))",
        "            + (list(filter(lambda p: p.requires_grad, self.module.batch_embeddings.parameters())) "
        "if self.n_batches > 1 else [])"),
}
_runtime = None


def _model_class():
    global _runtime
    torch.distributions.Distribution.set_default_validate_args(False)
    stub = types.ModuleType("torch_scatter")
    stub.__spec__ = ModuleSpec("torch_scatter", loader=None)

    def scatter_mean(*args, **kwargs):
        raise RuntimeError("torch_scatter.scatter_mean is not used by these runs")

    stub.scatter_mean = scatter_mean
    sys.modules["torch_scatter"] = stub
    if _runtime is None:
        _runtime = tempfile.TemporaryDirectory(prefix="cpa-")
        root = Path(_runtime.name)
        shutil.copytree(UPSTREAM / "src", root / "src")
        for filename, (before, after) in PATCHES.items():
            path = root / "src" / "state" / filename
            text = path.read_text()
            if text.count(before) != 1:
                raise ValueError(f"The CPA patch does not match {filename}")
            path.write_text(text.replace(before, after, 1))
        package = types.ModuleType("cpa_pinned")
        package.__path__ = [str(root / "src" / "state")]
        package.__spec__ = ModuleSpec("cpa_pinned", loader=None, is_package=True)
        sys.modules["cpa_pinned"] = package
    return importlib.import_module("cpa_pinned.tx.models.cpa").CPAPerturbationModel


def model_kwargs(seed, p_count):
    configs = UPSTREAM / "src" / "state" / "configs"
    kwargs = yaml.safe_load((configs / "model" / "cpa.yaml").read_text())["kwargs"].copy()
    training = yaml.safe_load((configs / "training" / "cpa.yaml").read_text())
    keys = ("n_epochs_kl_warmup", "n_steps_adv_warmup", "n_steps_pretrain_ae", "adv_steps", "reg_adv", "pen_adv",
            "lr", "wd", "adv_lr", "adv_wd", "step_size_lr", "do_clip_grad", "adv_loss", "batch_size")
    kwargs.update({key: training[key] for key in keys})
    for key in ("lr", "wd", "adv_lr", "adv_wd"):
        kwargs[key] = float(kwargs[key])
    kwargs.update(input_dim=2000, output_dim=2000, pert_dim=p_count, n_perts=p_count, n_cell_types=1, n_batches=1,
                  seed=seed)
    return kwargs


def anchored_step(model, batch, r_pool):
    """Encoder and decoder outputs, with the decoder's mean moved to input-set mean + r[p] + decoder output."""
    codes = batch["pert_emb"].argmax(-1)
    enc, dec = model._forward_step(batch)
    control = batch["ctrl_cell_emb"]
    bag = control.reshape(-1, 256, control.shape[1]).mean(1, keepdim=True).expand(-1, 256, -1).reshape(control.shape)
    loc = bag + r_pool[codes] + dec["px"].loc
    return enc, dict(dec, px=torch.distributions.Normal(loc=loc, scale=dec["px"].scale))


def masked_nll(px, target):
    scored = torch.distributions.Normal(px.loc[:, 50:], px.scale[:, 50:])
    return -scored.log_prob(target[:, 50:]).sum(-1).mean()


class Adapter:
    """CPA on one task: ``task`` is "natural" or "synthetic"; ``r`` holds the perturbations' mean offsets."""

    steps = STEPS

    def __init__(self, task, seed, r, device):
        self.task, self.seed, self.device = task, seed, device
        self.r = np.asarray(r, dtype=np.float32)
        self.p_count = len(self.r)
        self.r_pool = torch.tensor(self.r, dtype=torch.float32, device=device)

    def build(self):
        model = _model_class()(**model_kwargs(self.seed, self.p_count)).to(self.device)
        torch.nn.init.zeros_(model.module.decoder.mean_encoder.weight)
        torch.nn.init.zeros_(model.module.decoder.mean_encoder.bias)
        return model

    def load(self, checkpoint):
        model = self.build()
        model.load_state_dict(torch.load(checkpoint, map_location=self.device, weights_only=False)["model"])
        return model.eval()

    def optimizers(self, model):
        (optimizer, _adversary), (scheduler, *_) = model.configure_optimizers()
        return {"optimizer": optimizer, "scheduler": scheduler}

    def _inputs(self, x, codes):
        n = len(x)
        return dict(ctrl_cell_emb=x,
                    pert_emb=torch.nn.functional.one_hot(torch.as_tensor(codes, device=self.device),
                                                         self.p_count).float().repeat_interleave(256, dim=0),
                    cell_type_onehot=torch.ones(n, 1, device=self.device), batch=torch.ones(n, 1, device=self.device))

    def update(self, model, optimizers, batch, step):
        inputs = self._inputs(torch.from_numpy(batch["atoms"]).reshape(2048, 2000).to(self.device), batch["codes"])
        target = torch.from_numpy(batch["target"]).reshape(2048, 2000).to(self.device)
        enc, dec = anchored_step(model, dict(inputs, pert_cell_emb=target), self.r_pool)
        if self.task == "synthetic":
            loss = masked_nll(dec["px"], target)
        else:
            reconstruction, kl = model.module.loss(x_pert=target, encoder_outputs=enc, decoder_outputs=dec)
            loss = reconstruction + model.kl_weight * kl
        optimizers["optimizer"].zero_grad(set_to_none=True)
        loss.backward()
        optimizers["optimizer"].step()
        if step % (model.step_size_lr * 1000) == 0:
            optimizers["scheduler"].step()
        return loss.detach()

    # natural task
    def prepare(self, cells):
        return torch.as_tensor(cells, dtype=torch.float32, device=self.device)

    def predict_chunk(self, model, x, codes):
        _, dec = anchored_step(model, self._inputs(x.repeat(len(codes), 1), codes), self.r_pool)
        return dec["px"].loc.cpu().numpy().reshape(len(codes), 256, 2000)

    # synthetic task
    def predictor(self, model, codes=None):
        """``cells -> (raw, baseline)``. The cells are encoded once and decoded for all perturbations in one batch;
        in eval mode the rows are independent, so this equals one forward per perturbation."""
        if model.training or model.module.variational or model.module.encode_dosage:
            raise ValueError("Encode-once replay needs the deterministic CPA in eval mode")
        codes = list(range(self.p_count)) if codes is None else list(codes)
        module = model.module

        def predict(cells):
            x = torch.as_tensor(cells, device=self.device)
            basal = module.encoder(x)
            covariate = module.cell_type_embeddings(torch.zeros(256, dtype=torch.long, device=self.device))
            p = torch.as_tensor(codes, dtype=torch.long, device=self.device)
            z = basal[None] + module.pert_embeddings(p)[:, None] + covariate[None] + 0.0
            delta = module.forward_decoder(z.reshape(-1, basal.shape[-1]), library=(None, None))["px"].loc
            baseline = x.mean(0)[None] + self.r_pool[p]
            cells_out = baseline[:, None] + delta.reshape(len(p), 256, 2000)
            return (cells_out.cpu().numpy().mean(1, dtype=np.float64), baseline.cpu().numpy().astype(np.float64))

        return predict
