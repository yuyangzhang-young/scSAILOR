"""PrePR-CT for the paper's tasks.

The network (``MLP``, ``GNN``) and the loss (``loss_fct``, an exact earth mover's distance per input set)
are read from the unchanged files of ``third_party/prepr-ct-v1.0.0`` and executed without the statement
``torch.manual_seed(42)`` in ``GNN.__init__`` (``patches/prepr_ct.patch``), so that the run's seed sets the
initialization. Configuration: 2,000 genes, hidden size 128, one attention head, Sigmoid activation and
molecular-descriptor perturbation input.

Changes for this task: the last prediction layer starts at zero; the network's per-cell corrections of each
256-cell input set are averaged into one profile and added to the set's mean plus the perturbation's mean
offset ``r[p]`` (the "bag-shared" output). Each set's gene graph comes from ``graph.graph``. Training uses
Adam (learning rate 1e-3, weight decay 1e-5) over all parameters; in the synthetic task the loss covers
genes 50 onward. The model is read out after ``STEPS`` updates.
"""
import ast
from pathlib import Path

import numpy as np
import ot
import torch
from torch_geometric.nn import GATConv, Linear, TransformerConv

from .graph import graph

UPSTREAM = Path(__file__).resolve().parents[3] / "third_party" / "prepr-ct-v1.0.0"
STEPS = 10000
LABELS = np.repeat(np.arange(8), 256)   # input set of each of the 2,048 training rows


def _definitions():
    """``GNN`` and ``loss_fct`` from the upstream files, without the fixed seed in ``GNN.__init__``."""
    namespace = dict(torch=torch, np=np, ot=ot, TransformerConv=TransformerConv, GATConv=GATConv, Linear=Linear)
    for filename, names in (("model.py", ("MLP", "GNN")), ("utils.py", ("loss_fct",))):
        module = ast.parse((UPSTREAM / filename).read_text())
        kept = [node for node in module.body if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names]
        for node in kept:
            if node.name == "GNN":
                init = next(n for n in node.body if isinstance(n, ast.FunctionDef) and n.name == "__init__")
                seed = [n for n in init.body if isinstance(n, ast.Expr) and ast.unparse(n) == "torch.manual_seed(42)"]
                if len(seed) != 1:
                    raise ValueError("The PrePR-CT seed statement was not found exactly once")
                init.body.remove(seed[0])
        exec(compile(ast.Module(body=kept, type_ignores=[]), str(UPSTREAM / filename), "exec"), namespace)
    return namespace["GNN"], namespace["loss_fct"]


GNN, loss_fct = _definitions()


def shared_forward(model, nodes, edges, labels, keys, control, perturbation, r_rows):
    """Bag-shared output: per input set, the mean of the network's per-cell outputs, added to the set's mean
    plus ``r[p]``."""
    blocks = (-1, 256, control.shape[-1])
    shared = model(nodes, edges, labels, keys, control, perturbation, None).reshape(blocks).mean(1, keepdim=True)
    base = control.reshape(blocks).mean(1, keepdim=True)
    return (base + r_rows.reshape(blocks) + shared).reshape_as(control)


class Adapter:
    """PrePR-CT on one task: ``task`` is "natural" or "synthetic".

    ``r`` holds the mean offsets and ``features`` the molecular descriptors; ``feature_row`` and ``r_row``
    map a perturbation code to its row in each (identity unless some perturbations have no descriptors)."""

    steps = STEPS

    def __init__(self, task, seed, r, features, device, feature_row=None, r_row=None):
        self.task, self.seed, self.device = task, seed, device
        self.r_values = np.asarray(r)   # as given; the synthetic baseline is computed from it in float64
        self.r = torch.as_tensor(r, dtype=torch.float32, device=device)
        self.features = torch.as_tensor(features, dtype=torch.float32, device=device)
        self.feature_row = np.arange(len(features)) if feature_row is None else np.asarray(feature_row)
        self.r_row = np.arange(len(r)) if r_row is None else np.asarray(r_row)

    def build(self):
        torch.manual_seed(self.seed)
        model = GNN(total_genes=2000, num_perts=93, hidden_channels=128, in_head=1, act_fct="Sigmoid",
                    multi_pert=True).to(self.device)
        head = model.lin_predict.network[-1]
        torch.nn.init.zeros_(head.weight)
        torch.nn.init.zeros_(head.bias)
        return model

    def load(self, checkpoint):
        model = self.build()
        model.load_state_dict(torch.load(checkpoint, map_location=self.device, weights_only=False)["model"])
        return model.eval()

    def optimizers(self, model):
        return {"optimizer": torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)}

    def _rows(self, codes, repeat):
        codes = np.asarray(codes)
        perturbation = self.features[torch.as_tensor(self.feature_row[codes], device=self.device)]
        r_rows = self.r[torch.as_tensor(self.r_row[codes], device=self.device)]
        return perturbation.repeat_interleave(repeat, dim=0), r_rows.repeat_interleave(repeat, dim=0)

    def update(self, model, optimizers, batch, step):
        x = torch.from_numpy(batch["atoms"]).to(self.device)
        target = torch.from_numpy(batch["target"]).reshape(2048, 2000).to(self.device)
        perturbation, r_rows = self._rows(batch["codes"], 256)
        nodes, edges = {}, {}
        for slot in range(8):
            nodes[slot], edges[slot], _ = graph(x[slot])
        optimizers["optimizer"].zero_grad(set_to_none=True)
        prediction = shared_forward(model, nodes, edges, LABELS.tolist(), range(8), x.reshape(2048, 2000),
                                    perturbation, r_rows)
        if self.task == "synthetic":
            loss = loss_fct(prediction[:, 50:], target[:, 50:], LABELS)
        else:
            loss = loss_fct(prediction, target, LABELS)
        loss.backward()
        optimizers["optimizer"].step()
        return loss.detach()

    # natural task: the gene graph uses the float64 cells, the network float32
    def prepare(self, cells):
        x = torch.as_tensor(cells, dtype=torch.float64, device=self.device)
        nodes, edges, _ = graph(x)
        return x, nodes, edges

    def predict_chunk(self, model, prepared, codes):
        x, nodes, edges = prepared
        control = x.to(torch.float32).repeat(len(codes), 1)
        perturbation, r_rows = self._rows(codes, 256)
        prediction = shared_forward(model, {0: nodes}, {0: edges}, [0] * len(control), [0], control, perturbation, r_rows)
        return prediction.cpu().numpy().reshape(len(codes), 256, 2000)

    # synthetic task
    def predictor(self, model, codes=None):
        """``cells -> (raw, baseline)`` for all perturbations in one forward."""
        codes = list(range(len(self.r))) if codes is None else list(codes)

        def predict(cells):
            x = torch.as_tensor(cells, dtype=torch.float64, device=self.device)
            nodes, edges, _ = graph(x)
            control = x.to(torch.float32).repeat(len(codes), 1)
            perturbation, r_rows = self._rows(codes, 256)
            prediction = shared_forward(model, {0: nodes}, {0: edges}, [0] * len(control), [0], control, perturbation,
                                        r_rows)
            raw = prediction.cpu().numpy().reshape(len(codes), 256, 2000).mean(1, dtype=np.float64)
            baseline = cells.mean(0, dtype=np.float64)[None] + np.asarray(self.r_values, dtype=np.float64)[codes]
            return raw, baseline

        return predict
