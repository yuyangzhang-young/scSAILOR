"""What information localization adds to the assay's model contract.

Every call the method makes on a network, with its shapes, is the assay's (``scsailor.assay.contract``).
Localization adds, for the built models of ``scsailor.models`` (``summary``, the summary-statistics model;
``deepsets``; ``state``, the attention model):

``CUTS[name][cut]``
    A cut divides a network into a front and the part behind it, as parameter-name prefixes:
    ``(front, trained)``. A start trains the part behind the cut with the front frozen; ``trained`` None trains
    every parameter (the cut ``input``, whose front is empty). The cuts: ``summary`` ``projections`` (behind
    the mean and variance projections), ``deepsets`` ``phi`` (behind the per-cell encoder), ``state``
    ``block3`` (behind the fourth of eight blocks) and ``head`` (behind the last block).
``FRONT_PARTS``, ``BRANCH_ARMS``
    The summary-statistics model's parts that single arms reset or train alone: the mean projection
    (``backbone.sproj``) and the variance projection (``backbone.hproj``) of its front, and its two branches
    behind ``projections``: the interaction branch, whose hidden layer is multiplied by the perturbation
    embedding, and the common branch, which reads the embedding as an input.
``PRESENCE_SITES[name]``
    Where the presence profile reads a network, as module names: module outputs (``capture``), the first
    features of a module's input (``input_cuts``) and elementwise products of two readings (``products``).
``FRONT_READS_EMBEDDING[name]``, ``drug_embedding_copy(model, weight, site)``
    Where a model reads the perturbation embedding (``backbone.pert``) behind a cut while a frozen part reads
    it too: the attention head (``attention_head``, at every cut behind the input: the blocks read the embedding
    through ``task_to_token``) and the summary-statistics branches (``summary_common``, ``summary_interaction``,
    for the arms that train one branch alone). The trained part gets its own copy there; frozen parts keep
    reading the original. Hooks on the backbone do the swap, so the network's code is unchanged.
``training_update(model, optimizer, parameters, batch, lr, first_gene)``
    The one training update of every start: AdamW at the given rate, MSE on genes ``first_gene`` onward,
    the total gradient norm clipped at ``constants.GRADIENT_CLIP``. A nonfinite loss or gradient norm raises
    ``Nonfinite`` before the step.

Nothing here imports torch at module import.
"""
from typing import NamedTuple

from .constants import GRADIENT_CLIP, LEVEL

__all__ = ["CUTS", "FRONT_PARTS", "BRANCH_ARMS", "FRONT_READS_EMBEDDING", "PresenceSites", "PRESENCE_SITES",
           "drug_embedding_copy", "Nonfinite", "training_update"]

BLOCKS = tuple(f"backbone.blocks.{i}." for i in range(8))
# Per model and cut: (front prefixes, trained prefixes). None as the trained part trains every parameter.
CUTS = {
    "deepsets": {"input": ((), None),
                 "phi": (("backbone.phi",), ("backbone.rho", "backbone.pert", "backbone.depth"))},
    "summary": {"input": ((), None),
                "projections": (("backbone.sproj", "backbone.hproj"),
                                ("backbone.common_hidden", "backbone.w0", "backbone.inter_hidden", "backbone.w1",
                                 "backbone.pert", "backbone.depth"))},
    "state": {"input": ((), None),
              "block3": (("backbone.input", "backbone.task_to_token", "backbone.pert") + BLOCKS[:4],
                         BLOCKS[4:] + ("backbone.head", "backbone.depth")),
              "head": (("backbone.input", "backbone.task_to_token", "backbone.pert") + BLOCKS,
                       ("backbone.head", "backbone.depth"))},
}
# Arms that reset one projection of the summary-statistics front.
FRONT_PARTS = {"e_mean": ("backbone.sproj",), "e_var": ("backbone.hproj",), "interaction_e_mean": ("backbone.sproj",)}
# Arms that train one summary-statistics branch: (branch, its prefixes).
BRANCH_ARMS = {"interaction": ("interaction", ("backbone.inter_hidden", "backbone.w1")),
               "interaction_e_mean": ("interaction", ("backbone.inter_hidden", "backbone.w1")),
               "common": ("common", ("backbone.common_hidden", "backbone.w0"))}
# Models whose frozen front also reads the drug embedding at every cut behind the input: where the trained part reads
# its own copy.
FRONT_READS_EMBEDDING = {"state": "attention_head"}


class PresenceSites(NamedTuple):
    """Where a network's cuts are read (module names follow ``scsailor.models.backbones``)."""
    capture: tuple           # modules whose outputs are read: (cells, features) per cell, or (features,) per set
    input_cuts: dict = {}    # cut -> (module, width): the module's input, its first ``width`` features (None: all)
    products: dict = {}      # cut -> (left, right): the elementwise product of two readings


# The presence profile's read sites of the three built networks. ``backbone.pert`` (the drug embedding) is captured
# only for products and dropped after.
PRESENCE_SITES = {
    "deepsets": PresenceSites(
        capture=("backbone.phi.0", "backbone.phi.1", "backbone.phi.2", "backbone.phi", "backbone.rho.0",
                 "backbone.rho.1", "backbone.rho.2", "backbone.pert", "backbone.depth"),
        input_cuts={"pooled": ("backbone.rho.0", 128)}),
    "summary": PresenceSites(
        capture=("backbone.sproj", "backbone.hproj", "backbone.common_hidden", "backbone.inter_hidden",
                 "backbone.pert", "backbone.depth", "backbone.w0", "backbone.w1"),
        products={"inter_x_e": ("backbone.inter_hidden", "backbone.pert")}),
    "state": PresenceSites(
        capture=("backbone.input", *(f"backbone.blocks.{i}" for i in range(8)), "backbone.head.0", "backbone.head.1",
                 "backbone.head.2", "backbone.pert", "backbone.depth"),
        input_cuts={"backbone.head.input": ("backbone.head.0", None)}),
}


def drug_embedding_copy(model, weight, site):
    """A trained part's own copy of the drug embedding, swapped in where that part reads it.

    ``site`` is where the copy is read:
    - ``attention_head``: the drug slice of the head input [pooled cells 768 | depth 128 | drug 128 | size 1];
    - ``summary_common``: the drug slice of the common branch's first-layer input
      [u 640 | depth 128 | drug 128 | size 1];
    - ``summary_interaction``: the drug factor that multiplies the interaction branch's hidden layer before w1.
    Returns a module holding the copy as ``embedding`` (saved as ``embedding.weight``) and the hooks as ``handles``.
    """
    import torch

    copy = torch.nn.Module()
    copy.embedding = torch.nn.Embedding.from_pretrained(weight.detach().clone(), freeze=False)
    seen = {}

    def keep_p(module, args):
        seen["p"] = args[2]   # backbone(x, depth, p, m, level)

    def keep_hidden(module, args, output):
        seen["hidden"] = output

    def replace_slice(module, args):
        features = args[0]
        width = copy.embedding.embedding_dim
        start = features.shape[1] - 1 - width
        drug = copy.embedding(seen["p"]).to(features.dtype)
        return (torch.cat([features[:, :start], drug, features[:, start + width:]], dim=1),)

    def replace_product(module, args):
        return (seen["hidden"] * copy.embedding(seen["p"]).to(seen["hidden"].dtype),)

    backbone = model.backbone
    copy.handles = [backbone.register_forward_pre_hook(keep_p)]
    if site == "attention_head":
        copy.handles.append(backbone.head[0].register_forward_pre_hook(replace_slice))
    elif site == "summary_common":
        copy.handles.append(backbone.common_hidden[0].register_forward_pre_hook(replace_slice))
    elif site == "summary_interaction":
        copy.handles.append(backbone.inter_hidden.register_forward_hook(keep_hidden))
        copy.handles.append(backbone.w1.register_forward_pre_hook(replace_product))
    else:
        raise ValueError(site)
    return copy


class Nonfinite(ArithmeticError):
    """A nonfinite loss or gradient norm; the update was not applied."""

    def __init__(self, what, value):
        super().__init__(f"Nonfinite {what}: {value}")
        self.what, self.value = what, value


def training_update(model, optimizer, parameters, batch, lr, first_gene):
    """One update on ``batch`` (x, depth, p, y tensors); returns the loss and the gradient norm before clipping."""
    import torch
    from torch.nn import functional as F

    x, depth, p, y = batch
    for group in optimizer.param_groups:
        group["lr"] = lr
    optimizer.zero_grad(set_to_none=True)
    loss = F.mse_loss(model(x, depth, p, LEVEL)[:, first_gene:], y[:, first_gene:])
    if not torch.isfinite(loss):
        raise Nonfinite("loss", float(loss))
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(parameters, GRADIENT_CLIP)
    if not torch.isfinite(grad_norm):
        raise Nonfinite("gradient norm", float(grad_norm))
    optimizer.step()
    return loss, grad_norm
