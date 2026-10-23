"""The model contract: what the assay calls on a network, with shapes and callers.

The assay never builds a network itself (``reproduce/study.py`` builds it from ``scsailor.models``); every step receives
the model as an argument. B is the number of bags; one bag is 512 rows by 2,000 genes
(``constants.N_ROWS``, ``N_GENES``); P is the number of perturbations.

``model(x, depth, p, level)``, with ``canonicalize=True``
    x B x 512 x 2000 float32, depth B x 512 int64, p B int64, level in ``design.LEVELS``
    -> raw prediction B x 2000, before clipping. Called by ``train.train`` (B = 8).
``model(x, depth, p, level, canonicalize=False)``
    The same, for rows already in canonical order. Called by ``replay.replay_bag`` (up to 8
    perturbations of one bag per call) and ``probe.readout`` (B = 1, p = 0).
``model.canonical_rows(x, depth)``
    -> ``(x, depth)``, same shapes, rows stably sorted by depth. Called by replay and the probe.
``model.anchor(x, p, level)``
    Canonical rows -> B x 2000: ``c_train`` at L0, otherwise the row mean, plus ``r_pool[p]``. This is the
    baseline the network's output is added to. Replay saves it as ``anchor_raw``; the probe saves it as
    ``anchor_raw`` in ``PROBE_FEATURES.npz`` and ``LOCAL_PAIRS.h5``. The method keeps the name
    ``anchor``; the assay's variables holding its output are called ``baseline``.
``model.r_pool``
    Buffer P x 2000 (the generator's ``r``); replay takes P from its length.
``READOUT_MODULES[name]``
    Submodules whose outputs the probe reads through forward hooks: outputs with three dimensions
    (bag x row x feature) are averaged over rows, then all are concatenated in the listed order.
    ``state`` (the attention network): ``backbone.blocks.7`` -> B x 512 x 768; ``deepsets``:
    ``backbone.phi`` -> B x 512 x 128; ``summary``: ``backbone.common_hidden`` -> B x 256 and
    ``backbone.inter_hidden`` -> B x 128.
``parameters()``, ``train()``, ``state_dict()``, ``load_state_dict()``, ``eval()``
    Standard torch modules. Used by ``train``, and by ``load_checkpoint`` in ``reproduce/study.py``, which returns the model
    in eval mode as replay and the probe expect.

The built models pass all 512 carrier rows to their backbone, together with ``m = 256``, the number
of distinct cells (each written twice); no input rows are cut
(``scsailor.models.model.RawModel.forward``).
"""
READOUT_MODULES = {"state": ("backbone.blocks.7",), "deepsets": ("backbone.phi",),
                   "summary": ("backbone.common_hidden", "backbone.inter_hidden")}

__all__ = ["READOUT_MODULES"]
