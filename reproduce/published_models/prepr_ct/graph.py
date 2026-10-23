"""The gene graph of one input set, computed in float64.

Nodes are all panel genes, with each gene's mean and sample variance across the set's cells as features.
Edges join genes whose absolute correlation is at least the 99th percentile of all absolute correlations,
between genes with nonzero variance, without self-loops. This re-expresses PrePR-CT's co-expression graph
in our own code.
"""
import torch


def graph(atoms):
    x = atoms.to(torch.float64)
    mean = torch.mean(x, dim=0)
    variance = torch.var(x, dim=0, correction=1)
    centered = x-mean
    covariance = centered.T @ centered/(len(x)-1)
    scale = covariance.diag().sqrt()
    correlation = covariance/scale[:, None]/scale[None, :]
    correlation = torch.where(torch.isnan(correlation), 0., correlation).clamp(-1., 1.)
    absolute = correlation.abs()
    threshold = torch.quantile(absolute.reshape(-1), .99, interpolation='linear')
    valid = variance > 0
    keep = (absolute >= threshold) & valid[:, None] & valid[None, :]
    keep.fill_diagonal_(False)
    edges = keep.nonzero().T.contiguous()
    features = torch.stack((mean, variance), dim=1).to(torch.float32)
    return features, edges, threshold
