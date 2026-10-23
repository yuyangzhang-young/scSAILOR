"""Scores of natural-task predictions against measured responses.

All averages are weighted over prediction units (one context and perturbation each).

``risk``     total MSE and its two components: the common error (the error's weighted mean within each
             context, shared by all perturbations of that context) and the contrast error (the rest).
             Saved keys: ``total_mse``, ``common_error_mse``, ``difference_error_mse`` (contrast).
``project``  the per-context weighted mean that defines the common error.
``effects``  further read-outs of the predicted change from the context's control profile ``c``:
             mean absolute error, Pearson and Spearman correlation of the change, the share of the 50
             largest true changes that are among the 50 largest predicted changes with the same sign,
             and the cosine perturbation-discrimination score.
"""
import numpy as np
from scipy.stats import rankdata


def project(values,contexts,weights):
    out=np.empty_like(values)
    for context in np.unique(contexts):
        mask=contexts==context;out[mask]=np.average(values[mask],axis=0,weights=weights[mask])
    return out


def risk(error,contexts,weights):
    common=project(error,contexts,weights)
    mse=lambda v:float(np.average(np.mean(v*v,axis=1),weights=weights))
    return {'total_mse':mse(error),'common_error_mse':mse(common),'difference_error_mse':mse(error-common)}


def average_valid(values,weights):
    values=np.asarray(values);good=np.isfinite(values);mass=float(weights[good].sum()/weights.sum())
    return {'value':float(np.average(values[good],weights=weights[good])) if good.any() else None,'valid_weight':mass,'degenerate_weight':1-mass}


def row_correlation(a,b):
    ac=a-a.mean(axis=1,keepdims=True);bc=b-b.mean(axis=1,keepdims=True)
    den=np.sqrt(np.sum(ac*ac,axis=1)*np.sum(bc*bc,axis=1))
    return np.divide(np.sum(ac*bc,axis=1),den,out=np.full(len(a),np.nan),where=den>0)


def effects(pred,truth,c,contexts,codes,w):
    a=pred-c;b=truth-c
    pearson=row_correlation(a,b);spearman=row_correlation(rankdata(a,axis=1),rankdata(b,axis=1))
    top=[]
    for aa,bb in zip(a,b):
        t=np.argsort(-np.abs(bb),kind='stable')[:50];t=t[bb[t]!=0]
        guess=np.argsort(-np.abs(aa),kind='stable')[:50]
        top.append(float(np.mean(np.isin(t,guess)&(np.sign(aa[t])==np.sign(bb[t])))) if len(t) else np.nan)
    pds=np.full(len(a),np.nan);sizes={}
    for context in np.unique(contexts):
        mask=np.flatnonzero(contexts==context);pc=np.unique(codes[mask]);sizes[context]=len(pc)
        if len(pc)<2:continue
        target=np.stack([np.average(b[mask[codes[mask]==p]],axis=0,weights=w[mask[codes[mask]==p]]) for p in pc])
        norms=np.linalg.norm(target,axis=1)
        for p in pc:
            rows=mask[codes[mask]==p];prediction=np.average(a[rows],axis=0,weights=w[rows]);den=np.linalg.norm(prediction)*norms
            sim=np.divide(target@prediction,den,out=np.zeros(len(pc)),where=den>0)
            correct=sim[np.flatnonzero(pc==p)[0]];rank=np.sum(sim>correct)+(np.sum(sim==correct)-1)/2
            pds[rows]=1-rank/(len(pc)-1)
    return {'MAE':float(np.average(np.mean(np.abs(pred-truth),axis=1),weights=w)),
            'Pearson_delta':average_valid(pearson,w),'Spearman_signed_delta':average_valid(spearman,w),
            'direction_top50':average_valid(top,w),'cosine_PDS':average_valid(pds,w),'PDS_candidates':sizes}
