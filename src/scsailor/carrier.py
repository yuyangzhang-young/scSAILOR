"""Information-level carriers: input sets that keep exactly one level of the control cells' statistics.

From m measured control cells (rows sorted by sequencing depth), ``make_atoms`` builds m rows that keep
only the statistics of one level; ``make`` repeats them to the 512 rows a model receives. Levels:

=====  =======  ======================================================================
Key    Paper    The m rows keep
=====  =======  ======================================================================
L0     -        sequencing depths only (expression set to zero)
L1     Mean     each gene's mean
L2     +Var     each gene's mean and sample variance
L3     +Dist    each gene's marginal distribution (values shuffled across rows)
L3d    +DepCp   each gene's marginal within depth strata
L4     Cells    the cells themselves
=====  =======  ======================================================================

All construction is in float64; ``check_carriers`` verifies the moments and multisets.
"""
import numpy as np

U = np.log(10001.0)
LEVELS = ('L0','L1','L2','L3','L3d','L4')


def depth_bins(sorted_depth):
    m = len(sorted_depth)
    k = max(1,m//32)
    thresholds = np.unique(sorted_depth[np.ceil(np.arange(1,k)*m/k).astype(int)-1])
    thresholds = thresholds[thresholds < sorted_depth[-1]]
    return np.searchsorted(thresholds, sorted_depth, side='left')


def make_atoms(x, depth, level, rng):
    x = np.asarray(x,dtype=np.float64)
    depth = np.asarray(depth,dtype=np.int64)
    m,g = x.shape
    if 512 % m:
        raise ValueError('m must divide N=512')
    if np.all(depth[:-1] <= depth[1:]):
        d = depth
    else:
        order = np.argsort(depth,kind='stable')
        d = depth[order]
        x = x[order]
    if level == 'L0':
        c = np.zeros_like(x)
    elif level == 'L1':
        c = np.broadcast_to(x.mean(axis=0),x.shape).copy()
    elif level == 'L2':
        s = x.mean(axis=0)
        h = x.var(axis=0,ddof=1)
        t = m*s
        full = np.floor(t/U).astype(int)
        row=np.arange(m)[:,None]
        z=np.where(row<full[None,:],U,0.0)
        z+=np.where(row==full[None,:],(t-full*U)[None,:],0.0)
        vmax = np.mean((z-s)**2,axis=0)
        v = (m-1)*h/m
        ratio = np.divide(v,vmax,out=np.zeros_like(v),where=vmax>0)
        if np.any(ratio > 1+1e-8):
            raise ValueError('illegal L2 moment domain')
        c = s + np.sqrt(ratio)*(z-s)
        c=np.take_along_axis(c,np.argsort(rng.random((m,g)),axis=0),axis=0)
    elif level in ('L3','L3d'):
        c = np.empty_like(x)
        bins = np.zeros(m,dtype=np.int8) if level == 'L3' else depth_bins(d)
        for b in np.unique(bins):
            ix = np.flatnonzero(bins == b)
            values = np.sort(x[ix],axis=0)
            c[ix]=np.take_along_axis(values,np.argsort(rng.random((len(ix),g)),axis=0),axis=0)
    elif level == 'L4':
        c = x
    else:
        raise ValueError(level)
    return c,d


def make(x,depth,level,rng):
    c,d = make_atoms(x,depth,level,rng)
    k = 512//len(d)
    return np.repeat(c,k,axis=0).astype(np.float32),np.repeat(d,k)


def check_carriers(x,depth):
    """The specified L1/L2 moment and L3/L3d multiset checks."""
    x = np.asarray(x,dtype=np.float64)
    d = np.sort(depth)
    order = np.argsort(depth,kind='stable')
    source = x[order]
    m = len(depth)
    s = source.mean(0)
    h = source.var(0,ddof=1)
    results = {}
    for level in ('L1','L2','L3','L3d'):
        c,cd = make_atoms(source,d,level,np.random.default_rng(31001))
        if not np.array_equal(cd,d):
            raise AssertionError('depth mismatch')
        if level in ('L1','L2'):
            mean_error = float(np.max(np.abs(c.mean(0)-s)))
            variance_error = float(np.max(np.abs(c.var(0,ddof=1)-(h if level=='L2' else 0))/(1+h)))
            if mean_error > 1e-8 or variance_error > 1e-8:
                raise AssertionError(f'{level} moment recovery: {mean_error}, {variance_error}')
            replicated = np.repeat(c,512//m,axis=0)
            recovered = ((replicated-replicated.mean(0))**2).sum(0)/((512//m)*(m-1))
            if np.max(np.abs(recovered-(h if level=='L2' else 0))/(1+h)) > 1e-8:
                raise AssertionError(f'{level} replicated moment recovery')
            results[level] = {'mean_max_abs':mean_error,'variance_scaled_max_abs':variance_error}
        else:
            bins = np.zeros(m,dtype=np.int8) if level=='L3' else depth_bins(d)
            for b in np.unique(bins):
                ix = np.flatnonzero(bins==b)
                if not np.array_equal(np.sort(c[ix],axis=0),np.sort(source[ix],axis=0)):
                    raise AssertionError(f'{level} multiset mismatch in bin {b}')
            results[level] = {'multiset':'exact','bins':int(len(np.unique(bins)))}
    return results
