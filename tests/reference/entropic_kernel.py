"""Full-support entropic transition direction + exact finite-population step.

This is the recommended soft-probability variant: no endpoint is removed merely
because its gain is negative. Legal support is fixed before gain calibration.
Only when no positive first-order direction exists is the whole update frozen.
COUNT units, fixed positive diagonal W, independent DISJOINT blocks.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence
import numpy as np
from numpy.typing import ArrayLike
from scipy.special import logsumexp
from transition_kernel import BlockSupport, BlockTransition, KernelResult, complete_supports


@dataclass
class EntropicKernelResult(KernelResult):
    beta: float
    max_directional_gain: float
    required_directional_gain: float


def construct_entropic_kernel(
    current: Sequence[int],
    features: ArrayLike,
    target: ArrayLike,
    weights: ArrayLike,
    supports: Sequence[BlockSupport] | None = None,
    stay_probability: float = 0.9,
    progress_fraction: float = 0.5,
    damping: float = 1.0,
    numerical_tol: float = 1e-12,
) -> EntropicKernelResult:
    """Construct the recommended no-rejection kernel.

    The reference puts stay_probability on the current block. Positive mobility
    weights distribute the remaining mass over other legal block outcomes.
    progress_fraction is alpha in (0,1): require D >= alpha*sum(max gain per block).
    A monotone scalar solve chooses beta before any table is sampled.
    Floats are not interval-certified; tiny gains may be treated as numerical zero.
    """
    ss=np.asarray(current)
    if ss.ndim!=1 or not np.issubdtype(ss.dtype,np.integer) or len(ss)==0:
        raise ValueError('current must be a nonempty integer vector')
    s=ss.astype(np.int64,copy=True)
    a=np.asarray(features,dtype=float); y=np.asarray(target,dtype=float); w=np.asarray(weights,dtype=float)
    if a.ndim!=2 or min(a.shape)<1:
        raise ValueError('features must be a nonempty state-by-query matrix')
    M,Q=a.shape; N=len(s)
    if y.shape!=(Q,) or w.shape!=(Q,) or np.any(s<0) or np.any(s>=M):
        raise ValueError('incompatible shapes or invalid state ids')
    if not all(np.isfinite(v).all() for v in (a,y,w)) or np.any(w<=0):
        raise ValueError('finite inputs and positive weights required')
    if not (0<stay_probability<1 and 0<progress_fraction<1 and 0<damping<=1):
        raise ValueError('stay_probability and progress_fraction in (0,1), damping in (0,1]')
    if not np.isfinite(numerical_tol) or numerical_tol<0:
        raise ValueError('numerical_tol must be finite and nonnegative')
    if supports is None: supports=complete_supports(N,M)
    if sorted(i for b in supports for i in b.rows)!=list(range(N)) or any(not b.rows for b in supports):
        raise ValueError('supports must partition the row indices')
    e=y-a[s].sum(axis=0); old=float(np.dot(e*w,e)/2)
    blocks=[]; deltas=[]; references=[]
    for b in supports:
        src=tuple(int(s[i]) for i in b.rows)
        if len(set(b.outcomes))!=len(b.outcomes):
            raise ValueError('aggregate duplicate outcomes before calling')
        mob=b.mobility if b.mobility is not None else (1.0,)*len(b.outcomes)
        if len(mob)!=len(b.outcomes): raise ValueError('invalid mobility shape')
        dest=[]; cs=[]
        for u,c in zip(b.outcomes,mob):
            if len(u)!=len(src) or any(not isinstance(v,(int,np.integer)) or v<0 or v>=M for v in u):
                raise ValueError('invalid block outcome')
            if not np.isfinite(c) or c<0: raise ValueError('invalid mobility')
            if u!=src and c>0: dest.append(u);cs.append(float(c))
        outcomes=(src,)+tuple(dest)
        d=np.zeros((len(outcomes),Q))
        for j,u in enumerate(outcomes[1:],start=1): d[j]=a[list(u)].sum(axis=0)-a[list(src)].sum(axis=0)
        g=d@(w*e)-np.sum(d*d*w,axis=1)/2
        if dest:
            R=np.r_[stay_probability,(1-stay_probability)*np.asarray(cs)/sum(cs)]
        else: R=np.ones(1)
        blocks.append(BlockTransition(b.rows,outcomes,np.zeros(len(outcomes)),g,np.zeros(len(outcomes))))
        deltas.append(d); references.append(R)
    M_gain=float(sum(max(float(np.max(b.gains)),0.0) for b in blocks))
    requirement=progress_fraction*M_gain
    scale=max(1.0,old)

    def tilted(beta):
        ps=[]; D=0.0
        for b,R in zip(blocks,references):
            # Shift by maximum gain before multiplying beta to avoid positive overflow.
            logits=np.log(R)+beta*(b.gains-np.max(b.gains))
            P=np.exp(logits-logsumexp(logits))
            ps.append(P);D+=float(P@b.gains)
        return ps,D

    if M_gain<=numerical_tol*scale:
        beta=0.0; h=0.0;D=0.0;C=0.0
        for b in blocks: b.probabilities[0]=1.0
    else:
        ps,D=tilted(0.0)
        if D>=requirement:
            beta=0.0
        else:
            lo=0.0; hi=1.0/max(M_gain,1e-300)
            for _ in range(100):
                ps,D=tilted(hi)
                if D>=requirement: break
                hi*=2.0
            else: raise FloatingPointError('could not bracket beta; check numerical conditioning')
            for _ in range(80):
                mid=(lo+hi)/2.0
                _,Dm=tilted(mid)
                if Dm>=requirement: hi=mid
                else: lo=mid
            beta=hi; ps,D=tilted(beta)
        if D<=0: raise FloatingPointError('direction gain must be strictly positive')
        vs=[]; bmax=0.0
        for b,P,d in zip(blocks,ps,deltas):
            b.rates=P.copy();b.rates[0]=0.0
            bmax=max(bmax,float(b.rates.sum()));vs.append(b.rates@d)
        v=np.stack(vs);vsum=v.sum(axis=0)
        C=float((np.dot(vsum*w,vsum)-np.sum(v*v*w))/2)
        feasible=1.0/bmax
        h=damping*(min(feasible,D/(2*C)) if C>0 else feasible)
        for b in blocks:
            b.probabilities=h*b.rates
            b.probabilities[0]=1-float(b.probabilities[1:].sum())
            if b.probabilities[0]<-1e-12: raise FloatingPointError('invalid step')
            if b.probabilities[0]<0:
                b.probabilities[0]=0
                b.probabilities/=b.probabilities.sum()
    K=np.zeros((N,M))
    for b in blocks:
        for u,p in zip(b.outcomes,b.probabilities):
            for i,x in zip(b.rows,u): K[i,x]+=p
    return EntropicKernelResult(blocks,K,old,D,C,h,old-h*D+h*h*C,old-h*D/2,beta,M_gain,requirement)
