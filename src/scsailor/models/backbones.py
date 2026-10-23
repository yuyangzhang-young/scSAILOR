"""The three built models: the attention model, the DeepSets model and the summary-statistics model.

All three read the carrier rows (expression divided by ``U``, plus log sequencing depth), a learned
perturbation embedding and depth features, and return a profile over 2,000 genes. ``build(name, intercept)``
selects one by its saved key: ``state`` (the attention model, designed after State's published sizes:
8 blocks, width 768, 12 heads; it is not the published State model), ``deepsets`` or ``summary``.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F

from ..carrier import U, depth_bins


def init_new(module):
    for layer in module.modules():
        if isinstance(layer,nn.Linear):
            nn.init.xavier_uniform_(layer.weight)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)
        elif isinstance(layer,nn.Embedding):
            nn.init.normal_(layer.weight,std=0.02)
    for layer in module.modules():
        if isinstance(layer,nn.LayerNorm):
            nn.init.ones_(layer.weight)
            nn.init.zeros_(layer.bias)


def small_last(layer):
    nn.init.normal_(layer.weight,std=math.sqrt(1e-6/layer.in_features))
    if layer.bias is not None:
        nn.init.zeros_(layer.bias)


class RMSNorm(nn.Module):
    def __init__(self,width):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))

    def forward(self,x):
        return F.rms_norm(x,(x.shape[-1],),self.weight)


class AttentionBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.n1=RMSNorm(768)
        self.attn=nn.MultiheadAttention(768,12,dropout=0,batch_first=True)
        self.n2=RMSNorm(768)
        self.ff=nn.Sequential(nn.Linear(768,3072),nn.GELU(),nn.Linear(3072,768))

    def forward(self,x):
        y=self.n1(x)
        x=x+self.attn(y,y,y,need_weights=False,is_causal=False)[0]
        return x+self.ff(self.n2(x))


class Base(nn.Module):
    def __init__(self,intercept,p_count=8):
        super().__init__()
        self.register_buffer('intercept',torch.as_tensor(intercept,dtype=torch.float32))
        self.pert=nn.Embedding(p_count,128)
        self.depth=nn.Sequential(nn.Linear(512,128),nn.GELU(),nn.Linear(128,128))

    def common(self,depth,p,m):
        b=self.depth(torch.log1p(depth.float())/10)
        e=self.pert(p)
        vm=torch.full((len(p),1),math.log2(m)/9,dtype=b.dtype,device=b.device)
        return b,e,vm


class DeepSets(Base):
    def __init__(self,intercept):
        super().__init__(intercept)
        self.phi=nn.Sequential(nn.Linear(2001,256),nn.ReLU(),nn.Linear(256,128),nn.ReLU())
        self.rho=nn.Sequential(nn.Linear(385,256),nn.ReLU(),nn.Linear(256,2000))
        init_new(self)
        small_last(self.rho[-1])

    def forward(self,x,depth,p,m,level):
        b,e,vm=self.common(depth,p,m)
        rows=torch.cat((x/U,(torch.log1p(depth.float())/10).unsqueeze(-1)),dim=-1)
        pooled=self.phi(rows).mean(dim=1)
        return self.intercept+self.rho(torch.cat((pooled,b,e,vm),dim=-1))


class AttentionModel(Base):
    def __init__(self,intercept):
        super().__init__(intercept)
        self.input=nn.Linear(2001,768)
        self.task_to_token=nn.Linear(128,768)
        self.blocks=nn.ModuleList([AttentionBlock() for _ in range(8)])
        self.head=nn.Sequential(nn.Linear(1025,256),nn.GELU(),nn.Linear(256,2000))
        init_new(self)
        small_last(self.head[-1])

    def forward(self,x,depth,p,m,level):
        b,e,vm=self.common(depth,p,m)
        rows=torch.cat((x/U,(torch.log1p(depth.float())/10).unsqueeze(-1)),dim=-1)
        hidden=self.input(rows)+self.task_to_token(e)[:,None,:]
        for block in self.blocks:
            hidden=block(hidden)
        return self.intercept+self.head(torch.cat((hidden.mean(dim=1),b,e,vm),dim=-1))


class SummaryStatistics(Base):
    def __init__(self,intercept):
        super().__init__(intercept)
        self.sproj=nn.Linear(2000,128)
        self.hproj=nn.Linear(2000,128)
        self.qshared=nn.Linear(512,32)
        self.qproj=nn.Linear(2000*32,128)
        self.qdshared=nn.Linear(1024,32)
        self.qdproj=nn.Linear(2000*32,128)
        self.a=nn.Parameter(torch.empty(2001,128))
        self.b=nn.Parameter(torch.empty(2001,128))
        self.cproj=nn.Linear(128,128)
        self.common_hidden=nn.Sequential(nn.Linear(897,256),nn.GELU())
        self.inter_hidden=nn.Sequential(nn.Linear(769,128),nn.GELU())
        self.w0=nn.Linear(256,2000)
        self.w1=nn.Linear(128,2000)
        init_new(self)
        nn.init.xavier_uniform_(self.a)
        nn.init.xavier_uniform_(self.b)
        small_last(self.w0)
        small_last(self.w1)

    def _qd(self,x,depth,m):
        k=512//m
        result=[]
        for sample in range(x.shape[0]):
            atom_depth=depth[sample,::k].detach().cpu().numpy()
            bins=depth_bins(atom_depth)
            row_bins=torch.as_tensor(bins,device=x.device).repeat_interleave(k)
            chunks=[]
            for bin_id in range(int(bins.max())+1):
                chunks.append(torch.sort(x[sample,row_bins==bin_id,:],dim=0).values)
            result.append(torch.cat(chunks,dim=0))
        return torch.stack(result,dim=0)

    def forward(self,x,depth,p,m,level):
        b,e,vm=self.common(depth,p,m)
        zeros=lambda: torch.zeros((len(p),128),device=x.device,dtype=x.dtype)
        if level!='L0':
            s=self.sproj(x.mean(dim=1)/U)
        else:
            s=zeros()
        if level in ('L2','L3','L3d','L4'):
            k=512//m
            h=x.var(dim=1,unbiased=False)*(512/(k*(m-1)))/(U*U)
            hv=self.hproj(h)
        else:
            hv=zeros()
        if level in ('L3','L3d','L4'):
            q=torch.sort(x,dim=1).values.transpose(1,2)/U
            qv=self.qproj(F.gelu(self.qshared(q)).flatten(1))
        else:
            qv=zeros()
        if level in ('L3d','L4'):
            qd=self._qd(x,depth,m).transpose(1,2)/U
            dvec=(torch.log1p(depth.float())/10)[:,None,:].expand(-1,2000,-1)
            qdv=self.qdproj(F.gelu(self.qdshared(torch.cat((qd,dvec),dim=-1))).flatten(1))
        else:
            qdv=zeros()
        if level=='L4':
            v=torch.cat((x/U,(torch.log1p(depth.float())/10).unsqueeze(-1)),dim=-1)
            v=v-v.mean(dim=1,keepdim=True)
            c=((v@self.a)*(v@self.b)).sum(dim=1)/((512//m)*(m-1))
            cv=self.cproj(c)
        else:
            cv=zeros()
        u=torch.cat((s,hv,qv,qdv,cv),dim=-1)
        common=self.w0(self.common_hidden(torch.cat((u,b,e,vm),dim=-1)))
        interaction=self.w1(self.inter_hidden(torch.cat((u,b,vm),dim=-1))*e)
        return self.intercept+common+interaction


def build(name,intercept):
    return {'deepsets':DeepSets,'summary':SummaryStatistics,'state':AttentionModel}[name](intercept)
