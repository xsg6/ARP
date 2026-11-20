"""Frozen-state neural scorer (paper Sections IV-C through IV-F)."""
import copy
import numpy as np
import torch
from torch import nn
from sklearn.metrics import roc_auc_score


class Scorer(nn.Module):
    def __init__(self, mode, edit_dim, blocks, attention_width=64, hidden_width=512, dropout=.2):
        super().__init__()
        self.mode=mode
        self.blocks=blocks
        self.use_edit=mode not in {"probe","small","large","tiny"}
        self.indices={"diff":[], "small":[0], "large":[1], "tiny":[0], "edit_small":[0]}.get(mode,list(range(len(blocks))))
        self.attention=nn.ModuleDict({str(i):nn.Sequential(nn.Linear(blocks[i],attention_width),
                    nn.Tanh(),nn.Linear(attention_width,1,bias=False)) for i in self.indices})
        dim=(edit_dim if self.use_edit else 0)+sum(blocks[i] for i in self.indices)
        self.head=nn.Sequential(nn.Linear(dim,hidden_width),nn.GELU(),nn.Dropout(dropout),nn.Linear(hidden_width,1))

    def forward(self,x,h):
        parts=[x] if self.use_edit else []
        states=h.split(self.blocks,dim=-1)
        for i in self.indices:
            t=states[i]
            if self.mode=="last":
                v=t[:,-1]
            elif self.mode=="mean":
                v=t.mean(1)
            else:
                weights=self.attention[str(i)](t).softmax(1)
                v=(weights*t).sum(1)
            parts.append(v)
        return self.head(torch.cat(parts,dim=1)).flatten()


def train_head(mode, seed, rows, x, h, train, cal, dims, args):
    torch.manual_seed(seed)
    rng=np.random.default_rng(seed)
    device=args.device
    net=Scorer(mode,x.shape[1],dims,args.attention_width,args.hidden_width,args.dropout).to(device)
    opt=torch.optim.AdamW(net.parameters(),lr=args.learning_rate,weight_decay=args.weight_decay)
    y=rows.correct.to_numpy(int)
    ids=np.flatnonzero(train)
    counts=np.bincount(y[ids],minlength=2)
    if min(counts)==0:
        raise ValueError("both labels required in training")
    weights=len(ids)/(2*counts)
    groups=[np.array(g.index) for _,g in rows.loc[train].groupby("group")]
    def score(batch):
        a=torch.as_tensor(x[batch],dtype=torch.float32,device=device)
        b=torch.as_tensor(h[batch],dtype=torch.float32,device=device)
        return net(a,b)
    def predict(indices):
        net.eval()
        with torch.inference_mode():
            return np.concatenate([score(indices[i:i+args.batch_size]).cpu().numpy() for i in range(0,len(indices),args.batch_size)])
    best_auc=-1.; stale=0; best=None
    for epoch in range(args.epochs):
        # Previous-epoch scores are used for hard negative mining within each bug.
        mining=np.zeros(len(rows)); mining[ids]=predict(ids)
        pairs=[]
        for group in groups:
            neg=group[y[group]==0]
            if not len(neg):
                continue
            for pos in group[y[group]==1]:
                hard=neg[np.argsort(-mining[neg],kind="stable")[:min(2,len(neg))]]
                remaining=np.setdiff1d(neg,hard)
                random_neg=rng.choice(remaining,min(4-len(hard),len(remaining)),replace=False)
                pairs.extend((pos,n) for n in np.r_[hard,random_neg])
        pair_array=np.asarray(pairs,dtype=int).reshape(-1,2)
        order=rng.permutation(ids); net.train()
        for offset in range(0,len(order),args.batch_size):
            batch=order[offset:offset+args.batch_size]
            labels=torch.as_tensor(y[batch],device=device,dtype=torch.float32)
            w=torch.as_tensor(weights[y[batch]],device=device,dtype=torch.float32)
            loss=nn.functional.binary_cross_entropy_with_logits(score(batch),labels,weight=w)
            if len(pair_array) and mode!="no_rank":
                subset=pair_array[rng.integers(len(pair_array),size=len(batch))]
                plus,minus=score(subset.ravel()).sigmoid().reshape(-1,2).T
                loss=loss+args.rank_weight*(args.rank_margin-plus+minus).relu().mean()
            opt.zero_grad(); loss.backward(); opt.step()
        valid=np.flatnonzero(cal & rows.passed.to_numpy(bool))
        auc=roc_auc_score(y[valid],predict(valid))
        if auc>best_auc:
            best_auc=auc; stale=0; best=copy.deepcopy(net.state_dict())
        else:
            stale+=1
            if stale>=args.patience:
                break
    net.load_state_dict(best)
    return predict(np.arange(len(rows)))
