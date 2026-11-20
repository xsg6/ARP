"""Exploratory generation intervention and pass@k evaluation (RQ5)."""
import argparse
import json
import math
from pathlib import Path

import numpy as np
from prepare import records


def make_adapter(d, hidden, prototype_dim):
    import torch
    from torch import nn
    class Adapter(nn.Module):
        def __init__(self):
            super().__init__()
            self.encode=nn.Linear(d,prototype_dim,bias=False)
            self.project=nn.Linear(prototype_dim,hidden,bias=False)
            self.gate=nn.Linear(2*hidden,hidden)
        def prototype(self,features):
            return self.project(nn.functional.normalize(self.encode(features),dim=-1).mean(0))
        def inject(self,h,z):
            dtype=h.dtype
            h=h.to(z.dtype)
            gate=self.gate(torch.cat([h,z.expand_as(h)],dim=-1)).sigmoid()
            return nn.functional.layer_norm(h+gate*z,(h.shape[-1],)).to(dtype)
    return Adapter()


def install(model, adapter, z):
    def hook(module,inputs):
        return (adapter.inject(inputs[0],z),*inputs[1:])
    return model.get_output_embeddings().register_forward_pre_hook(hook)


def train(args):
    import torch
    from transformers import AutoModelForCausalLM,AutoTokenizer
    from prepare import repair_prompt
    rows=records(args.manifest)
    cache=np.load(args.cache,allow_pickle=False)
    if [r["id"] for r in rows]!=list(cache["ids"]):
        raise ValueError("cache/order mismatch")
    indices=[i for i,r in enumerate(rows) if r["split"]=="train" and r["correct"]==1]
    if not indices:
        raise ValueError("need positively adjudicated training fixes")
    model=AutoModelForCausalLM.from_pretrained(args.model,revision=args.revision,torch_dtype="auto").to(args.device)
    model.requires_grad_(False).eval()
    tokenizer=AutoTokenizer.from_pretrained(args.model,revision=args.revision)
    features=torch.as_tensor(cache["diff"][indices],device=args.device,dtype=torch.float32)
    torch.manual_seed(args.seed)
    adapter=make_adapter(features.shape[1],model.config.hidden_size,args.prototype_dim).to(args.device)
    opt=torch.optim.AdamW(adapter.parameters(),lr=args.lr)
    rng=np.random.default_rng(args.seed)
    for _ in range(args.epochs):
        for i in rng.permutation(indices):
            row=rows[i]
            prompt=tokenizer.apply_chat_template([{"role":"user","content":repair_prompt(row)}],
                                                tokenize=False,add_generation_prompt=True)
            prompt_ids=tokenizer.encode(prompt,add_special_tokens=False)
            target=tokenizer.encode(row["patch"],add_special_tokens=False)+[tokenizer.eos_token_id]
            ids=torch.tensor([prompt_ids+target],device=args.device)
            if not prompt_ids:
                raise ValueError("empty prompt")
            with torch.no_grad():
                states=model.base_model(input_ids=ids,use_cache=False).last_hidden_state[:,len(prompt_ids)-1:-1]
            z=adapter.prototype(features)
            logits=model.get_output_embeddings()(adapter.inject(states,z)).float()
            loss=torch.nn.functional.cross_entropy(logits[0],torch.tensor(target,device=args.device))
            opt.zero_grad();loss.backward();opt.step()
    with Path(args.out).open("xb") as f:
        torch.save(dict(state=adapter.state_dict(),z=adapter.prototype(features).detach().cpu(),
                        dims=(features.shape[1],model.config.hidden_size,args.prototype_dim),
                        model=args.model,revision=args.revision),f)


def attach(model,args):
    import torch
    ckpt=torch.load(args.adapter,map_location=args.device,weights_only=True)
    if (args.model,args.revision)!=(ckpt["model"],ckpt["revision"]):
        raise ValueError("adapter and decoder revisions differ")
    adapter=make_adapter(*ckpt["dims"]).to(args.device).eval()
    adapter.load_state_dict(ckpt["state"])
    return install(model,adapter,ckpt["z"])


def summarize(path,roster,draws,ks):
    rows=records(path)
    by_bug={(str(r["project"]),str(r["bug"])):[] for r in records(roster) if r["split"]=="test"}
    for row in rows:
        if row["split"]=="test":
            by_bug[(str(row["project"]),str(row["bug"]))].append(row)
    stats=[]
    for key,group in by_bug.items():
        if any(r["correct"] not in (0,1) or r["passed"] not in (0,1) for r in group):
            raise ValueError("need adjudicated outcomes")
        if sum(r["draw_count"] for r in group)>draws:
            raise ValueError("too many draws")
        c=sum(r["draw_count"]*r["correct"] for r in group)
        p=sum(r["draw_count"]*r["passed"] for r in group)
        # Empty generations count as unsuccessful draws. Verify generation completed for every roster bug.
        stats.append(dict(project=key[0],bug=key[1],correct_draws=c,plausible_draws=p,
                          **{f"pass_at_{k}":1-math.comb(draws-c,k)/math.comb(draws,k) for k in ks}))
    if not stats:
        raise ValueError("empty test roster")
    return dict(bugs=len(stats),bugs_with_correct=sum(r["correct_draws"]>0 for r in stats),
                plausible_to_correct=(sum(r["plausible_draws"] for r in stats)/sum(r["correct_draws"] for r in stats)
                                      if sum(r["correct_draws"] for r in stats) else None),
                **{f"pass_at_{k}":np.mean([r[f"pass_at_{k}"] for r in stats]) for k in ks},by_bug=stats)


if __name__=="__main__":
    ap=argparse.ArgumentParser(); sub=ap.add_subparsers(dest="cmd",required=True)
    t=sub.add_parser("train")
    for arg in ["manifest","cache","out"]:
        t.add_argument(arg)
    t.add_argument("--model",default="Qwen/Qwen2.5-Coder-7B-Instruct");t.add_argument("--revision",required=True)
    t.add_argument("--device",default="cuda");t.add_argument("--epochs",type=int,default=1)
    t.add_argument("--lr",type=float,default=.001);t.add_argument("--prototype-dim",type=int,default=128)
    t.add_argument("--seed",type=int,default=11)
    e=sub.add_parser("compare");e.add_argument("baseline");e.add_argument("injected");e.add_argument("roster")
    e.add_argument("out");e.add_argument("--draws",type=int,required=True);e.add_argument("--ks",nargs="+",type=int,default=[1,10])
    a=ap.parse_args()
    if a.cmd=="train":
        train(a)
    else:
        if min(a.ks)<1 or max(a.ks)>a.draws:
            ap.error("k must be between 1 and total draws per defect")
        with Path(a.out).open("x") as f:
            json.dump({s:summarize(getattr(a,s),a.roster,a.draws,a.ks) for s in ["baseline","injected"]},f,indent=2)
