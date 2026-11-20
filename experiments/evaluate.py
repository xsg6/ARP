"""Evaluate held-out projects using paired, bug-level comparisons."""
import argparse
import json
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from model import Ranker


def load(manifest, cache):
    rows = pd.read_json(manifest, lines=True, dtype={"id": str, "bug": str, "project": str})
    rows = rows.reset_index(drop=True)
    data = np.load(cache, allow_pickle=False)
    if rows.id.duplicated().any() or list(rows.id) != list(data["ids"]):
        raise ValueError("unique manifest IDs must exactly match cache order")
    if (set(rows[rows.split=="test"].project) & set(rows[rows.split!="test"].project) or
        rows.groupby(["project","bug"]).split.nunique().max()!=1 or set(rows.split)!={"train","cal","test"}):
        raise ValueError("test projects must be held out and a bug cannot span train/cal")
    if not rows.correct.isin([0, 1]).all() or not rows.passed.isin([0, 1]).all() or (rows.correct > rows.passed).any():
        raise ValueError("need adjudicated binary labels; correct implies passed")
    if rows.duplicated(["project", "bug", "patch"]).any():
        raise ValueError("deduplicate identical patches within each bug first")
    numeric = rows[["seconds", "n_tests", "gen_score"]].to_numpy(float)
    if not np.isfinite(numeric).all() or (rows.seconds <= 0).any() or (rows.n_tests < 0).any():
        raise ValueError("invalid runtimes, test counts or generation scores")
    x, h, overhead = data["diff"], data["prefix"], data["seconds"]
    if x.ndim != 2 or h.ndim != 3 or h.shape[1] != 10 or any(len(a) != len(rows) for a in (x,h,overhead)):
        raise ValueError("invalid cache dimensions")
    if any(not np.isfinite(a).all() for a in (x,h,overhead)) or (overhead < 0).any():
        raise ValueError("invalid cache values")
    rows["group"] = list(map(json.dumps, zip(rows.project, rows.bug)))
    return rows, x, h, overhead


def families(rows, x, h, args):
    train = rows.split.eq("train").to_numpy()
    if args.train_on == "plausible":
        train &= rows.passed.to_numpy(bool)
    cal = rows.split.eq("cal").to_numpy()
    out = {}
    modes=["diff","probe","last","mean","fusion","no_rank"]
    cache=np.load(args.cache,allow_pickle=False)
    if args.backend=="paper":
        from scorer import train_head
        modes.extend(["small","large","edit_small"])
        if "tiny" in cache:
            modes.append("tiny")
    for mode in modes:
        heads = []
        for seed in args.seeds:
            if args.backend=="paper":
                hidden=cache["tiny"] if mode=="tiny" else h
                dims=[hidden.shape[-1]] if mode=="tiny" else list(cache["widths"].astype(int))
                heads.append(train_head(mode,seed,rows,x,hidden,train,cal,dims,args))
                continue
            model = Ranker("fusion" if mode == "no_rank" else mode, seed,
                           rank_weight=0 if mode == "no_rank" else 1)
            model.fit(x[train], h[train], rows.correct.to_numpy(int)[train], rows.group.to_numpy()[train])
            heads.append(model.logits(x, h))
        raw = np.array(heads)
        mean = raw.mean(0)
        scaler = LogisticRegression(C=1.0).fit(mean[cal,None], rows.correct[cal])
        # Platt scaling is fitted only on calibration projects, including for weak ablations.
        probs = expit(scaler.intercept_[0] + scaler.coef_[0,0]*raw)
        out[mode] = (scaler.predict_proba(mean[:,None])[:,1], probs.std(0), expit(mean), raw)
    return out


def rank_rows(g, score, name):
    order = np.argsort(-score[g.index], kind="stable")
    correct = g.correct.to_numpy()[order]
    hit = np.flatnonzero(correct)
    return dict(group=g.group.iloc[0], project=g.project.iloc[0], method=name,
                mrr=1/(hit[0]+1) if len(hit) else 0., top1=int(correct[0]),
                top5=int(correct[:5].any()), repairable=int(correct.any()))


def cost_features(rows):
    return np.log1p(np.column_stack([rows.patch.str.len(), rows.original.str.len(),
                                     (rows.patch.str.len()-rows.original.str.len()).abs(), rows.n_tests]))


def replay(g, policy, budget, rule, args):
    p = g.p.to_numpy()
    u = g.u.to_numpy()
    cost = g.cost.to_numpy()
    actual = g.seconds.to_numpy()
    passed = g.passed.to_numpy(bool)
    correct = g.correct.to_numpy(bool)
    # Charge the complete feature pass once, including unselected candidates.
    needs_score = policy not in {"exhaustive", "cost", "generation"} or rule == "best"
    elapsed = g.overhead.sum() if needs_score else 0.
    left = np.ones(len(g), bool)
    chosen = []
    scale = 1.
    while left.any() and elapsed < budget:
        predicted = np.maximum(cost * scale, 1e-6)
        priority = p.copy()
        if policy == "exhaustive":
            priority = -np.arange(len(g), dtype=float)
        elif policy == "generation":
            priority = g.gen_score.to_numpy()
        elif policy == "cost":
            priority = 1 / predicted
        elif policy in {"ratio", "joint", "entropy", "uncalibrated"}:
            bonus = u if policy == "joint" else 0.
            if policy == "entropy":
                t = np.clip(p, 1e-12, 1-1e-12)
                bonus = -(t*np.log(t)+(1-t)*np.log1p(-t))
            prior = g.raw_p.to_numpy() if policy == "uncalibrated" else p
            if policy == "uncalibrated":
                bonus = g.raw_u.to_numpy()
            priority = (prior + args.reward*bonus) / predicted
        elif policy == "no_cost":
            priority = p + args.reward*u
        eligible = left.copy()
        if policy == "fixed":
            eligible &= p >= args.threshold
            if len(chosen) >= args.k:
                break
        if policy in {"cost", "ratio", "joint", "entropy", "uncalibrated", "no_cost"}:
            if policy != "no_cost":
                eligible &= predicted <= budget-elapsed
        if not eligible.any():
            break
        i = int(np.argmax(np.where(eligible, priority, -np.inf)))
        left[i] = False
        remaining = budget-elapsed
        if actual[i] > remaining:
            return finish(g, chosen, rule, budget, correct, passed, p, i)
        elapsed += actual[i]
        chosen.append(i)
        scale = (1 + actual[chosen].sum()) / (1 + cost[chosen].sum())
        if rule == "first" and passed[i]:
            break
    return finish(g, chosen, rule, min(elapsed,budget), correct, passed, p, None)


def finish(g, chosen, rule, seconds, correct, passed, p, interrupted):
    plausible = [i for i in chosen if passed[i]]
    selected = None
    if plausible:
        selected = plausible[0] if rule in {"first", "all"} else max(plausible, key=lambda i:p[i])
    return dict(group=g.group.iloc[0], project=g.project.iloc[0], seconds=seconds,
                attempts=len(chosen)+int(interrupted is not None), completed=len(chosen),
                repairable=int(correct.any()), retained=int(correct[chosen].any()) if chosen else 0,
                returned=int(selected is not None), success=int(correct[selected]) if selected is not None else 0,
                order=json.dumps(g.id.iloc[chosen].tolist()), returned_id=g.id.iloc[selected] if selected is not None else "",
                interrupted_id=g.id.iloc[interrupted] if interrupted is not None else "")


def paired(wide, a, b, rng):
    delta = (wide[a]-wide[b]).to_numpy()
    boot = [rng.choice(delta, len(delta)).mean() for _ in range(2000)]
    return dict(a=a,b=b,n=len(delta),difference=delta.mean(),lo=np.quantile(boot,.025),
                hi=np.quantile(boot,.975),better=int((delta>0).sum()),worse=int((delta<0).sum()))


def select_fixed(rows, scores, overhead):
    """Choose the cheapest development-set operating point retaining at least 95%."""
    dev=rows[rows.split=="cal"]
    ranked=[]
    for _,g in dev.groupby("group"):
        order=np.argsort(-scores[g.index],kind="stable")
        idx=g.index.to_numpy()[order]
        ranked.append((scores[idx],rows.correct.to_numpy()[idx],rows.seconds.to_numpy()[idx]))
    denominator=sum(bool(c.any()) for _,c,_ in ranked)
    if not denominator:
        raise ValueError("calibration pool has no repairable bugs")
    thresholds=np.unique(np.r_[0.,np.quantile(scores[dev.index],np.linspace(0,1,21))])
    best=None
    for k in range(1,max(len(p) for p,_,_ in ranked)+1):
        for threshold in thresholds:
            duration=overhead[dev.index].sum()
            kept=0
            for p,c,t in ranked:
                use=p[:k]>=threshold
                kept+=bool(c[:k][use].any())
                duration+=t[:k][use].sum()
            if kept/denominator>=.95:
                key=(duration,k,float(threshold))
                if best is None or key<best:
                    best=key
    return best[1:]


def auc_intervals(plausible, scores, rng):
    # For each draw, give all candidates in the same defect equal replicate multiplicity.
    groups, inverse=np.unique(plausible.group.to_numpy(),return_inverse=True)
    labels=plausible.correct.to_numpy()
    if len(np.unique(labels))<2:
        return []
    draws={m:[] for m in scores}
    predictions={m:v[plausible.index] for m,v in scores.items()}
    for _ in range(1000):
        weights=np.bincount(rng.integers(len(groups),size=len(groups)),minlength=len(groups))[inverse]
        if len(np.unique(labels[weights>0]))<2:
            continue
        for m,values in predictions.items():
            draws[m].append(roc_auc_score(labels,values,sample_weight=weights))
    output=[]
    for m,estimates in draws.items():
        if estimates:
            lo,hi=np.quantile(estimates,[.025,.975])
            output.append(dict(method=m,lo=lo,hi=hi,valid_draws=len(estimates)))
    return output


def assess(rows,x,h,overhead,args):
    out = args.out
    out.mkdir(parents=True,exist_ok=False)
    import time
    start=time.perf_counter()
    scores = families(rows,x,h,args)
    fit_and_score_seconds=time.perf_counter()-start
    if args.select_fixed:
        args.k,args.threshold=select_fixed(rows,scores["fusion"][0],overhead)
    test = rows.loc[rows.split == "test"].copy()
    roster = test[["project","bug","group"]].drop_duplicates().copy()
    if args.roster:
        roster = pd.read_json(args.roster,lines=True,dtype={"project":str,"bug":str})
        roster = roster[roster.split=="test"].copy()
        roster["group"] = list(map(json.dumps,zip(roster.project,roster.bug)))
        if roster.group.duplicated().any() or not set(test.group).issubset(roster.group):
            raise ValueError("invalid or incomplete test bug roster")
    missing = roster[~roster.group.isin(test.group)]

    train = rows.loc[rows.split == "train"]
    costs = make_pipeline(StandardScaler(), Ridge(alpha=1)).fit(cost_features(train), np.log(train.seconds))
    test["cost"] = np.exp(costs.predict(cost_features(test)))
    idx = test.index
    p,u,raw,heads = scores["fusion"]
    test = test.assign(p=p[idx],u=u[idx],raw_p=raw[idx],raw_u=expit(heads[:,idx]).std(0),overhead=overhead[idx])
    test[["id","project","bug","p","u","raw_p","cost","seconds","overhead"]].to_csv(out/"predictions.csv",index=False)
    ranks, quality, reliability = [], [], []
    rank_scores = {k:v[0] for k,v in scores.items()}
    rank_scores["generation"] = rows.gen_score.to_numpy()
    if "ods_score" in rows:
        if not np.isfinite(rows.ods_score).all():
            raise ValueError("incomplete ODS scores")
        rank_scores["ODS"]=rows.ods_score.to_numpy()
    for j, seed in enumerate(args.seeds):
        rank_scores[f"fusion_seed_{seed}"] = heads[j]
    for name,score in rank_scores.items():
        ranks.extend(rank_rows(g,score,name) for _,g in test.groupby("group"))
        for bug in missing.itertuples():
            ranks.append(dict(group=bug.group,project=bug.project,method=name,mrr=0.,top1=0,top5=0,repairable=0))
        plausible = test[test.passed == 1]
        auc = roc_auc_score(plausible.correct,score[plausible.index]) if plausible.correct.nunique()==2 else None
        quality.append(dict(method=name,plausible_auc=auc))
    for name,prob in [("raw",raw[idx]),("calibrated",p[idx])]:
        bins = np.minimum((prob*10).astype(int),9)
        for j in np.unique(bins):
            mask = bins == j
            reliability.append(dict(model=name,bin=j,count=mask.sum(),confidence=prob[mask].mean(),
                                    observed=test.correct.to_numpy()[mask].mean()))
        quality.append(dict(method=name,brier=np.mean((prob-test.correct.to_numpy())**2)))
    if not args.skip_auc_ci:
        pd.DataFrame(auc_intervals(test[test.passed==1],{k:v[0] for k,v in scores.items()},np.random.default_rng(731))).to_csv(out/"auc_ci.csv",index=False)
    pd.DataFrame(quality).to_csv(out/"quality.csv",index=False)
    pd.DataFrame(reliability).to_csv(out/"reliability.csv",index=False)
    rank = pd.DataFrame(ranks)
    rank.to_csv(out/"rank_by_bug.csv",index=False)
    repairable=rank[rank.repairable==1].copy()
    repairable["first_rank"]=1/repairable.mrr
    repairable.groupby("method").agg(mrr=("mrr","mean"),top1=("top1","mean"),top5=("top5","mean"),
        median_first_rank=("first_rank","median")).to_csv(out/"rank_repairable.csv")
    random_rng=np.random.default_rng(101)
    random_reps=[]
    for _,g in test.groupby("group"):
        hits=g.correct.to_numpy(bool)
        if hits.any():
            copies=np.array([random_rng.permutation(hits) for _ in range(1000)])
            first=copies.argmax(axis=1)+1
            random_reps.append(first)
    if random_reps:
        first=np.stack(random_reps)
        pd.Series(dict(top1=(first==1).mean(),top5=(first<=5).mean(),mrr=(1/first).mean(),
                       median_first_rank=np.median(first,axis=0).mean())).to_csv(out/"random_repairable.csv")
    legacy=[]
    with np.load(args.cache,allow_pickle=False) as cache:
        ds=cache["decoder_seconds"] if "decoder_seconds" in cache else None
        es=cache["edit_seconds"] if "edit_seconds" in cache else None
    for _,g in test.groupby("group"):
        for method,order in [("exhaustive",list(g.index)),("generation_top20",list(g.nlargest(20,"gen_score").index))]:
            pick=test.loc[order]
            legacy.append(dict(group=g.group.iloc[0],method=method,retained=int(pick.correct.any()),
                               repairable=int(g.correct.any()),count=len(pick),seconds=pick.seconds.sum()))
        for method in scores:
            prob=scores[method][0][g.index]
            order=np.argsort(-prob,kind="stable")[:args.k]
            pick=g.iloc[order[prob[order]>=args.threshold]]
            cost=g.overhead.sum()
            if ds is not None and es is not None:
                use_edit=method not in {"small","large","tiny","probe"}
                models={"small":[0],"large":[1],"tiny":[2],"edit_small":[0],"diff":[]}.get(method,[0,1])
                cost=(es[g.index].sum() if use_edit else 0)+sum(ds[m,g.index].sum() for m in models)
            legacy.append(dict(group=g.group.iloc[0],method=method,retained=int(pick.correct.any()),
                               repairable=int(g.correct.any()),count=len(pick),seconds=pick.seconds.sum()+cost))
    legacy=pd.DataFrame(legacy)
    legacy.to_csv(out/"fixed_k_cost_by_bug.csv",index=False)
    table=legacy.groupby("method")[["retained","repairable","count","seconds"]].sum()
    table["retention"]=table.retained/table.repairable.replace(0,np.nan)
    table["time_saving"]=1-table.seconds/table.loc["exhaustive","seconds"]
    table.to_csv(out/"fixed_k_cost.csv")
    rank.groupby("method")[["mrr","top1","top5"]].mean().to_csv(out/"rank_all_bugs.csv")
    decisions = []
    policies = ["generation","fixed","score","cost","ratio","joint","entropy","no_cost","uncalibrated"]
    for _,g in test.groupby("group"):
        decisions.append(dict(policy="exhaustive",budget=np.inf,rule="all",
                              **replay(g,"exhaustive",np.inf,"all",args)))
        for policy in policies:
            for budget in args.budgets:
                for rule in ("first","best"):
                    decisions.append(dict(policy=policy,budget=budget,rule=rule,
                                          **replay(g,policy,budget,rule,args)))
    settings={(d["policy"],d["budget"],d["rule"]) for d in decisions}
    for bug in missing.itertuples():
        for policy,budget,rule in sorted(settings):
            decisions.append(dict(policy=policy,budget=budget,rule=rule,group=bug.group,project=bug.project,
                seconds=0.,attempts=0,completed=0,repairable=0,retained=0,returned=0,success=0,order="[]",
                returned_id="",interrupted_id=""))
    logs=pd.DataFrame(decisions)
    logs.to_csv(out/"schedule_by_bug.csv",index=False)
    summary=logs.groupby(["policy","budget","rule"]).agg(
        bugs=("group","size"), repaired=("success","sum"), returned=("returned","sum"),
        retained=("retained","sum"), repairable=("repairable","sum"), attempts=("attempts","sum"),
        seconds=("seconds","sum"))
    summary["success_rate"]=summary.repaired/summary.bugs
    summary["return_precision"]=summary.repaired/summary.returned.replace(0,np.nan)
    summary["return_coverage"]=summary.returned/summary.bugs
    summary["wrong_returns"]=summary.returned-summary.repaired
    summary["retention"]=summary.retained/summary.repairable.replace(0,np.nan)
    summary.to_csv(out/"schedule.csv")
    rng=np.random.default_rng(813)
    cis=[]
    for metric in ["mrr","top1","top5"]:
        wide=rank.pivot(index="group",columns="method",values=metric)
        for baseline in ["probe","last","diff","mean","no_rank","generation"]:
            cis.append(dict(metric=metric,**paired(wide,"fusion",baseline,rng)))
    for (budget,rule),g in logs[logs.policy!="exhaustive"].groupby(["budget","rule"]):
        for metric in ["success","retained"]:
            if metric=="retained":
                g_eval=g[g.repairable==1]
            else:
                g_eval=g
            if len(g_eval):
                wide=g_eval.pivot(index="group",columns="policy",values=metric)
                cis.append(dict(metric=metric,budget=budget,rule=rule,**paired(wide,"joint","ratio",rng)))
    pd.DataFrame(cis).to_csv(out/"paired.csv",index=False)
    def digest(path):
        with open(path, "rb") as stream:
            sha=hashlib.sha256()
            for block in iter(lambda:stream.read(1024*1024),b""):
                sha.update(block)
        return sha.hexdigest()
    hashes={key:digest(path) for key,path in [("manifest",args.manifest),("cache",args.cache),("roster",args.roster)] if path}
    (out/"settings.json").write_text(json.dumps({**vars(args),"out":str(out),
        "sha256":hashes,"fit_and_score_seconds":fit_and_score_seconds,"cost_mae_seconds":float((test.cost-test.seconds).abs().mean())},default=str,indent=2))


if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("manifest"); ap.add_argument("cache"); ap.add_argument("out",type=Path)
    ap.add_argument("--roster",help="all bugs, including those with no candidates; JSONL project, bug, split")
    ap.add_argument("--seeds",nargs="+",type=int,default=[11,23,37,53,71])
    ap.add_argument("--budgets",nargs="+",type=float,required=True)
    ap.add_argument("--train-on",choices=["all","plausible"],default="plausible")
    ap.add_argument("--select-fixed",action="store_true",help="select K and threshold on cal to retain at least 95%")
    ap.add_argument("--skip-auc-ci",action="store_true",help="omit expensive cluster bootstrap for development runs")
    ap.add_argument("--backend",choices=["paper","linear"],default="paper")
    ap.add_argument("--device",default="cpu")
    ap.add_argument("--epochs",type=int,default=100)
    ap.add_argument("--reward",type=float,required=True); ap.add_argument("--threshold",type=float,required=True)
    ap.add_argument("--k",type=int,required=True)
    args=ap.parse_args()
    if min(args.budgets)<=0 or len(set(args.seeds))<2 or args.k<1 or not 0<=args.threshold<=1 or args.reward<0:
        ap.error("invalid budgets, ensemble, threshold, reward or K")
    assess(*load(args.manifest,args.cache),args)
