"""Paired validation studies on a shared candidate pool and a fixed return rule."""
import numpy as np
import pandas as pd


def cost_errors(test, training_median):
    observed = test.seconds.to_numpy()
    status = test.get("status", pd.Series("", index=test.index)).fillna("").astype(str)
    timing = np.full(len(test), "unreported", dtype=object)
    timing[status.isin(["passed", "failed", "compile_failed"])] = "completed"
    timing[status.isin(["test_timeout", "compile_timeout"])] = "timeout"
    records = []
    for name, prediction in [("regression", test.cost.to_numpy()),
                             ("training_median", np.full(len(test), training_median))]:
        errors = pd.DataFrame({"project":test.project.to_numpy(), "timing":timing,
                               "residual":observed-prediction})
        for (project, kind), part in errors.groupby(["project","timing"]):
            residual = part.residual
            # For a stopped run, only the time until interruption was observed.
            records.append(dict(project=project, timing=kind, cost_model=name, candidates=len(part),
                mae_seconds=residual.abs().mean() if kind=="completed" else np.nan,
                median_absolute_error_seconds=residual.abs().median() if kind=="completed" else np.nan,
                mean_underprediction_lower_bound_seconds=residual.clip(lower=0).mean()
                if kind=="timeout" else np.nan))
    return pd.DataFrame(records)


def contrasts(records):
    rng = np.random.default_rng(8261)
    estimates = []
    for (budget, rule), group in records.groupby(["budget", "rule"]):
        outcomes = group.pivot(index="group", columns=["representation","cost_model","policy"],
                               values="success")
        if outcomes.isna().any().any():
            raise ValueError("paired arms must contain the same bug roster")
        a = outcomes["last","regression","ratio"]
        b = outcomes["last","regression","joint"]
        c = outcomes["fusion","regression","ratio"]
        d = outcomes["fusion","regression","joint"]
        comparisons = {
            "reward_last": b-a,
            "reward_fusion": d-c,
            "representation_x_reward": (d-c)-(b-a),
            "predicted_vs_constant_cost_ratio": c-outcomes["fusion","training_median","ratio"],
            "predicted_vs_constant_cost_joint": d-outcomes["fusion","training_median","joint"],
        }
        for name, change in comparisons.items():
            delta = change.to_numpy()
            boot = [rng.choice(delta, len(delta), replace=True).mean() for _ in range(2000)]
            lo, hi = np.quantile(boot, [.025,.975])
            estimates.append(dict(budget=budget,rule=rule,contrast=name,bugs=len(delta),
                difference_pp=100*delta.mean(),lo_pp=100*lo,hi_pp=100*hi,
                positive_bug_differences=(delta>0).sum(),negative_bug_differences=(delta<0).sum()))
    return pd.DataFrame(estimates)


def write_studies(test, logs, last_scores, training_median, replay, args):
    # Reuse the two multiprefix arms from the main evaluation. For best-return
    # analyses all arms use the same frozen multiprefix score to select the return.
    baseline = logs[logs.policy.isin(["ratio","joint"])].copy()
    empty = baseline[~baseline.group.isin(test.group)]
    variants = [baseline.assign(representation="fusion", cost_model="regression")]
    last_p, last_u = last_scores[:2]
    for representation, model in [("last","regression"),("fusion","training_median")]:
        alternate = test.assign(return_p=test.p)
        if representation=="last":
            alternate = alternate.assign(p=last_p[test.index], u=last_u[test.index])
        else:
            alternate = alternate.assign(cost=float(training_median))
        records = []
        for _, candidates in alternate.groupby("group"):
            for budget in args.budgets:
                for rule in ("first","best"):
                    for policy in ("ratio","joint"):
                        outcome = replay(candidates, policy, budget, rule, args)
                        records.append(dict(policy=policy,budget=budget,rule=rule,**outcome))
        frame = pd.concat([pd.DataFrame(records), empty], ignore_index=True)
        variants.append(frame.assign(representation=representation, cost_model=model))
    records = pd.concat(variants, ignore_index=True)
    keys = ["representation","cost_model","policy","budget","rule"]
    if records.duplicated(keys+["group"]).any():
        raise ValueError("a bug appeared more than once in a study arm")
    records.to_csv(args.out/"validation_studies_by_bug.csv",index=False)
    summary = records.groupby(keys).agg(bugs=("group","size"),repaired=("success","sum"),
        returned=("returned","sum"),repairable=("repairable","sum"),retained=("retained","sum"),completed=("completed","sum"),
        total_replay_seconds=("seconds","sum"))
    summary["wrong_returns"]=summary.returned-summary.repaired
    summary["pool_without_correct"]=summary.bugs-summary.repairable
    summary["correct_not_validated"]=summary.repairable-summary.retained
    summary["correct_validated_not_returned"]=summary.retained-summary.repaired
    summary["success_rate"]=summary.repaired/summary.bugs
    summary.to_csv(args.out/"validation_studies.csv")
    contrasts(records).to_csv(args.out/"validation_contrasts.csv",index=False)
    cost_errors(test, training_median).to_csv(args.out/"cost_errors_by_project.csv",index=False)
