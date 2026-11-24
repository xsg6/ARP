"""Produce project breakdowns, budget curves and calibration plots from a run."""
import argparse
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def report(run):
    ranking=pd.read_csv(run/"rank_by_bug.csv")
    ranking.groupby(["project","method"])[["top1","top5","mrr"]].mean().to_csv(run/"ranking_by_project.csv")
    log=pd.read_csv(run/"schedule_by_bug.csv")
    log=log[log.policy!="exhaustive"]
    cols=["project","budget","rule","policy"]
    project=log.groupby(cols).agg(bugs=("group","size"),repaired=("success","sum"),returned=("returned","sum"),
                                  kept=("retained","sum"),repairable=("repairable","sum"),seconds=("seconds","sum"))
    project["success_rate"]=project.repaired/project.bugs
    project["retention"]=project.kept/project.repairable.replace(0,np.nan)
    project.to_csv(run/"schedule_by_project.csv")
    pair=log[log.policy.isin(["joint","ratio"])].pivot(index=["project","group","budget","rule"],
                                                       columns="policy",values="success")
    pair.loc[pair.joint!=pair.ratio].to_csv(run/"changed_repairs.csv")
    summary=pd.read_csv(run/"schedule.csv")
    fig,axes=plt.subplots(1,2,figsize=(10,4),sharey=True)
    for ax,rule in zip(axes,["first","best"]):
        for name in ["fixed","score","cost","ratio","joint"]:
            part=summary[(summary.policy==name)&(summary.rule==rule)].sort_values("budget")
            ax.plot(part.budget,part.success_rate,marker="o",label=name)
        ax.set(xlabel="Per-bug budget (s)",title=rule+" return",ylim=(-.02,1.02))
    axes[0].set_ylabel("Correct returns / all test bugs")
    axes[-1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(run/"budget_curve.pdf"); plt.close(fig)
    bins=pd.read_csv(run/"reliability.csv")
    fig,ax=plt.subplots(figsize=(4.5,4))
    ax.plot([0,1],[0,1],color="gray",linestyle="--")
    for name,g in bins.groupby("model"):
        ax.plot(g.confidence,g.observed,marker="o",label=name)
    ax.set(xlim=(0,1),ylim=(0,1),xlabel="Mean predicted probability",ylabel="Observed correctness")
    ax.legend(); fig.tight_layout(); fig.savefig(run/"calibration.pdf"); plt.close(fig)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("run",type=Path)
    report(parser.parse_args().run)
