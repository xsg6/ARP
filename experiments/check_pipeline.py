"""Integration check with generated numerical fixtures, never thesis evidence."""
import json
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from evaluate import load, replay
from types import SimpleNamespace


def main(paper=False, destination=None):
    rng=np.random.default_rng(42)
    rows=[]; x=[]; h=[]
    for part in ("train","cal","test"):
        for bug in range(9):
            for j in range(4):
                y=int(j==bug%4 and bug!=8)
                rows.append(dict(project=part,split=part,bug=str(bug),id=f"{part}-{bug}-{j}",
                                 original="x",patch=f"patch{j}",gen_score=-j,n_tests=4,
                                 seconds=1+j/4,passed=int(y or j==1),correct=y))
                x.append(rng.normal(y,.6,4))
                h.append(rng.normal(y,.6,(10,5)))
    from contextlib import nullcontext
    if destination is not None:
        destination.mkdir(parents=True, exist_ok=False)
    with (TemporaryDirectory() if destination is None else nullcontext(destination)) as tmp:
        tmp=Path(tmp); manifest=tmp/"fixture.jsonl"; cache=tmp/"fixture.npz"; out=tmp/"out"
        validation=tmp/"measured.jsonl"
        decisions=tmp/"decisions.csv"
        validation.write_text("\n".join(json.dumps({**r, "correct":None if r["passed"] else 0}) for r in rows))
        decisions.write_text("id,correct\n"+"".join(f"{r['id']},{r['correct']}\n" for r in rows if r["passed"]))
        from labels import merge
        merge(validation,decisions,manifest)
        if destination is not None:
            (tmp/"SYNTHETIC.txt").write_text("Synthetic integration fixture. Do not use these results as empirical thesis evidence.\n")
        np.savez(cache,ids=[r["id"] for r in rows],diff=x,prefix=h,seconds=np.full(len(x),.01),widths=[2,3],tiny=np.array(h)[:,:,:2])
        roster=tmp/"roster.jsonl"
        roster.write_text("\n".join(json.dumps(dict(project="test",split="test",bug=str(i))) for i in range(10)))
        subprocess.run([sys.executable,str(Path(__file__).with_name("run.py")),"study",str(manifest),str(cache),str(out),
                        "--device","cpu","--hidden-width","32","--attention-width","8","--skip-auc-ci","--select-fixed","--extended-studies","--backend","paper" if paper else "linear","--epochs","2","--train-on","all","--roster",str(roster),"--seeds","5","7","--budgets","2","5","--k","2","--threshold",".5","--reward",".2"],check=True)
        settings=json.loads((out/"settings.json").read_text())
        assert settings["attention_width"]==8 and settings["hidden_width"]==32 and settings["budgets"]==[2,5]
        assert (out/"budget_curve.pdf").stat().st_size > 0
        summary=pd.read_csv(out/"schedule.csv")
        assert (summary.bugs==10).all() and (summary.repairable==8).all()
        assert ((summary.repaired<=summary.retained)&(summary.repaired<=summary.returned)).all()
        paired=pd.read_csv(out/"paired.csv")
        assert {"ratio","probe","last","diff"}.issubset(paired.b.unique())
        records=pd.read_csv(out/"schedule_by_bug.csv")
        assert (records.seconds<=records.budget).all()
        assert {"prefix_3","prefix_5"}.issubset(set(pd.read_csv(out/"rank_by_bug.csv").method))
        studies=pd.read_csv(out/"validation_studies.csv")
        assert len(studies)==2*2*6 and (studies.bugs==10).all()
        outcomes=studies[["pool_without_correct","correct_not_validated","correct_validated_not_returned","repaired"]]
        assert (outcomes>=0).all().all() and (outcomes.sum(axis=1)==studies.bugs).all()
        assert (studies.loc[studies.rule=="first","correct_validated_not_returned"]==0).all()
        contrast=pd.read_csv(out/"validation_contrasts.csv")
        cross=contrast[contrast.contrast=="representation_x_reward"].set_index(["budget","rule"])
        rates=studies[studies.cost_model=="regression"].pivot(index=["budget","rule"],
            columns=["representation","policy"],values="success_rate")
        expected=(rates["fusion","joint"]-rates["fusion","ratio"])-(rates["last","joint"]-rates["last","ratio"])
        np.testing.assert_allclose(cross.difference_pp,100*expected,atol=1e-10)
        assert (pd.read_csv(out/"cost_errors_by_project.csv").timing=="unreported").all()
        from validation_studies import cost_errors
        timings=pd.DataFrame(dict(project=["p"]*3,status=["passed","test_timeout",None],
                                  seconds=[3.,7.,4.],cost=[2.,3.,4.]))
        errors=cost_errors(timings,5.).set_index(["cost_model","timing"])
        assert errors.loc[("regression","completed"),"mae_seconds"]==1.
        assert errors.loc[("regression","timeout"),"mean_underprediction_lower_bound_seconds"]==4.
        assert pd.isna(errors.loc[("training_median","unreported"),"mae_seconds"])
        g=pd.DataFrame(dict(group=["b","b"],project=["p","p"],id=["a","b"],p=[.9,.8],u=[0.,0.],
                            cost=[1.,1.],seconds=[4.,1.],passed=[1,1],correct=[0,1],overhead=[0.,0.]))
        result=replay(g,"ratio",2,"first",SimpleNamespace(reward=0))
        assert result["completed"]==0 and result["interrupted_id"]=="a" and result["returned"]==0
        result=replay(g,"ratio",8,"first",SimpleNamespace(reward=0))
        assert result["returned"]==1 and result["success"]==0
        g["return_p"]=[.1,.95]
        assert replay(g,"ratio",8,"best",SimpleNamespace(reward=0))["returned_id"]=="b"
    print("Synthetic integration and budget boundary checks passed.")


if __name__=="__main__":
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paper",action="store_true")
    parser.add_argument("--output",type=Path,help="keep synthetic inputs and results in a new directory")
    args=parser.parse_args()
    main(args.paper,args.output)
