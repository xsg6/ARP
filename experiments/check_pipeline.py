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


def main():
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
    with TemporaryDirectory() as tmp:
        tmp=Path(tmp); manifest=tmp/"fixture.jsonl"; cache=tmp/"fixture.npz"; out=tmp/"out"
        manifest.write_text("\n".join(json.dumps(r) for r in rows))
        np.savez(cache,ids=[r["id"] for r in rows],diff=x,prefix=h,seconds=np.full(len(x),.01))
        roster=tmp/"roster.jsonl"
        roster.write_text("\n".join(json.dumps(dict(project="test",split="test",bug=str(i))) for i in range(10)))
        subprocess.run([sys.executable,str(Path(__file__).with_name("evaluate.py")),str(manifest),str(cache),str(out),
                        "--roster",str(roster),"--seeds","5","7","--budgets","2","5","--k","2","--threshold",".5","--reward",".2"],check=True)
        from report import report
        report(out)
        assert (out/"budget_curve.pdf").stat().st_size > 0
        summary=pd.read_csv(out/"schedule.csv")
        assert (summary.bugs==10).all() and (summary.repairable==8).all()
        assert ((summary.repaired<=summary.retained)&(summary.repaired<=summary.returned)).all()
        paired=pd.read_csv(out/"paired.csv")
        assert {"ratio","probe","last","diff"}.issubset(paired.b.unique())
        records=pd.read_csv(out/"schedule_by_bug.csv")
        assert (records.seconds<=records.budget).all()
        g=pd.DataFrame(dict(group=["b","b"],project=["p","p"],id=["a","b"],p=[.9,.8],u=[0.,0.],
                            cost=[1.,1.],seconds=[4.,1.],passed=[1,1],correct=[0,1],overhead=[0.,0.]))
        result=replay(g,"ratio",2,"first",SimpleNamespace(reward=0))
        assert result["completed"]==0 and result["interrupted_id"]=="a" and result["returned"]==0
        result=replay(g,"ratio",8,"first",SimpleNamespace(reward=0))
        assert result["returned"]==1 and result["success"]==0
    print("Synthetic integration and budget boundary checks passed.")


if __name__=="__main__":
    main()
