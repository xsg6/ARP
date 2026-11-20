"""Extract ODS features with Coming and train the held-out-project baseline."""
import argparse
from pathlib import Path
import json
import shutil
import subprocess
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from prepare import records


def main(args):
    rows=records(args.manifest)
    features=[]
    for row in rows:
        with TemporaryDirectory() as tmp:
            work=Path(tmp)
            pair=work/"input"/"change"/"File"
            pair.mkdir(parents=True)
            for version in ["s","t"]:
                shutil.copyfile(row["ods_"+version],pair/f"change_File_{version}.java")
            subprocess.run(["java","-classpath",str(Path(args.jar).resolve()),"fr.inria.coming.main.ComingMain",
                            "-input","files","-mode","features","-location",str(work/"input"),
                            "-output",str(work/"out")],cwd=work,check=True,timeout=args.timeout)
            tables=list(work.rglob("test.csv"))
            if len(tables)!=1:
                raise ValueError("expected Coming test.csv for "+row["id"])
            table=pd.read_csv(tables[0])
            if len(table)!=1:
                raise ValueError("expected one feature row per patch")
            features.append(table.iloc[:,1:])
    columns=features[0].columns.tolist()
    if any(f.columns.tolist()!=columns for f in features):
        raise ValueError("inconsistent ODS feature schema")
    matrix=pd.concat(features,ignore_index=True)
    train=np.array([r["split"]=="train" and r["passed"]==1 for r in rows])
    labels=np.array([r["correct"] for r in rows],dtype=int)
    model=XGBClassifier(max_depth=6,gamma=.5,learning_rate=.3,random_state=args.seed,n_jobs=1)
    model.fit(matrix[train],labels[train])
    scores=model.predict_proba(matrix)[:,list(model.classes_).index(1)]
    with Path(args.out).open("x") as f:
        for r,score in zip(rows,scores):
            f.write(json.dumps({**r,"ods_score":float(score)},ensure_ascii=False)+"\n")


if __name__=="__main__":
    ap=argparse.ArgumentParser()
    for key in ["manifest","jar","out"]:
        ap.add_argument(key)
    ap.add_argument("--timeout",type=int,default=300);ap.add_argument("--seed",type=int,default=42)
    main(ap.parse_args())
