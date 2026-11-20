"""Generate, validate and encode single-region Defects4J repair candidates."""
import argparse
import hashlib
import json
import random
import re
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np


def records(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def run(cmd, cwd, timeout):
    start = time.perf_counter()
    try:
        proc = subprocess.run(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              timeout=timeout, text=True, errors="replace")
        return proc.returncode, proc.stdout, time.perf_counter()-start
    except subprocess.TimeoutExpired:
        return 124, "timeout", time.perf_counter()-start


def generate(args, bugs):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.manual_seed(args.seed)
    model = AutoModelForCausalLM.from_pretrained(args.model,revision=args.revision,torch_dtype="auto").to(args.device).eval()
    tok = AutoTokenizer.from_pretrained(args.model,revision=args.revision)
    for bug in bugs:
        prompt = bug.get("prompt", "Repair the following buggy Java code. Return only the replacement code.\n"+bug["original"])
        message = tok.apply_chat_template([{"role":"user","content":prompt}], tokenize=False,add_generation_prompt=True)
        ids=tok(message,return_tensors="pt",add_special_tokens=False).input_ids.to(args.device)
        seen=set()
        for sample in range(args.samples):
            with torch.inference_mode():
                seq=model.generate(ids,do_sample=True,temperature=args.temperature,top_p=args.top_p,
                                   max_new_tokens=args.max_tokens,pad_token_id=tok.eos_token_id)
                generated=seq[0,ids.shape[1]:]
                # Re-score raw model likelihood, independent of the sampling temperature.
                logits=model(seq).logits[0,ids.shape[1]-1:-1].float().log_softmax(-1)
                score=logits.gather(1,generated[:,None]).mean().item()
            patch=tok.decode(generated,skip_special_tokens=True).strip()
            fence=re.fullmatch(r"```(?:java)?\s*\n(.*?)\n?```",patch,re.S)
            patch=(fence.group(1) if fence else patch).strip()
            if not patch or patch in seen:
                continue
            seen.add(patch)
            identity=json.dumps([bug["project"],bug["bug"],patch],ensure_ascii=False)
            yield {**bug,"patch":patch,"id":hashlib.sha256(identity.encode()).hexdigest(),
                   "gen_score":score,"sample":sample,"generation_model":args.model,
                   "generation_revision":args.revision,"generation_seed":args.seed}


def validate(args, candidates):
    for row in candidates:
        with tempfile.TemporaryDirectory() as tmp:
            work=Path(tmp)/"checkout"
            cmd=["defects4j","checkout","-p",str(row["project"]),"-v",str(row["bug"])+"b","-w",str(work)]
            subprocess.run(cmd,check=True,capture_output=True,timeout=args.timeout)
            target=(work/row["file"]).resolve()
            if not target.is_relative_to(work.resolve()):
                raise ValueError("source file must be inside checkout")
            encoding=row.get("encoding","utf-8")
            lines=target.read_bytes().decode(encoding).splitlines(keepends=True)
            start,end=row["start_line"]-1,row["end_line"]
            if not 0<=start<end<=len(lines) or "".join(lines[start:end]).strip()!=row["original"].strip():
                raise ValueError("source span does not match buggy checkout: "+row["id"])
            newline="\r\n" if lines[start].endswith("\r\n") else "\n"
            replacement=row["patch"].replace("\r\n","\n").replace("\n",newline)
            if lines[end-1].endswith("\n"):
                replacement=replacement.rstrip("\r\n")+newline
            target.write_bytes(("".join(lines[:start])+replacement+"".join(lines[end:])).encode(encoding))
            code,compile_log,compile_s=run(["defects4j","compile"],work,args.timeout)
            test_s=0.; test_log=""; passed=False
            status="compile_timeout" if code==124 else "compile_failed"
            if code==0:
                test_code,test_log,test_s=run(["defects4j","test"],work,args.timeout)
                count=re.findall(r"^Failing tests:\s*(\d+)\s*$",test_log,re.M)
                if test_code==124:
                    status="test_timeout"
                elif not count:
                    raise RuntimeError("test harness did not report outcome: "+test_log[-500:])
                else:
                    passed=test_code==0 and int(count[-1])==0
                    status="passed" if passed else "failed"
            yield {**row,"passed":int(passed),"correct":None if passed else 0,
                   "status":status,"compile_seconds":compile_s,"test_seconds":test_s,
                   "seconds":compile_s+test_s,"compile_log":compile_log,"test_log":test_log}


def extract(args, rows):
    import torch
    from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer
    device=args.device
    ids=np.array([r["id"] for r in rows])
    overhead=np.zeros(len(rows))
    def sync():
        if str(device).startswith("cuda"):
            torch.cuda.synchronize()
    def load(name, rev, cls):
        tok=AutoTokenizer.from_pretrained(name,revision=rev)
        net=cls.from_pretrained(name,revision=rev,torch_dtype="auto").to(device).eval()
        return tok,net

    diff=[]
    tok,net=load(args.encoder,args.encoder_revision,AutoModel)
    with torch.inference_mode():
        for i,row in enumerate(rows):
            sync(); start=time.perf_counter()
            vec=[]
            for code in [row["original"],row["patch"]]:
                # UniXcoder encoder-only input format, mean over non-padding tokens.
                tokens=[tok.cls_token,"<encoder-only>",tok.sep_token]+tok.tokenize(code)[:args.encoder_limit-4]+[tok.sep_token]
                input_ids=torch.tensor([tok.convert_tokens_to_ids(tokens)],device=device)
                vec.append(net(input_ids=input_ids).last_hidden_state[0].float().mean(0).cpu().numpy())
            diff.append(np.concatenate([vec[0],vec[1],vec[1]-vec[0]]))
            sync(); overhead[i]+=time.perf_counter()-start
    del net
    blocks=[]
    for name,rev in zip(args.decoders,args.decoder_revisions,strict=True):
        tok,net=load(name,rev,AutoModelForCausalLM)
        block=[]
        with torch.inference_mode():
            for i,row in enumerate(rows):
                sync(); start=time.perf_counter()
                patch_ids=tok.encode(row["patch"],add_special_tokens=False)
                if not patch_ids or len(patch_ids)>=args.context:
                    raise ValueError("empty or oversized replacement: "+row["id"])
                prefix=tok.encode("Buggy code:\n"+row["original"]+"\nReplacement:\n",add_special_tokens=False)
                prefix=prefix[-(args.context-len(patch_ids)):]
                inp=torch.tensor([prefix+patch_ids],device=device)
                state=net.base_model(input_ids=inp,use_cache=False).last_hidden_state[0]
                positions=len(prefix)+np.ceil(np.arange(1,11)*len(patch_ids)/10).astype(int)-1
                block.append(state[torch.tensor(positions,device=device)].float().cpu().numpy())
                sync(); overhead[i]+=time.perf_counter()-start
        blocks.append(np.stack(block)); del net
    with Path(args.out).open("xb") as stream:
        np.savez(stream,ids=ids,diff=np.stack(diff),prefix=np.concatenate(blocks,axis=-1),seconds=overhead)
    Path(str(args.out)+".json").write_text(json.dumps(vars(args),indent=2))


if __name__=="__main__":
    ap=argparse.ArgumentParser()
    sub=ap.add_subparsers(dest="stage",required=True)
    gen=sub.add_parser("generate"); val=sub.add_parser("validate"); enc=sub.add_parser("encode")
    for parser in [gen,val,enc]:
        parser.add_argument("input"); parser.add_argument("out")
    for parser in [gen,enc]:
        parser.add_argument("--device",default="cuda")
    gen.add_argument("--model",default="Qwen/Qwen2.5-Coder-7B-Instruct")
    gen.add_argument("--revision",required=True); gen.add_argument("--samples",type=int,required=True)
    gen.add_argument("--temperature",type=float,default=.8); gen.add_argument("--top-p",type=float,default=.95)
    gen.add_argument("--seed",type=int,default=11); gen.add_argument("--max-tokens",type=int,default=512)
    val.add_argument("--timeout",type=int,default=600)
    enc.add_argument("--encoder",default="microsoft/unixcoder-base")
    enc.add_argument("--encoder-revision",required=True)
    enc.add_argument("--decoders",nargs=2,default=["Qwen/Qwen2.5-Coder-1.5B","Qwen/Qwen2.5-Coder-7B"])
    enc.add_argument("--decoder-revisions",nargs=2,required=True)
    enc.add_argument("--encoder-limit",type=int,default=512); enc.add_argument("--context",type=int,default=4096)
    args=ap.parse_args()
    items=records(args.input)
    if args.stage=="encode":
        extract(args,items)
    else:
        outputs=generate(args,items) if args.stage=="generate" else validate(args,items)
        with Path(args.out).open("x") as stream:
            for row in outputs:
                stream.write(json.dumps(row,ensure_ascii=False)+"\n"); stream.flush()
