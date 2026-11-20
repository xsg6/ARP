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


def repair_prompt(bug):
    if "prompt" in bug:
        return bug["prompt"]
    return ("Repair the localized buggy Java region. Return only the replacement code.\nMethod:\n"
            +bug.get("buggy_context",bug["original"])+"\nRegion:\n"+bug["original"]
            +"\nFailing test and diagnostic:\n"+bug.get("failure",""))


def generate(args, bugs):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.manual_seed(args.seed)
    model = AutoModelForCausalLM.from_pretrained(args.model,revision=args.revision,torch_dtype="auto").to(args.device).eval()
    tok = AutoTokenizer.from_pretrained(args.model,revision=args.revision)
    if args.adapter:
        from injection import attach
        attach(model,args)
    for bug in bugs:
        prompt = repair_prompt(bug)
        message = tok.apply_chat_template([{"role":"user","content":prompt}], tokenize=False,add_generation_prompt=True)
        ids=tok(message,return_tensors="pt",add_special_tokens=False).input_ids.to(args.device)
        seen={}
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
            if not patch:
                continue
            if patch in seen:
                seen[patch]["draw_count"]+=1
                continue
            identity=json.dumps([bug["project"],bug["bug"],patch],ensure_ascii=False)
            seen[patch]={**bug,"draw_count":1,"patch":patch,"id":hashlib.sha256(identity.encode()).hexdigest(),
                   "gen_score":score,"sample":sample,"generation_model":args.model,
                   "generation_revision":args.revision,"generation_seed":args.seed}
        yield from seen.values()


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
            ods={}
            if args.sources:
                output=Path(args.sources)/hashlib.sha256(row["id"].encode()).hexdigest()
                output.mkdir(parents=True,exist_ok=True)
                (output/"before.java").write_bytes("".join(lines).encode(encoding))
                (output/"after.java").write_bytes(target.read_bytes())
                ods={"ods_s":str((output/"before.java").resolve()),"ods_t":str((output/"after.java").resolve())}
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
            yield {**row,**ods,"passed":int(passed),"correct":None if passed else 0,
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

    profiles=[]
    def profile(model_name,timings,arrays):
        profiles.append(dict(model=model_name,extraction_seconds=float(np.sum(timings)),
            feature_bytes=sum(a.nbytes for a in arrays),
            peak_cuda_bytes=torch.cuda.max_memory_allocated() if str(device).startswith("cuda") else None))
    def reset_peak():
        if str(device).startswith("cuda"):
            torch.cuda.reset_peak_memory_stats()
    reset_peak()
    diff=[]
    tok,net=load(args.encoder,args.encoder_revision,AutoModel)
    with torch.inference_mode():
        for i,row in enumerate(rows):
            sync(); start=time.perf_counter()
            vec=[]
            buggy=row.get("buggy_context",row["original"])
            if "patched_context" not in row and buggy.count(row["original"])!=1:
                raise ValueError("ambiguous edit span; supply patched_context for "+row["id"])
            patched=row.get("patched_context",buggy.replace(row["original"],row["patch"],1))
            for code in [buggy,patched]:
                # UniXcoder encoder-only input format, mean over non-padding tokens.
                tokens=[tok.cls_token,"<encoder-only>",tok.sep_token]+tok.tokenize(code)[:args.encoder_limit-4]+[tok.sep_token]
                input_ids=torch.tensor([tok.convert_tokens_to_ids(tokens)],device=device)
                vec.append(net(input_ids=input_ids).last_hidden_state[0].float().mean(0).cpu().numpy())
            diff.append(np.concatenate([vec[0],vec[1],np.abs(vec[1]-vec[0])]))
            sync(); overhead[i]+=time.perf_counter()-start
    profile(args.encoder,overhead,diff)
    del net
    edit_seconds=overhead.copy()
    blocks=[]; block_seconds=[]; widths=[]
    for name,rev in zip(args.decoders,args.decoder_revisions,strict=True):
        reset_peak()
        tok,net=load(name,rev,AutoModelForCausalLM)
        previous=overhead.copy()
        block=[]
        with torch.inference_mode():
            for i,row in enumerate(rows):
                sync(); start=time.perf_counter()
                patch_ids=tok.encode(row["patch"],add_special_tokens=False)
                if not patch_ids or len(patch_ids)>=args.context:
                    raise ValueError("empty or oversized replacement: "+row["id"])
                prefix=tok.encode("Buggy code:\n"+row.get("buggy_context",row["original"])+"\nReplacement:\n",add_special_tokens=False)
                prefix=prefix[-(args.context-len(patch_ids)):]
                inp=torch.tensor([prefix+patch_ids],device=device)
                state=net.base_model(input_ids=inp,use_cache=False).last_hidden_state[0]
                positions=len(prefix)+np.maximum(1,np.arange(1,11)*len(patch_ids)//10)-1
                block.append(state[torch.tensor(positions,device=device)].float().cpu().numpy())
                sync(); overhead[i]+=time.perf_counter()-start
        profile(name,overhead-previous,block)
        blocks.append(np.stack(block)); widths.append(blocks[-1].shape[-1]); block_seconds.append(overhead-previous); del net
    with Path(args.out).open("xb") as stream:
        np.savez(stream,ids=ids,diff=np.stack(diff),prefix=np.concatenate(blocks[:2],axis=-1),seconds=edit_seconds+sum(block_seconds[:2]),
                 edit_seconds=edit_seconds,decoder_seconds=np.stack(block_seconds),widths=widths[:2],
                 **({"tiny":blocks[2]} if len(blocks)>2 else {}))
    Path(str(args.out)+".json").write_text(json.dumps({**vars(args),"profile":profiles},indent=2))


if __name__=="__main__":
    ap=argparse.ArgumentParser()
    sub=ap.add_subparsers(dest="stage",required=True)
    gen=sub.add_parser("generate"); val=sub.add_parser("validate"); enc=sub.add_parser("encode")
    for parser in [gen,val,enc]:
        parser.add_argument("input"); parser.add_argument("out")
    for parser in [gen,enc]:
        parser.add_argument("--device",default="cuda")
    gen.add_argument("--adapter",help="optional trained RQ5 injection checkpoint")
    gen.add_argument("--model",default="Qwen/Qwen2.5-Coder-7B-Instruct")
    gen.add_argument("--revision",required=True); gen.add_argument("--samples",type=int,required=True)
    gen.add_argument("--temperature",type=float,default=.8); gen.add_argument("--top-p",type=float,default=.95)
    gen.add_argument("--seed",type=int,default=11); gen.add_argument("--max-tokens",type=int,default=512)
    val.add_argument("--sources",help="export full Java files for Coming")
    val.add_argument("--timeout",type=int,default=600)
    enc.add_argument("--encoder",default="microsoft/unixcoder-base")
    enc.add_argument("--encoder-revision",required=True)
    enc.add_argument("--decoders",nargs="+",default=["Qwen/Qwen2.5-Coder-1.5B","Qwen/Qwen2.5-Coder-7B"])
    enc.add_argument("--decoder-revisions",nargs="+",required=True)
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
