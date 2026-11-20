"""CPU integration check of extraction, teacher forcing and generation using tiny local weights."""
import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import (PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM,
                          RobertaConfig, RobertaModel)

from prepare import extract, generate
from injection import train


def main():
    torch.set_num_threads(1);torch.manual_seed(19)
    with TemporaryDirectory() as temp:
        root=Path(temp)
        tokens=["<pad>","<unk>","<s>","</s>","<encoder-only>","return","a","b",";","Buggy","Replacement"]
        vocabulary={token:i for i,token in enumerate(tokens)}
        backend=Tokenizer(WordLevel(vocabulary,unk_token="<unk>"));backend.pre_tokenizer=Whitespace()
        tok=PreTrainedTokenizerFast(tokenizer_object=backend,pad_token="<pad>",unk_token="<unk>",
                                    cls_token="<s>",sep_token="</s>",bos_token="<s>",eos_token="</s>")
        tok.chat_template="{% for m in messages %}{{m['content']}}{% endfor %}"
        dec=str(root/"decoder");enc=str(root/"encoder")
        decoder=Qwen2ForCausalLM(Qwen2Config(vocab_size=len(tokens),hidden_size=16,intermediate_size=32,
                            num_hidden_layers=1,num_attention_heads=2,num_key_value_heads=2,
                            bos_token_id=2,eos_token_id=3,pad_token_id=0))
        decoder.save_pretrained(dec);tok.save_pretrained(dec)
        RobertaModel(RobertaConfig(vocab_size=len(tokens),hidden_size=12,intermediate_size=24,
                            num_hidden_layers=1,num_attention_heads=2,max_position_embeddings=128)).save_pretrained(enc)
        tok.save_pretrained(enc)
        row=dict(id="sample",project="fixture",bug="1",split="train",original="a",patch="return b ;",
                 buggy_context="return a ;",correct=1,passed=1)
        inp=root/"input.jsonl";inp.write_text(json.dumps(row)+"\n")
        cache=str(root/"cache.npz"); ckpt=str(root/"adapter.pt")
        extract(argparse.Namespace(device="cpu",encoder=enc,encoder_revision="local",decoders=[dec]*3,
                    decoder_revisions=["local"]*3,encoder_limit=48,context=64,out=cache),[row])
        with np.load(cache) as cached:
            assert cached["prefix"].shape==(1,10,32) and cached["tiny"].shape==(1,10,16)
            assert cached["diff"].shape==(1,36) and np.isfinite(cached["prefix"]).all()
        train(argparse.Namespace(manifest=str(inp),cache=cache,model=dec,revision="local",device="cpu",
                                 epochs=1,prototype_dim=5,seed=19,lr=.001,out=ckpt))
        for adapter in (None,ckpt):
            outputs=list(generate(argparse.Namespace(seed=21,model=dec,revision="local",device="cpu",samples=5,
                         temperature=.8,top_p=.95,max_tokens=6,adapter=adapter),[row]))
            assert all(np.isfinite(o["gen_score"]) for o in outputs)
    print("Tiny local model extraction, gated adapter training and generation completed.")


if __name__=="__main__":
    main()
