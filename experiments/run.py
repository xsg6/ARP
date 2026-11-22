"""Launch the repository's experiment stages with a shared JSON configuration."""
import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
STAGES = {
    "generate": ("prepare.py", "generate"),
    "labels": ("labels.py",),
    "validate": ("prepare.py", "validate"),
    "encode": ("prepare.py", "encode"),
    "evaluate": ("evaluate.py",),
    "report": ("report.py",),
    "ods": ("ods.py",),
    "injection_train": ("injection.py", "train"),
    "injection_compare": ("injection.py", "compare"),
}


def lock_models(config, destination):
    from huggingface_hub import HfApi
    api = HfApi()
    def revision(name, ref):
        if re.fullmatch("[0-9a-f]{40}", ref):
            return ref
        return api.model_info(name, revision=ref).sha
    gen = config["generate"]
    gen["revision"] = revision(gen["model"], gen["revision"])
    enc = config["encode"]
    enc["encoder_revision"] = revision(enc["encoder"], enc["encoder_revision"])
    enc["decoder_revisions"] = [revision(name, ref) for name, ref in
                                zip(enc["decoders"], enc["decoder_revisions"], strict=True)]
    with destination.open("x", encoding="utf-8") as file:
        json.dump(config, file, indent=2)
        file.write("\n")


def command(config, stage, arguments):
    script, *action = STAGES[stage]
    options = dict(config.get(stage, {}))
    if stage == "injection_train":
        options.update({k: config["generate"][k] for k in ("model", "revision")})
    if stage == "injection_compare":
        options["draws"] = config["generate"]["samples"]
    split = next((i for i, arg in enumerate(arguments) if arg.startswith("--")), len(arguments))
    cmd = [sys.executable, str(ROOT / script), *action, *arguments[:split]]
    for name, value in options.items():
        flag = "--" + name.replace("_", "-")
        if isinstance(value, bool):
            if value:
                cmd.append(flag)
        else:
            values = value if isinstance(value, list) else [value]
            cmd.extend([flag, *map(str, values)])
    return cmd + arguments[split:]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/thesis.json")
    parser.add_argument("stage", choices=["lock", "study", *STAGES])
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if args.stage == "lock":
        if len(args.arguments) != 1:
            parser.error("lock requires an output JSON path")
        lock_models(config, Path(args.arguments[0]))
    elif args.stage == "study":
        if len(args.arguments) < 3:
            parser.error("study requires MANIFEST CACHE RESULT_DIR followed by evaluation options")
        subprocess.run(command(config, "evaluate", args.arguments), check=True)
        subprocess.run(command(config, "report", [args.arguments[2]]), check=True)
    else:
        subprocess.run(command(config, args.stage, args.arguments), check=True)
