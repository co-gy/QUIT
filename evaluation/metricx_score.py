"""
python metricx_score.py \
    -s txt-wmt25-hy/src.txt \
    -t txt-wmt25-hy/standard-mbr-512.txt "txt-wmt25-hy/ada-mbr-512-(ws=1|alpha=0.001).txt" "txt-wmt25-hy/ada-mbr-512-(ws=3|alpha=0.001).txt" "txt-wmt25-hy/ada-mbr-512-(ws=5|alpha=0.001).txt" "txt-wmt25-hy/ada-mbr-512-(ws=8|alpha=0.001).txt"   "txt-wmt25-hy/ada-mbr-512-(ws=10|alpha=0.001).txt" "txt-wmt25-hy/ada-mbr-512-(ws=12|alpha=0.001).txt" "txt-wmt25-hy/ada-mbr-512-(ws=15|alpha=0.001).txt" "txt-wmt25-hy/ada-mbr-512-(ws=20|alpha=0.001).txt" "txt-wmt25-hy/ada-mbr-512-(ws=30|alpha=0.001).txt" \
    -r txt-wmt25-hy/ref.txt \
    --to_json evaluate-wmt25-hy-metricx.json
"""
import subprocess
import os
import argparse
import json
import tempfile

import numpy as np
from tqdm import tqdm

def metricx_batch(srcs, refs, hypos):

    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = os.path.join(tmpdir, "input.jsonl")
        with open(input_path, "w") as f:
            for src, ref, hypo in zip(srcs, refs, hypos):
                json.dump({
                    "source": src,
                    "reference": ref,
                    "hypothesis": hypo
                }, f)
                f.write("\n")
        output_path = os.path.join(tmpdir, 'output.jsonl')
        cmd = [
            "python",
            "-m", "metricx24.predict",
            "--tokenizer", "google/mt5-xl",
            "--model_name_or_path", "google/metricx-24-hybrid-xl-v2p6",
            "--max_input_length", "1536",
            "--batch_size", "16",
            "--input_file", input_path,
            "--output_file", output_path
        ]
        custom_env = os.environ.copy()
        custom_env["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
        subprocess.run(cmd, env=custom_env, check=True)
        with open(output_path, "r") as f:
            res = [json.loads(line) for line in f]
    return [item["prediction"] for item in res]


parser = argparse.ArgumentParser()
parser.add_argument("-s", type=str)
parser.add_argument('-t', nargs='+', type=str)
parser.add_argument("-r", type=str)
parser.add_argument("--to_json", type=str)
args = parser.parse_args()

res = {}

with open(args.s, "r") as f:
    srcs = f.readlines()

with open(args.r, "r") as f:
    refs = f.readlines()

for t in tqdm(args.t):
    with open(t, "r") as f:
        hypos = f.readlines()
    scores = metricx_batch(srcs, refs, hypos)
    res[t] = [{
        "src": src,
        "mt": mt,
        "ref": ref,
        "metricx": score
    } for src, mt, ref, score in zip(srcs, hypos, refs, scores)]

    with open(args.to_json, "w") as f:
        json.dump(res, f)

    print("metricx result save at", args.to_json)
    print("metricx:", np.mean(scores))



