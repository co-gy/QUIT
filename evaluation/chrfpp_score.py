"""
python chrfpp_score.py \
    -s txt-wmt25-hy/src.txt \
    -t txt-wmt25-hy/standard-mbr-512.txt "txt-wmt25-hy/ada-mbr-512-(ws=1|alpha=0.001).txt" "txt-wmt25-hy/ada-mbr-512-(ws=3|alpha=0.001).txt" "txt-wmt25-hy/ada-mbr-512-(ws=5|alpha=0.001).txt" "txt-wmt25-hy/ada-mbr-512-(ws=8|alpha=0.001).txt"   "txt-wmt25-hy/ada-mbr-512-(ws=10|alpha=0.001).txt" "txt-wmt25-hy/ada-mbr-512-(ws=12|alpha=0.001).txt" "txt-wmt25-hy/ada-mbr-512-(ws=15|alpha=0.001).txt" "txt-wmt25-hy/ada-mbr-512-(ws=20|alpha=0.001).txt" "txt-wmt25-hy/ada-mbr-512-(ws=30|alpha=0.001).txt" \
    -r txt-wmt25-hy/ref.txt \
    --to_json evaluate-wmt25-hy-chrfpp.json
"""


import sacrebleu
from functools import lru_cache
import json
import argparse
from tqdm import tqdm
import numpy as np


@lru_cache(maxsize=1024)
def chrf_pp(hypothesis, reference):
    return sacrebleu.sentence_chrf(hypothesis, [reference], word_order=2).score


def chrf_pp_batch(hypotheses, references):
    result = []
    for h, r in zip(hypotheses, references):
        result.append(chrf_pp(h, r))
    return result

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", type=str)
    parser.add_argument('-t', nargs='+', type=str)
    parser.add_argument("-r", type=str)
    parser.add_argument("--to_json", type=str)
    args = parser.parse_args()
    print(args.s, args.t, args.r)

    res = {}

    with open(args.s, "r") as f:
        srcs = f.readlines()

    with open(args.r, "r") as f:
        refs = f.readlines()

    for t in tqdm(args.t):
        with open(t, "r") as f:
            hypos = f.readlines()
        scores = chrf_pp_batch(hypos, refs)
        res[t] = [{
            "src": src,
            "mt": mt,
            "ref": ref,
            "chrf++": score
        } for src, mt, ref, score in zip(srcs, hypos, refs, scores)]

    with open(args.to_json, "w") as f:
        json.dump(res, f)
    print("chrf++ result save at", args.to_json)
    print("chrf++:", np.mean(scores))
