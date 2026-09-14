import argparse
import os
import json
from collections import defaultdict

from tqdm import tqdm
from gemba.utils import get_gemba_scores


print(os.environ["OPENAI_API_KEY"])
print(os.environ.get("OPENAI_API_BASE", None))

parser = argparse.ArgumentParser()
parser.add_argument("-s", type=str)
parser.add_argument("-t", nargs="+", type=str)
parser.add_argument("--to_json", type=str)
parser.add_argument("--wmt_data", type=str)
parser.add_argument(
    "--workers",
    type=int,
    default=16,
    help="Number of concurrent requests inside GEMBA bulk_request",
)
args = parser.parse_args()

print(args.s, args.t, args.to_json, args.wmt_data)
print("workers:", args.workers)


with open(args.wmt_data, "r") as f:
    wmt_data = [json.loads(l) for l in f]
with open(args.s, "r") as f:
    srcs = f.readlines()
lang_groups = defaultdict(list)
for idx, wmt_item in enumerate(wmt_data):
    lang_pair = (wmt_item["src_lang"], wmt_item["tgt_lang"])
    lang_groups[lang_pair].append(idx)
print(f"Found {len(lang_groups)} language pairs:")

for (src_lang, tgt_lang), indices in lang_groups.items():
    print(f"{src_lang} -> {tgt_lang}: {len(indices)} samples")

res = {}

for t in tqdm(args.t, desc="systems"):
    print("=" * 80)
    print("Evaluating:", t)
    print("=" * 80)

    with open(t, "r") as f:
        hypos = f.readlines()

    results = [None] * len(srcs)


    for group_idx, ((src_lang, tgt_lang), indices) in enumerate(lang_groups.items(), start=1):
        print(f"[{group_idx}/{len(lang_groups)}] {src_lang} -> {tgt_lang}: {len(indices)} samples")
        group_srcs = [srcs[idx] for idx in indices]
        group_hypos = [hypos[idx] for idx in indices]

        answers = get_gemba_scores(
            group_srcs,
            group_hypos,
            src_lang,
            tgt_lang,
            "GEMBA-MQM",
            "gemma",
            None,
            max_workers=args.workers,
        )

        for idx, answer in zip(indices, answers):
            results[idx] = {
                "src": srcs[idx],
                "mt": hypos[idx],
                "gemba-mqm": answer,
            }

    res[t] = results

    with open(args.to_json, "w") as f:
        json.dump(res, f, ensure_ascii=False)

    print("gemba-mqm result saved at", args.to_json)