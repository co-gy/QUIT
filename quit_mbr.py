import argparse
import numpy as np
import json
import re
from tqdm import tqdm



parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="wmt25")
parser.add_argument("--model", type=str, default="Hy-MT2-30B-A3B")
parser.add_argument("--hypo_nums", type=int, default=512)
parser.add_argument("--k", type=int, default=8, help="batch size")
parser.add_argument("--w", type=int, default=8, help="window size")
parser.add_argument("--alpha", type=float, default=0.001, help="stopping threshold")
args = parser.parse_args()


# ========================= load data =========================
short_model_name = {
    "Hy-MT2-30B-A3B": "hy",
    "Qwen3-8B": "qwen",
    "translategemma-12b-it": "gemma",
    "vllm-translategemma-12b-it": "gemma"
}[args.model]
txt_dir = f"txt-{args.dataset}-{short_model_name}"
um_path = f"utility_matrix/{args.dataset}-{short_model_name}-um.npy"
wmt_data_path = f"mt_data/{args.dataset}-genmt-{args.model}-{args.hypo_nums}.jsonl"
print("loading utility matrix...")
um_lst = np.load(um_path)
print("loading wmt data...")
with open(wmt_data_path, "r", encoding="utf-8") as f:
    wmt_data = [json.loads(line) for line in f]
sentence_num = um_lst.shape[0]
hypothesis_num = um_lst.shape[1]
print("=" * 10, "  data  ", "="* 10)
print(f"utility matrix({um_lst.shape}):", um_path)
print(f"wmt data({len(wmt_data)}):", wmt_data_path)


# ========================= QUIT param =========================
batch_size = args.k
window_size = args.w
alpha = args.alpha
print("="* 10, "QUIT param", "=" * 10)
print("batch size:", batch_size)
print("window size:", window_size)
print("alpha:", alpha)


# ========================= QUIT mbr process =========================
compute_analyze_result = {}
ada_mbr = []
stop_N = []
for sentence_id in range(sentence_num):
    um = um_lst[sentence_id]
    candidate = wmt_data[sentence_id]["result"]
    M_window = []
    for i in range(1, hypothesis_num // batch_size + 1):
        N = i*batch_size
        M = um[:N, :N].mean(axis=1).max()
        if len(M_window) >= window_size:
            M_window.clear()
        M_window.append(M)

        if len(M_window) == window_size and max(M_window) - min(M_window) <= alpha:
            stop_N.append(N)
            candidate_id = um[:N, :N].mean(axis=1).argmax()
            ada_mbr.append(candidate[candidate_id])
            break

    else:
        stop_N.append(hypothesis_num)
        candidate_id = um.mean(axis=1).argmax()
        ada_mbr.append(candidate[candidate_id])


# ======================================================================
# --- save result ---
ada_output = f"{txt_dir}/QUIT-mbr-{args.hypo_nums}-(k={batch_size}|w={window_size}|alpha={alpha}).txt"
def oneline(s):
    LINE_SEPARATORS = re.compile(r"[\n\r\v\f\x1c-\x1e\x85\u2028\u2029]+")
    return LINE_SEPARATORS.sub(" ", s).strip()
with open(ada_output, "w", encoding="utf-8") as f:
    f.writelines([oneline(sentence)+"\n" for sentence in ada_mbr])

# ----- analyze -----
standard_utility_calls = sentence_num * hypothesis_num**2
ada_utility_calls = sum(N**2 for N in stop_N)
print("=" * 10, "analyze", "=" * 10)
print(f"utility calls ratio: {ada_utility_calls / standard_utility_calls:.4f}")

# tokens.shape = (sent_num, cand_num)
tokens = np.array([item["hypo_num_tokens"] for item in wmt_data])
standard_tokens = tokens.sum().item()
ada_tokens = sum(t[:N].sum().item() for N, t in zip(stop_N, tokens))
print(f"output token ratio: {ada_tokens / standard_tokens:.4f}")
print("---" * 3)

compute_analyze_result["standard_output_tokens"] = standard_tokens
compute_analyze_result["standard_utility_calls"] = standard_utility_calls
compute_analyze_result[f"k={batch_size}|w={window_size}|alpha={alpha}"] = {
    "window_size": window_size,
    "alpha": alpha,
    "utility_calls_ratio": ada_utility_calls / standard_utility_calls,
    "output_token_ratio": ada_tokens / standard_tokens,
}

with open(f"{args.dataset}-{short_model_name}-mbr-analyze.json", "w") as f:
    json.dump(compute_analyze_result, f)
