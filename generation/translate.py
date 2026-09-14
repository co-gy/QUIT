from LLM import llm
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import json
import os

MAX_WORKERS = 512
hypothesis_num = 512
model = "Hy-MT2-30B-A3B"
path = f"wmt24-genmt-{model}-{hypothesis_num}.jsonl"

prompt = """Translate the following text into {target_lang}. Note that you should only output the translated result without any additional explanation:

{source_text}"""

if os.path.exists(path):
    with open(path, "r") as f:
        data_lst = [json.loads(line) for line in f]
    n = 0
    for item in data_lst:
        if item.get("result") is not None and len(item["result"]) == hypothesis_num:
            n += 1
    print("already", n)

with tqdm(total=sum(len(item["src_text"]) for item in data_lst)) as pbar:
    for idx, item in enumerate(data_lst):
        if item.get("result") is None:
            def translate(_):
                messages = [{"role": "user", "content": prompt.format(target_lang=item["tgt_lang"], source_text=item["src_text"])}]
                result = llm(messages, temperature=1.2)
                return result
            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                item["result"] = list(executor.map(translate, range(hypothesis_num)))
            with open(path, "w") as f:
                for item in data_lst:
                    json.dump(item, f, ensure_ascii=False)
                    f.write("\n")
        pbar.set_description(f"{idx}/{len(data_lst)}")
        pbar.update(len(item["src_text"]))
