# Candidate Generation

We generate translation candidates with three LLM-based NMT models, served through vLLM's OpenAI-compatible API:

| Model | Checkpoint |
| --- | --- |
| Hy-MT2 | [tencent/Hy-MT2-30B-A3B](https://huggingface.co/tencent/Hy-MT2-30B-A3B) |
| TranslateGemma | [Infomaniak-AI/vllm-translategemma-12b-it](https://huggingface.co/Infomaniak-AI/vllm-translategemma-12b-it) |
| Qwen3 | [Qwen/Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B) |

## Prompt Templates

Hy-MT2 and Qwen3 use the Hy-MT2 template, with a target language name such as `German`:

```python
prompt = """Translate the following text into {target_lang}. Note that you should only output the translated result without any additional explanation:

{source_text}"""
```

TranslateGemma uses the following template, with source and target language codes:

```python
prompt = "<<<source>>>{source_lang}<<<target>>>{target_lang}<<<text>>>{source_text}"
```

Each prompt is sent as a single user message. `translate.py` currently implements the Hy-MT2/Qwen3 template; for TranslateGemma, update both the template and its `.format(...)` arguments to include `source_lang` and `target_lang`.

## Generation Settings

| Setting | Value |
| --- | --- |
| Candidate num per source | 512 |
| Temperature | 1.2 |
| `top_p` | 1 |
| `top_k` | -1 (disabled) |
| Maximum output tokens per candidate | 1024 |
| Maximum concurrent requests | 512 |

`LLM.py` wraps the API client and passes `enable_thinking=False` in the chat template settings. `translate.py` sends one request per candidate and processes source sentences sequentially.

## Usage

1. Serve the model with vLLM, then set `base_url`, `api_key`, and the served model name in `LLM.py`'s `llm = LLM(...)` configuration.
2. In `translate.py`, set the model, candidate num (`hypothesis_num`), input `path`, and prompt. The input JSONL must contain `src_text` and a target language name in `tgt_lang` (e.g., `German`), which is inserted directly into the prompt. Omit `result` or set it to `null` for sentences to generate.
3. Run from this directory:

   ```bash
   pip install openai tqdm
   python translate.py
   ```

Importing `LLM.py` also sends its example `hello` request and prints the response before generation starts.

Candidates are saved to `result` in the same JSONL file after each source sentence. Existing non-null `result` entries are skipped. For reranking, place the file in `../mt_data/` and follow the [main README](../README.md); add aligned `hypo_num_tokens` for Quit's token-cost analysis, as this script saves text only.
