# MT Evaluation

We evaluate selected translations with the reranker and four external metrics:

| Metric | Model / Implementation |
| --- | --- |
| Reranker (MBR) | [Unbabel/wmt22-comet-da](https://huggingface.co/Unbabel/wmt22-comet-da) |
| Reranker (QE) | [Unbabel/wmt22-cometkiwi-da](https://huggingface.co/Unbabel/wmt22-cometkiwi-da) |
| chrF++ | SacreBLEU |
| xCOMET | [Unbabel/XCOMET-XXL](https://huggingface.co/Unbabel/XCOMET-XXL) |
| MetricX | [google/metricx-24-hybrid-xl-v2p6](https://huggingface.co/google/metricx-24-hybrid-xl-v2p6) |
| GEMBA-MQM | [google/gemma-4-31B-it](https://huggingface.co/google/gemma-4-31B-it) |

## Usage

Inputs follow the [COMET format](../README.md#comet-data), with one selected translation per source. Run from this directory; use `-t hyp1.txt hyp2.txt` to evaluate multiple systems.

### Reranker Scores

Score selected translations against references with the MBR utility model:

```bash
comet-score \
  -s src.txt \
  -t hyp1.txt \
  -r ref.txt \
  --model Unbabel/wmt22-comet-da \
  --to_json evaluate-wmt25-hy-comet.json
```

For QE, no reference is needed:

```bash
comet-score \
  -s src.txt \
  -t hyp1.txt \
  --model Unbabel/wmt22-cometkiwi-da
```

### chrF++

```bash
python chrfpp_score.py \
    -s src.txt \
    -t hyp1.txt \
    -r ref.txt \
    --to_json evaluate-wmt25-hy-chrfpp.json
```

### xCOMET

```bash
comet-score \
  -s src.txt \
  -t hyp1.txt \
  -r ref.txt \
  --model Unbabel/XCOMET-XXL \
  --to_json evaluate-wmt25-hy-xcomet.json
```

### MetricX

Requires [MetricX](https://github.com/google-research/metricx), with `metricx24.predict` available in the Python environment.

```bash
python metricx_score.py \
    -s src.txt \
    -t hyp1.txt \
    -r ref.txt \
    --to_json evaluate-wmt25-hy-metricx.json
```

### GEMBA-MQM

We use a modified [GEMBA](https://github.com/MicrosoftTranslator/GEMBA) with concurrent requests. Configure its API endpoint and key, with the model served as `gemma`. `--wmt_data` provides language names and must align with the text files; no reference is needed.

```bash
python gemba_score.py \
    -s src.txt \
    -t hyp1.txt \
    --to_json evaluate-wmt25-hy-gemba-mqm.json \
    --wmt_data ../mt_data/wmt25-genmt-Hy-MT2-30B-A3B-512.jsonl \
    --workers 16
```
