# The datasets

The pipeline starts from released datasets: the six-peer streams, every peer's verified misleading answers, and the
misleading streams built from them. `manifest.json` lists every file with its sha256; `download.py` fetches them from a
Hugging Face dataset repository (`repo` in `manifest.json`, or `--repo`), checks every file and unpacks it into `data/`, the
layout the pipeline reads.

```bash
python datasets/download.py                                   # everything, into data/
python datasets/download.py --only capability_supported capability_challenging               # some datasets
python datasets/download.py --repo <user>/<dataset>           # from another copy of the release
```

| dataset (configs/datasets/) | file in the release | goes to |
|---|---|---|
| `address_fit` | `data/address_fit.jsonl.gz` | `data/address_fit/stream.jsonl` (17,709 events: GSM8K, SQuAD, APPS; six peers; used label-free, to fit the record's addresses) |
| `capability_supported` | `data/capability_supported.jsonl.gz` | `data/capability_supported/test.jsonl` (4,319 events: GSM8K test, SQuAD dev, APPS test with hidden tests) |
| `capability_challenging` | `data/capability_challenging.jsonl.gz` | `data/capability_challenging/test.jsonl` (17,403 events: PIQA, MMLU, OpenBookQA, SciQ, BBH, SuperGLUE) |
| `capability_supported_misleading`, `capability_challenging_misleading` | `answers/<dataset>.jsonl.gz` | `data/<dataset>/<peer>/`: every peer's misleading answer to every event |
| `capability_supported_misleading_p000` … `p100`, `capability_challenging_misleading_p000` … `p100` | `data/<dataset>.jsonl.gz` | `data/<dataset>/test.jsonl` + `manifest.json`: 0–100% of every peer's answers misleading |

The peers, in `peer_0` … `peer_5` order, are registered in `configs/peers/`: gemma-3-4b-it, Phi-4-mini-instruct,
Qwen2.5-Coder-7B-Instruct, Llama-3.1-8B-Instruct, DeepSeek-Coder-V2-Lite-Instruct, DeepSeek-R1-Distill-Qwen-7B.

A misleading dataset can also be rebuilt locally from its stream and answers (`PYTHONPATH=. python -m pipeline.streams
build <names>`); `python -m pipeline.streams digest <file>` compares it with the `digest` in the manifest. The MuSiQue
sub-task stream of the agent-team experiment is built from the benchmark's own file (`pipeline.team build`).
