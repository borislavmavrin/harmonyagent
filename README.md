# Harmony Agent

First independent reproduction of OpenAI's published SWE Verified and AIME2025 scores for `gpt-oss-20b` with tools.

Harmony Agent encodes and decodes messages in `gpt-oss`'s native Harmony format, bypassing the lossy Chat Completions conversion. It also provides the model's in-distribution tools (`container.exec`, `repo_browser.*`, and `apply_patch`), which we reverse-engineered from the model's training priors.

## Results

| Benchmark | Published | HarmonyAgent | 95% CI |
|---|---|---|---|
| SWE Verified HIGH | 60.7% | 60.4% | [56.2%, 64.8%] |
| SWE Verified MEDIUM | 53.2% | 53.3% | [49.3%, 57.7%] |
| AIME 2025 MEDIUM w/ tools | 90.4% | 91.7% | [87.5%, 95.0%] |

## How to run

```bash
# Start vLLM server
docker run --ipc=host --gpus all --rm --memory 20g --cpus 6 -p 8000:8000 -v ~/.cache/:/root/.cache/ vllm/vllm-openai:v0.14.1-cu130 --model openai/gpt-oss-20b --tensor-parallel-size 1 --max-model-len 131072

# Set up environment
uv venv --python 3.12

# Run benchmarks
uv run python run_swe.py
uv run python run_aime2025.py
```

## Paper

[In harmony with gpt-oss arxiv](https://arxiv.org/abs/TODO)

[In harmony with gpt-oss](./paper.pdf)

## Citation

```bibtex
@article{mavrin2026harmony,
  title={In harmony with gpt-oss},
  author={Mavrin, Borislav},
  year={2026}
}
```
