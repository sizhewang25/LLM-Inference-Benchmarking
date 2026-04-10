# Test Report — Benchmark Refinement (2026-04-10)

## Environment

| Field | Value |
|---|---|
| Date run | _TBD_ |
| Operator | _TBD_ |
| GPU | _TBD_ |
| CUDA / driver | _TBD_ |
| PyTorch version | _TBD_ |
| `gptqmodel` version | _TBD_ |
| `transformers` version | _TBD_ |

## Static checks

- [ ] `python -m py_compile bench_utils.py benchmark_gguf.py benchmark_quantize.py`

## Commands run

```sh
# (fill in actual invocations)
python benchmark_gguf.py 2>&1 | tee tasks/20260410-benchmark-refinement/gguf-run.log
python benchmark_quantize.py 2>&1 | tee tasks/20260410-benchmark-refinement/quantize-run.log
```

## Results — `benchmark_gguf.py` (Qwen2.5-7B-Instruct)

### FP16

| Metric | Value |
|---|---|
| Throughput (tok/s) | _TBD_ |
| TTFT mean (ms) | _TBD_ |
| TPOT mean (ms) | _TBD_ |
| Latency mean (ms) | _TBD_ |
| Peak memory (MB) | _TBD_ |
| TPOT skipped | _TBD_ |

### GGUF Q4_K_M

| Metric | Value |
|---|---|
| Throughput (tok/s) | _TBD_ |
| TTFT mean (ms) | _TBD_ |
| TPOT mean (ms) | _TBD_ |
| Latency mean (ms) | _TBD_ |
| Peak memory (MB) | _TBD_ |
| TPOT skipped | _TBD_ |

### Comparison

| Metric | FP16 | GGUF | Delta |
|---|---|---|---|
| Throughput | _TBD_ | _TBD_ | _TBD_ |
| TTFT | _TBD_ | _TBD_ | _TBD_ |
| TPOT | _TBD_ | _TBD_ | _TBD_ |
| Latency | _TBD_ | _TBD_ | _TBD_ |
| Peak memory | _TBD_ | _TBD_ | _TBD_ |

Speedup: _TBD_ × | Memory saved: _TBD_ %

## Results — `benchmark_quantize.py` (DeepSeek-R1-Distill-Qwen-7B)

### FP16

| Metric | Value |
|---|---|
| Throughput (tok/s) | _TBD_ |
| TTFT mean (ms) | _TBD_ |
| TPOT mean (ms) | _TBD_ |
| Latency mean (ms) | _TBD_ |
| Peak memory (MB) | _TBD_ |
| TPOT skipped | _TBD_ |

### GPTQ 4-bit

| Metric | Value |
|---|---|
| Throughput (tok/s) | _TBD_ |
| TTFT mean (ms) | _TBD_ |
| TPOT mean (ms) | _TBD_ |
| Latency mean (ms) | _TBD_ |
| Peak memory (MB) | _TBD_ |
| TPOT skipped | _TBD_ |

### Comparison

| Metric | FP16 | GPTQ | Delta |
|---|---|---|---|
| Throughput | _TBD_ | _TBD_ | _TBD_ |
| TTFT | _TBD_ | _TBD_ | _TBD_ |
| TPOT | _TBD_ | _TBD_ | _TBD_ |
| Latency | _TBD_ | _TBD_ | _TBD_ |
| Peak memory | _TBD_ | _TBD_ | _TBD_ |

Speedup: _TBD_ × | Memory saved: _TBD_ %

## Pre-change vs post-change TTFT delta

The chat-template change adds prefill tokens. Document the difference here so
future runs aren't compared apples-to-oranges with the pre-template baseline.

| Script | TTFT before (ms) | TTFT after (ms) | Δ |
|---|---|---|---|
| `benchmark_gguf.py` FP16 | _TBD_ | _TBD_ | _TBD_ |
| `benchmark_gguf.py` GGUF | _TBD_ | _TBD_ | _TBD_ |
| `benchmark_quantize.py` FP16 | _TBD_ | _TBD_ | _TBD_ |
| `benchmark_quantize.py` GPTQ | _TBD_ | _TBD_ | _TBD_ |

## Anomalies

- _Note any TPOT skips, OOMs, output-quality regressions, etc._

## Sample output spot-check

- [ ] DeepSeek-R1-Distill output includes `<think>` reasoning block (chat template applied)
- [ ] Qwen2.5-Instruct output uses ChatML format markers correctly

## Sign-off

- [ ] All six commits landed
- [ ] Static checks pass
- [ ] Functional runs match expectations
- [ ] Operator: ____________  Date: ____________
