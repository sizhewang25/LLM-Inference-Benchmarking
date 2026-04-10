# Background — Benchmark Refinement (2026-04-10)

## Why this work exists

A code review of the LLM-Inference-Benchmarking-2 harness surfaced four issues
that affect either the **robustness** or the **realism** of the FP16-vs-quantized
comparison. None block the harness from running, but each one can silently
distort results or break in subtle ways when models are swapped.

This task addresses three of them in code and captures the fourth as design
notes.

## The four findings

### 1. Tokenizer drift risk (`benchmark_gguf.py`)

[benchmark_gguf.py:53](../../benchmark_gguf.py#L53) reuses the base HF
tokenizer for the quantized model run. Today this is fine because
`benchmark_gguf.py` quantizes `Qwen2.5-7B-Instruct` itself and saves the same
tokenizer to `quant_path`. But if the quantization step ever:

- changes special tokens (`pad_token`, `bos_token`, etc.)
- ships a different chat template
- extends the vocab

…the tokenizer used at inference time will silently mismatch what the model
expects, producing garbage outputs or off-by-one alignment.

**Fix:** load `AutoTokenizer.from_pretrained(quant_path)` for the quant run.

### 2. Raw prompts on instruct models

Both [benchmark_gguf.py](../../benchmark_gguf.py) and
[benchmark_quantize.py](../../benchmark_quantize.py) feed raw `PROMPTS` strings
through `tokenizer(...)` without applying the chat template. The base models
in question (`Qwen2.5-7B-Instruct`, `DeepSeek-R1-Distill-Qwen-7B`) are
**instruct/chat tuned** and were trained against specific prompt formats.

DeepSeek-R1-Distill in particular is trained to emit `<think>` reasoning
traces and breaks without its template applied.

Realistic serving applies the template, so the current numbers under-report
the prefill length you'd see in production.

**Fix:** add a `format_prompts` helper in `bench_utils.py` that calls
`tokenizer.apply_chat_template(...)` if a template is set, and apply it in both
benchmark scripts symmetrically.

### 3. TPOT floor silently corrupts stats

[bench_utils.py:112](../../bench_utils.py#L112) uses
`max(new_tokens - 1, 1)` to guard against divide-by-zero. The side effect is
that if generation stops at 0 or 1 tokens, the computed TPOT equals the full
latency (because `decode_time / 1 = full latency`) and gets blended into the
mean with no warning.

**Fix:** Option A from the review — skip the prompt's TPOT contribution and
warn naming the prompt index. Surface a `tpot_skipped` count in the result
dict so `print_results` can flag it.

### 4. Batch-size strategy not documented

The PyTorch-vs-llama.cpp comparison has different sweet spots for batch=1
(latency-friendly, favors GGUF) vs batch=N (compute-bound, favors GPTQ
kernels). The harness only supports batch=1 today, and there is no record of
why or what the expansion plan should look like.

**Fix:** docs only — write `docs/batch-size-notes.md` capturing the reasoning
and the recommended sweep `[1, 4, 8, 16]`. No code change in this iteration.

## Intent

Robustness + realism, **not** new features. Out of scope:

- Implementing batching
- Refactoring [quantize.py](../../quantize.py) (currently orphaned)
- JSON/CSV result persistence
- argparse CLI front-end
- Choosing Option B (`min_new_tokens=max_new_tokens`) for TPOT — explicitly
  rejected; Option A only.
