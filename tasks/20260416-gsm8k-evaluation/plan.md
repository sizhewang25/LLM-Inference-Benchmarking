# GSM8K Evaluation — Plan

## Background

The current benchmarks use 15 manual prompts for speed measurement and WikiText-2
for perplexity. This has two gaps:

1. **Statistical rigor** — only 10 of 15 manual prompts are used per run, too few
   for meaningful confidence intervals.
2. **No task accuracy metric** — perplexity measures language modeling quality but
   doesn't tell you whether quantization breaks the model's ability to reason
   correctly.

GSM8K (1,319 grade-school math problems from OpenAI) solves both: the model
generates 50–200 tokens of step-by-step reasoning per question, giving meaningful
speed metrics, and the final answer is a number that's trivially scored for
accuracy.

## Context

- **Dataset:** `openai/gsm8k` on HuggingFace — 1,319 test questions, 2 columns
  (`question`, `answer`). Answers end with `#### <number>`.
- **Loading pattern:** `hf_hub_download` + `pyarrow.parquet` — same as the existing
  `load_wikitext2_tokens()` in `bench_utils.py`, avoids the `datasets` dependency.
- **Existing infra:** `benchmark()` and `benchmark_mlx_model()` handle speed timing;
  `compute_perplexity()` / `compute_perplexity_mlx()` handle quality; all three
  benchmark scripts (`benchmark_gguf.py`, `benchmark_quantize.py`, `benchmark_mlx.py`)
  already integrate both.
- **Reasoning models:** QwQ-32B and similar models output `\boxed{number}` instead
  of `#### number` — the answer parser must handle both formats.

## Goals

1. Add `evaluate_gsm8k()` and `evaluate_gsm8k_mlx()` to `bench_utils.py` that
   return speed metrics + accuracy in a single pass.
2. Replace the current `benchmark()` / `benchmark_mlx_model()` calls in all three
   benchmark scripts with the new GSM8K evaluation functions.
3. Update `print_results()` and `print_comparison()` to display GSM8K accuracy.
4. Preserve existing perplexity (WikiText-2) as a complementary quality signal.

## Approach

- **New functions in `bench_utils.py`:**
  - `load_gsm8k_questions(num_samples=100, seed=42)` — load and parse dataset
  - `parse_gsm8k_answer(model_output)` — extract number from `####`, `\boxed{}`,
    or last-number fallback
  - `evaluate_gsm8k(model, tokenizer, questions, max_new_tokens, warmup_runs, device)`
    — mirrors `benchmark()` timing structure + accuracy scoring
  - `evaluate_gsm8k_mlx(...)` — MLX counterpart
  - `GSM8K_INSTRUCTION` constant — instruction prefix for prompts
- **Script updates:** swap `benchmark()` → `evaluate_gsm8k()` in each script,
  add `gsm8k_samples` and `gsm8k_max_tokens` config vars.
- **Old functions preserved:** `benchmark()` and `benchmark_mlx_model()` stay in
  `bench_utils.py` for backwards compatibility.

## Caveats

- `max_new_tokens` should be 512 for GSM8K (reasoning chains can be verbose);
  the current scripts use 128.
- The 0.5B model may score poorly on GSM8K — that's expected and fine for
  validating the harness. Real accuracy comparisons need the 7B+ models.
- `mlx_lm` `response.text` accumulation behavior needs verification at runtime —
  may need to concatenate chunks if it's per-token rather than cumulative.
- Fixed seed (42) ensures reproducible question subsets across runs, which is
  critical for fair FP16-vs-quantized comparisons.
