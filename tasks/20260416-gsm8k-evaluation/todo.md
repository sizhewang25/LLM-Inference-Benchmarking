# GSM8K Evaluation — Todo

## Phase 0: Data Loading & Parsing

- [ ] Add `load_gsm8k_questions(num_samples, seed)` to `bench_utils.py`
  - [ ] Download via `hf_hub_download` + `pyarrow.parquet`
  - [ ] Parse `#### <number>` from answer column, strip commas, return float
  - [ ] Reproducible random subset via fixed seed
- [ ] Add `parse_gsm8k_answer(model_output)` to `bench_utils.py`
  - [ ] Priority 1: `#### <number>` regex
  - [ ] Priority 2: `\boxed{<number>}` regex (reasoning models)
  - [ ] Priority 3: last number in text (fallback)
- [ ] Add `GSM8K_INSTRUCTION` constant to `bench_utils.py`

## Phase 1: Evaluation Functions

- [ ] Add `evaluate_gsm8k(model, tokenizer, questions, max_new_tokens, warmup_runs, device)` to `bench_utils.py`
  - [ ] Format prompts: `GSM8K_INSTRUCTION + question` → `format_prompts()`
  - [ ] Mirror `benchmark()` timing (warmup, TTFT via streamer, TPOT, sync_device)
  - [ ] Decode generated tokens to text, score via `parse_gsm8k_answer()`
  - [ ] Return 11-key speed dict + `gsm8k_accuracy`, `gsm8k_correct`, `gsm8k_total`
- [ ] Add `evaluate_gsm8k_mlx(model, tokenizer, questions, max_new_tokens, warmup_runs)` to `bench_utils.py`
  - [ ] Mirror `benchmark_mlx_model()` timing (stream_generate, mx.eval)
  - [ ] Collect output text from stream, score accuracy
  - [ ] Same return dict format

## Phase 2: Display Updates

- [ ] Update `print_results()` to show GSM8K accuracy when present
- [ ] Update `print_comparison()` to show GSM8K accuracy delta when present

## Phase 3: Script Integration

- [ ] Update `benchmark_gguf.py`: replace `benchmark()` with `evaluate_gsm8k()`, add config vars
- [ ] Update `benchmark_quantize.py`: same changes
- [ ] Update `benchmark_mlx.py`: replace `benchmark_mlx_model()` with `evaluate_gsm8k_mlx()`, add config vars
- [ ] Ensure `load_gsm8k_questions()` called once, reused for both FP16 and quantized runs

## Phase 4: Verification

- [ ] Static syntax check on all modified files
- [ ] Smoke test `benchmark_mlx.py` with `gsm8k_samples=5`
- [ ] Confirm GSM8K accuracy appears in printed output and JSON dump
- [ ] Confirm speed metrics (TTFT/TPOT/throughput) are reasonable
- [ ] Confirm perplexity still works alongside GSM8K
