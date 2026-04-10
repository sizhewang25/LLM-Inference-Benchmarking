# To-Do — Benchmark Refinement (2026-04-10)

## Docs

- [x] Write `docs/batch-size-notes.md` capturing PyTorch vs llama.cpp batching tradeoffs
- [x] Scaffold `tasks/20260410-benchmark-refinement/` with background, plan, todo, report

## Code

### `bench_utils.py`

- [x] Replace `decode_tokens = max(new_tokens - 1, 1)` with skip-and-warn (Option A)
  - [x] Track `tpot_skipped` counter
  - [x] Append to `all_ttft` / `all_latency` / `all_tokens` only when skipping
  - [x] Add `tpot_skipped` to returned dict
  - [x] Guard `mean(all_tpot)` / `stdev(all_tpot)` against empty list
- [x] Update `print_results` to display `tpot_skipped` when > 0
- [x] Add `format_prompts(tokenizer, prompts) -> list[str]` helper
  - [x] Returns templated strings if `tokenizer.chat_template` is set
  - [x] Falls back to raw prompts otherwise

### `benchmark_gguf.py`

- [x] After loading the quant model, load `quant_tokenizer = AutoTokenizer.from_pretrained(quant_path)`
- [x] Set `pad_token_id` on `quant_tokenizer` if missing
- [x] Apply `format_prompts(tokenizer, prompts)` for the FP16 run
- [x] Apply `format_prompts(quant_tokenizer, prompts)` for the GGUF run
- [x] Pass `quant_tokenizer` (not base `tokenizer`) into the GGUF `benchmark(...)` call

### `benchmark_quantize.py`

- [x] Apply `format_prompts(tokenizer, prompts)` for the FP16 run
- [x] Try-load tokenizer from `quant_id`, fall back to base on failure
- [x] Apply `format_prompts(quant_tokenizer, prompts)` for the GPTQ run
- [x] Pass the chosen tokenizer into the GPTQ `benchmark(...)` call

## Verification

- [x] Static syntax check on each edited file (`python -m py_compile`)
- [ ] (When GPU available) Run `python benchmark_gguf.py`, capture results in `report.md`
- [ ] (When GPU available) Run `python benchmark_quantize.py`, capture results in `report.md`
- [ ] Confirm DeepSeek-R1-Distill outputs include `<think>` reasoning trace (template applied correctly)
- [ ] Confirm no `tpot_skipped` warnings under normal operation
- [ ] Document TTFT delta vs pre-change baseline in `report.md`

## Sign-off

- [x] All commits land on main
- [ ] Test report filled out (pending GPU run)
