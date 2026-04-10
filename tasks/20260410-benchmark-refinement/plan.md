# Project Plan — Benchmark Refinement (2026-04-10)

## Goal

Make the FP16-vs-quantized comparison in this repo more **robust** (no
silent stat corruption, no tokenizer drift) and more **realistic** (chat
templates applied), and capture the unfinished batching design in docs.

## Scope

Code-level fixes to existing files plus two new documentation locations:

- `docs/` — design notes (batch sizing).
- `tasks/20260410-benchmark-refinement/` — this task's tracking docs.

No new modules. No new dependencies.

## Deliverables

| # | Deliverable | File(s) |
|---|---|---|
| 1 | Batch-size design notes | `docs/batch-size-notes.md` |
| 2 | Task tracking docs | `tasks/20260410-benchmark-refinement/{background,plan,todo,report}.md` |
| 3 | TPOT skip-and-warn (Option A) | `bench_utils.py` |
| 4 | Chat-template helper + use sites | `bench_utils.py`, `benchmark_gguf.py`, `benchmark_quantize.py` |
| 5 | Quant tokenizer load | `benchmark_gguf.py`, `benchmark_quantize.py` |

## Files touched

- [bench_utils.py](../../bench_utils.py) — `benchmark()`, `print_results()`, new `format_prompts()`.
- [benchmark_gguf.py](../../benchmark_gguf.py) — load `quant_tokenizer` from `quant_path`, format both prompt sets.
- [benchmark_quantize.py](../../benchmark_quantize.py) — try-load tokenizer from `quant_id`, format both prompt sets.

## Out of scope

- Batching implementation (only design notes in this iteration).
- Changes to [quantize.py](../../quantize.py) or [test.py](../../test.py).
- JSON/CSV persistence of results.
- argparse CLI.
- TPOT Option B (`min_new_tokens=max_new_tokens`) — explicitly rejected.

## Commit sequence

1. `docs: add batch-size notes for PyTorch vs llama.cpp comparison`
2. `docs(tasks): scaffold 20260410-benchmark-refinement task docs`
3. `fix(bench): skip TPOT when <2 tokens generated instead of masking`
4. `feat(bench): apply chat templates to instruct-model prompts`
5. `fix(bench): load tokenizer from quantized path to avoid drift`
6. `docs(tasks): mark benchmark-refinement todo items complete`

Each commit is independent and revertible.

## Risks

- **Chat-template prefill grows TTFT.** Numbers in `report.md` will not
  be directly comparable to the pre-change baseline. This is intentional —
  the new numbers are *more* realistic. Document the delta in the test
  report.
- **`AutoTokenizer.from_pretrained(quant_id)` may fail** for the GPTQ run if
  the ModelCloud repo doesn't ship a tokenizer. Wrap in try/except and fall
  back to base tokenizer with a warning.
- **Functional verification requires GPU + model weights.** Static checks
  (syntax, import) can run anywhere; full benchmark runs need the user's
  hardware.
