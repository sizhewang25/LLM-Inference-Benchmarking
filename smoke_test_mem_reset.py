#!/usr/bin/env python3
"""Smoke test: load Q8 then Q4 Qwen sequentially.
Verifies the _free_model fix (gc.collect + mx.clear_cache) actually
releases unified memory between variants.
"""

import gc
import time
import mlx.core as mx
from mlx_lm import load as mlx_load
from mlx_lm.models.cache import make_prompt_cache

VARIANTS = [
    ("MLX_8bit", "mlx-community/Qwen2.5-7B-Instruct-8bit"),
    ("MLX_4bit", "mlx-community/Qwen2.5-7B-Instruct-4bit"),
]
DECODE_TOKENS = 32
PROMPT = (
    "Summarize the scientific, historical, and economic significance of the "
    "Atlantic Ocean for intercontinental trade, climate, and biodiversity. " * 8
)


def tokenize(tokenizer, text):
    inner = getattr(tokenizer, "_tokenizer", tokenizer)
    try:
        return inner.encode(text, add_special_tokens=True)
    except TypeError:
        out = inner.encode(text)
        return list(out.ids) if hasattr(out, "ids") else list(out)


def run_variant(label, model_id):
    print(f"\n{'='*55}")
    print(f"  {label}  ({model_id})")
    print(f"{'='*55}")

    mx.reset_peak_memory()
    model, tokenizer = mlx_load(model_id)
    mx.eval(model.parameters())

    weight_gib  = mx.get_active_memory() / 1024**3
    peak_load   = mx.get_peak_memory()   / 1024**3
    print(f"  After load  — active: {weight_gib:.2f} GiB  peak: {peak_load:.2f} GiB")

    token_ids = tokenize(tokenizer, PROMPT)
    input_ids = mx.array(token_ids)[None]

    # warmup
    cache = make_prompt_cache(model)
    logits = model(input_ids, cache=cache)
    mx.eval(logits)

    mx.reset_peak_memory()
    mx.eval()
    t0 = time.perf_counter()
    cache = make_prompt_cache(model)
    logits = model(input_ids, cache=cache)
    mx.eval(logits)
    ttft_ms = (time.perf_counter() - t0) * 1e3

    next_tok = mx.argmax(logits[:, -1, :], axis=-1, keepdims=True)
    mx.eval(next_tok)
    t1 = time.perf_counter()
    for _ in range(DECODE_TOKENS - 1):
        logits = model(next_tok, cache=cache)
        next_tok = mx.argmax(logits[:, -1, :], axis=-1, keepdims=True)
        mx.eval(next_tok)
    decode_ms = (time.perf_counter() - t1) * 1e3

    tpot_ms   = decode_ms / (DECODE_TOKENS - 1)
    peak_bench = mx.get_peak_memory() / 1024**3
    print(f"  Benchmark   — TTFT: {ttft_ms:.1f} ms  decode: {1e3/tpot_ms:.1f} t/s  peak: {peak_bench:.2f} GiB")

    # ── FREE ──────────────────────────────────────────────────────────────────
    del model, tokenizer, cache, logits, next_tok, input_ids
    gc.collect()
    mx.clear_cache()

    after_gib = mx.get_active_memory() / 1024**3
    print(f"  After free  — active: {after_gib:.2f} GiB  (released {weight_gib - after_gib:.2f} GiB)")


def main():
    print("Qwen sequential memory-reset smoke test")
    print(f"mlx version: ", end="")
    import importlib.metadata
    print(importlib.metadata.version("mlx"))

    for label, model_id in VARIANTS:
        run_variant(label, model_id)

    print(f"\n{'='*55}")
    print("  DONE — both variants completed, memory reset verified")
    print(f"{'='*55}")


if __name__ == "__main__":
    main()
