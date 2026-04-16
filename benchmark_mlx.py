import logging
import time

from bench_utils import (
    PROMPTS,
    benchmark_mlx_model,
    dump_results,
    format_prompts,
    free_memory,
    print_comparison,
    print_results,
    setup_run_logging,
)

try:
    from mlx_lm import load as mlx_load
except ImportError as exc:
    raise SystemExit(
        "mlx-lm is required. Install with: pip install mlx-lm"
    ) from exc

log = logging.getLogger("bench")

# ── Configuration ──────────────────────────────────────────────────────────────
# FP16 baseline: standard HuggingFace repo; mlx_lm converts weights to MLX
# format on first load and caches the result in ~/.cache/huggingface/.
fp16_model_id  = "Qwen/Qwen2.5-0.5B-Instruct"
# MLX-quantized: pre-quantized INT4 model from the mlx-community org on Hub.
quant_model_id = "mlx-community/Qwen2.5-0.5B-Instruct-4bit"
num_samples    = 10
max_new_tokens = 128
warmup_runs    = 2


def main():
    run_dir = setup_run_logging(__file__)
    log.info("fp16 model  : %s", fp16_model_id)
    log.info("quant model : %s", quant_model_id)

    prompts = PROMPTS[:num_samples]

    # ── 1. Benchmark FP16 (MLX) ───────────────────────────────────────────────
    log.info("loading FP16 model via mlx_lm...")
    t0 = time.perf_counter()
    fp16_model, fp16_tokenizer = mlx_load(fp16_model_id)
    log.info("loaded in %.1fs", time.perf_counter() - t0)

    fp16_prompts = format_prompts(fp16_tokenizer, prompts)

    log.info("benchmarking FP16 model...")
    fp16_results = benchmark_mlx_model(
        fp16_model, fp16_tokenizer, fp16_prompts, max_new_tokens, warmup_runs
    )
    print_results("FP16 (MLX)", fp16_results)
    dump_results(run_dir, "FP16 (MLX)", fp16_results)

    del fp16_model, fp16_tokenizer
    free_memory()

    # ── 2. Benchmark MLX 4-bit ────────────────────────────────────────────────
    log.info("loading MLX-quantized model: %s", quant_model_id)
    t0 = time.perf_counter()
    quant_model, quant_tokenizer = mlx_load(quant_model_id)
    log.info("loaded in %.1fs", time.perf_counter() - t0)

    quant_prompts = format_prompts(quant_tokenizer, prompts)

    log.info("benchmarking MLX 4-bit model...")
    quant_results = benchmark_mlx_model(
        quant_model, quant_tokenizer, quant_prompts, max_new_tokens, warmup_runs
    )
    label = "MLX 4-bit (Quantized)"
    print_results(label, quant_results)
    dump_results(run_dir, label, quant_results)

    del quant_model, quant_tokenizer
    free_memory()

    # ── 3. Comparison ─────────────────────────────────────────────────────────
    print_comparison("FP16 (MLX)", fp16_results, "MLX-4bit", quant_results)


if __name__ == "__main__":
    main()
