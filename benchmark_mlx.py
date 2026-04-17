import logging
import time

from bench_utils import (
    compute_perplexity_mlx,
    dump_results,
    dump_samples_csv,
    evaluate_gsm8k_mlx,
    free_memory,
    load_gsm8k_questions,
    load_wikitext2_tokens,
    print_comparison,
    print_results,
    setup_run_logging,
)

try:
    import mlx.core as mx
    from mlx_lm import load as mlx_load
except ImportError as exc:
    raise SystemExit(
        "mlx-lm is required. Install with: pip install mlx-lm"
    ) from exc

log = logging.getLogger("bench")

# ── Configuration ──────────────────────────────────────────────────────────────
MODEL_CONFIGS = [
    {
        "name": "Qwen2.5-0.5B-Instruct",
        "fp16_id": "Qwen/Qwen2.5-0.5B-Instruct",
        "quant_id": "mlx-community/Qwen2.5-0.5B-Instruct-4bit",
    },
    # {
    #     "name": "Llama-3.1-8B-Instruct",
    #     "fp16_id": "meta-llama/Llama-3.1-8B-Instruct",
    #     "quant_id": "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit",
    # },
    # {
    #     "name": "Qwen2.5-7B-Instruct",
    #     "fp16_id": "Qwen/Qwen2.5-7B-Instruct",
    #     "quant_id": "mlx-community/Qwen2.5-7B-Instruct-4bit",
    # },
    # {
    #     "name": "Gemma-2-9B-it",
    #     "fp16_id": "google/gemma-2-9b-it",
    #     "quant_id": "mlx-community/gemma-2-9b-it-4bit",
    # },
]

gsm8k_samples    = 50
gsm8k_max_tokens = 512
warmup_runs      = 2


def benchmark_model_pair(run_dir, config, gsm8k_questions):
    """Run FP16 vs MLX 4-bit benchmark for a single model config."""
    model_name = config["name"]
    fp16_id = config["fp16_id"]
    quant_id = config["quant_id"]

    log.info("=" * 60)
    log.info("MODEL: %s", model_name)
    log.info("=" * 60)

    # ── 1. Benchmark FP16 (MLX) ──────────────────────────────────────────
    log.info("loading FP16 model via mlx_lm: %s", fp16_id)
    mx.reset_peak_memory()
    t0 = time.perf_counter()
    fp16_model, fp16_tokenizer = mlx_load(fp16_id)
    mx.eval(fp16_model.parameters())
    weight_mem = mx.get_peak_memory() / 1024 / 1024
    log.info("loaded in %.1fs (weight mem: %.1f MB)", time.perf_counter() - t0, weight_mem)

    log.info("evaluating GSM8K (FP16 MLX)...")
    mx.reset_peak_memory()
    fp16_results, fp16_samples = evaluate_gsm8k_mlx(
        fp16_model, fp16_tokenizer, gsm8k_questions, gsm8k_max_tokens, warmup_runs
    )
    fp16_results["weight_mem_mb"] = weight_mem
    fp16_results["runtime_mem_mb"] = max(0, fp16_results["peak_mem_mb"] - weight_mem)
    log.info("FP16 MLX GSM8K accuracy: %.1f%%", fp16_results["gsm8k_accuracy"] * 100)

    log.info("computing perplexity (FP16 MLX)...")
    wikitext_tokens = load_wikitext2_tokens(fp16_tokenizer)
    fp16_results["perplexity"] = compute_perplexity_mlx(fp16_model, wikitext_tokens)
    log.info("FP16 MLX perplexity: %.2f", fp16_results["perplexity"])

    fp16_label = f"FP16 MLX ({model_name})"
    print_results(fp16_label, fp16_results)
    dump_results(run_dir, fp16_label, fp16_results)
    dump_samples_csv(run_dir, fp16_label, fp16_samples)

    del fp16_model, fp16_tokenizer
    free_memory()

    # ── 2. Benchmark MLX 4-bit ───────────────────────────────────────────
    log.info("loading MLX-quantized model: %s", quant_id)
    mx.reset_peak_memory()
    t0 = time.perf_counter()
    quant_model, quant_tokenizer = mlx_load(quant_id)
    mx.eval(quant_model.parameters())
    quant_weight_mem = mx.get_peak_memory() / 1024 / 1024
    log.info("loaded in %.1fs (weight mem: %.1f MB)", time.perf_counter() - t0, quant_weight_mem)

    log.info("evaluating GSM8K (MLX 4-bit)...")
    mx.reset_peak_memory()
    quant_results, quant_samples = evaluate_gsm8k_mlx(
        quant_model, quant_tokenizer, gsm8k_questions, gsm8k_max_tokens, warmup_runs
    )
    quant_results["weight_mem_mb"] = quant_weight_mem
    quant_results["runtime_mem_mb"] = max(0, quant_results["peak_mem_mb"] - quant_weight_mem)
    log.info("MLX 4-bit GSM8K accuracy: %.1f%%", quant_results["gsm8k_accuracy"] * 100)

    log.info("computing perplexity (MLX 4-bit)...")
    quant_wikitext_tokens = load_wikitext2_tokens(quant_tokenizer)
    quant_results["perplexity"] = compute_perplexity_mlx(quant_model, quant_wikitext_tokens)
    log.info("MLX 4-bit perplexity: %.2f", quant_results["perplexity"])

    quant_label = f"MLX 4-bit ({model_name})"
    print_results(quant_label, quant_results)
    dump_results(run_dir, quant_label, quant_results)
    dump_samples_csv(run_dir, quant_label, quant_samples)

    del quant_model, quant_tokenizer
    free_memory()

    # ── 3. Comparison ────────────────────────────────────────────────────
    print_comparison(f"FP16 MLX {model_name}", fp16_results, f"MLX-4bit {model_name}", quant_results)


def main():
    run_dir = setup_run_logging(__file__)
    log.info("models: %s", [c["name"] for c in MODEL_CONFIGS])

    gsm8k_questions = load_gsm8k_questions(num_samples=gsm8k_samples)
    log.info("loaded %d GSM8K questions", len(gsm8k_questions))

    for config in MODEL_CONFIGS:
        benchmark_model_pair(run_dir, config, gsm8k_questions)


if __name__ == "__main__":
    main()
