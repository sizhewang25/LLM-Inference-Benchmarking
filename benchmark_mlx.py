import logging
import time

from bench_utils import (
    compute_perplexity_mlx,
    dump_results,
    evaluate_gsm8k_mlx,
    free_memory,
    load_gsm8k_questions,
    load_wikitext2_tokens,
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
gsm8k_samples    = 100
gsm8k_max_tokens = 512
warmup_runs      = 2


def main():
    run_dir = setup_run_logging(__file__)
    log.info("fp16 model  : %s", fp16_model_id)
    log.info("quant model : %s", quant_model_id)

    gsm8k_questions = load_gsm8k_questions(num_samples=gsm8k_samples)
    log.info("loaded %d GSM8K questions", len(gsm8k_questions))

    # ── 1. Benchmark FP16 (MLX) ───────────────────────────────────────────────
    log.info("loading FP16 model via mlx_lm...")
    t0 = time.perf_counter()
    fp16_model, fp16_tokenizer = mlx_load(fp16_model_id)
    log.info("loaded in %.1fs", time.perf_counter() - t0)

    log.info("evaluating GSM8K (FP16 MLX)...")
    fp16_results = evaluate_gsm8k_mlx(
        fp16_model, fp16_tokenizer, gsm8k_questions, gsm8k_max_tokens, warmup_runs
    )
    log.info("FP16 MLX GSM8K accuracy: %.1f%%", fp16_results["gsm8k_accuracy"] * 100)

    log.info("computing perplexity (FP16 MLX)...")
    wikitext_tokens = load_wikitext2_tokens(fp16_tokenizer)
    fp16_results["perplexity"] = compute_perplexity_mlx(fp16_model, wikitext_tokens)
    log.info("FP16 MLX perplexity: %.2f", fp16_results["perplexity"])

    print_results("FP16 (MLX)", fp16_results)
    dump_results(run_dir, "FP16 (MLX)", fp16_results)

    del fp16_model, fp16_tokenizer
    free_memory()

    # ── 2. Benchmark MLX 4-bit ────────────────────────────────────────────────
    log.info("loading MLX-quantized model: %s", quant_model_id)
    t0 = time.perf_counter()
    quant_model, quant_tokenizer = mlx_load(quant_model_id)
    log.info("loaded in %.1fs", time.perf_counter() - t0)

    log.info("evaluating GSM8K (MLX 4-bit)...")
    quant_results = evaluate_gsm8k_mlx(
        quant_model, quant_tokenizer, gsm8k_questions, gsm8k_max_tokens, warmup_runs
    )
    log.info("MLX 4-bit GSM8K accuracy: %.1f%%", quant_results["gsm8k_accuracy"] * 100)

    log.info("computing perplexity (MLX 4-bit)...")
    quant_wikitext_tokens = load_wikitext2_tokens(quant_tokenizer)
    quant_results["perplexity"] = compute_perplexity_mlx(quant_model, quant_wikitext_tokens)
    log.info("MLX 4-bit perplexity: %.2f", quant_results["perplexity"])

    label = "MLX 4-bit (Quantized)"
    print_results(label, quant_results)
    dump_results(run_dir, label, quant_results)

    del quant_model, quant_tokenizer
    free_memory()

    # ── 3. Comparison ─────────────────────────────────────────────────────────
    print_comparison("FP16 (MLX)", fp16_results, "MLX-4bit", quant_results)


if __name__ == "__main__":
    main()
