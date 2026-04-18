import logging
import os
import subprocess
import sys
import time

from bench_utils import (
    compute_perplexity_mlx,
    dump_results_table,
    evaluate_gsm8k_mlx,
    finalize_result,
    free_memory,
    load_gsm8k_questions,
    load_wikitext2_tokens,
    print_results_table,
    setup_run_logging,
)

try:
    import mlx.core as mx
    from mlx_lm import load as mlx_load
except ImportError as exc:
    raise SystemExit(
        "mlx-lm is required. Install with: pip install mlx-lm"
    ) from exc

from importlib.metadata import PackageNotFoundError, version as _pkg_version

log = logging.getLogger("bench")

# Script-level runtime identity — stamped onto every result.
FRAMEWORK = "MLX"
ENGINE = "mlx-lm"
try:
    ENGINE_VERSION = _pkg_version("mlx-lm")
except PackageNotFoundError:
    ENGINE_VERSION = None

# ── Configuration ──────────────────────────────────────────────────────────────
# Flat list: one entry per benchmark run. FP16 baselines are peers with quant
# variants — no pairing, no duplication when a base model has multiple quants.
RUN_CONFIGS = [
    # ── Qwen Family
    {
        "name": "Qwen2.5-7B-Instruct",
        "variant": "FP16",
        "quant_bits": 16,
        "model_id": "Qwen/Qwen2.5-7B-Instruct",
    },
    {
        "name": "Qwen2.5-7B-Instruct",
        "variant": "MLX 4-bit",
        "quant_method": "affine",
        "quant_bits": 4,
        "quant_format": "affine-gs64",
        "model_id": "mlx-community/Qwen2.5-7B-Instruct-4bit",
    },
    {
        "name": "Qwen2.5-7B-Instruct",
        "variant": "MLX 2-bit",
        "quant_method": "affine",
        "quant_bits": 2,
        "quant_format": "affine-gs64",
        "model_id": "models/Qwen2.5-7B-Instruct-mlx-2bit",
        "source_fp16": "Qwen/Qwen2.5-7B-Instruct",
    },

    # ── Llama Family
    {
        "name": "Llama-3.1-8B-Instruct",
        "variant": "FP16",
        "quant_bits": 16,
        "model_id": "meta-llama/Llama-3.1-8B-Instruct",
    },
    {
        "name": "Llama-3.1-8B-Instruct",
        "variant": "MLX 4-bit",
        "quant_method": "affine",
        "quant_bits": 4,
        "quant_format": "affine-gs64",
        "model_id": "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit",
    },
    {
        "name": "Llama-3.1-8B-Instruct",
        "variant": "MLX 2-bit",
        "quant_method": "affine",
        "quant_bits": 2,
        "quant_format": "affine-gs64",
        "model_id": "models/Llama-3.1-8B-Instruct-mlx-2bit",
        "source_fp16": "meta-llama/Llama-3.1-8B-Instruct",
    },

    # ── Gemma Family
    {
        "name": "Gemma-2-9B-it",
        "variant": "FP16",
        "quant_bits": 16,
        "model_id": "google/gemma-2-9b-it",
    },
    {
        "name": "Gemma-2-9B-it",
        "variant": "MLX 4-bit",
        "quant_method": "affine",
        "quant_bits": 4,
        "quant_format": "affine-gs64",
        "model_id": "mlx-community/gemma-2-9b-it-4bit",
    },
    {
        "name": "Gemma-2-9B-it",
        "variant": "MLX 2-bit",
        "quant_method": "affine",
        "quant_bits": 2,
        "quant_format": "affine-gs64",
        "model_id": "models/Gemma-2-9B-it-mlx-2bit",
        "source_fp16": "google/gemma-2-9b-it",
    },
]

gsm8k_samples    = 10
gsm8k_max_tokens = 512
warmup_runs      = 2


def _ensure_local_quant(model_id, source_fp16, quant_bits):
    """Self-quantize `source_fp16` to `model_id` via mlx_lm.convert if missing."""
    if os.path.isdir(model_id):
        return
    log.info("self-quantizing %s to %d-bit at %s...", source_fp16, quant_bits, model_id)
    subprocess.run([
        sys.executable, "-m", "mlx_lm", "convert",
        "--hf-path", source_fp16,
        "-q",
        "--q-bits", str(quant_bits),
        "--q-group-size", "64",
        "--mlx-path", model_id,
    ], check=True)
    log.info("self-quantization complete: %s", model_id)


def benchmark_run(run_dir, config, gsm8k_questions):
    """Run GSM8K + WikiText-2 PPL for a single model variant.

    Returns the augmented results dict on success, or None on failure (e.g. OOM).
    """
    name = config["name"]
    variant = config["variant"]
    model_id = config["model_id"]
    label = f"{variant} ({name})"

    log.info("=" * 60)
    log.info("RUN: %s", label)
    log.info("=" * 60)

    if config.get("source_fp16") and model_id.startswith(("models/", "/", ".")):
        _ensure_local_quant(model_id, config["source_fp16"], config["quant_bits"])

    model = tokenizer = None
    try:
        log.info("loading model: %s", model_id)
        mx.reset_peak_memory()
        t0 = time.perf_counter()
        model, tokenizer = mlx_load(model_id)
        mx.eval(model.parameters())
        weight_mem = mx.get_peak_memory() / 1024 / 1024
        log.info("loaded in %.1fs (weight mem: %.1f MB)", time.perf_counter() - t0, weight_mem)

        log.info("evaluating GSM8K...")
        mx.reset_peak_memory()
        results, samples = evaluate_gsm8k_mlx(
            model, tokenizer, gsm8k_questions, gsm8k_max_tokens, warmup_runs
        )
        log.info("GSM8K accuracy: %.1f%%", results["gsm8k_accuracy"] * 100)

        log.info("computing perplexity...")
        wikitext_tokens = load_wikitext2_tokens(tokenizer)
        results["perplexity"] = compute_perplexity_mlx(model, wikitext_tokens)
        log.info("perplexity: %.2f", results["perplexity"])

        return finalize_result(
            run_dir, label, results, samples, name, variant, weight_mem,
            framework=FRAMEWORK,
            engine=ENGINE,
            engine_version=ENGINE_VERSION,
            quant_method=config.get("quant_method"),
            quant_bits=config.get("quant_bits"),
            quant_format=config.get("quant_format"),  # e.g. "affine-gs64" (MLX has no GGUF-style preset names)
            kernel="mlx",                              # mlx_lm's built-in matmul/dequant path
        )
    except Exception as e:
        log.warning("run failed (likely OOM): %s — skipping", e)
        return None
    finally:
        model = None
        tokenizer = None
        free_memory()


def _parse_cli(args):
    """Parse optional index args; no args → run all configs."""
    if not args:
        return RUN_CONFIGS
    return [RUN_CONFIGS[int(a)] for a in args]


def main():
    run_dir = setup_run_logging(__file__)
    configs = _parse_cli(sys.argv[1:])

    log.info("runs: %s", [f"{c['variant']} {c['name']}" for c in configs])

    gsm8k_questions = load_gsm8k_questions(num_samples=gsm8k_samples)
    log.info("loaded %d GSM8K questions", len(gsm8k_questions))

    rows = []
    for cfg in configs:
        result = benchmark_run(run_dir, cfg, gsm8k_questions)
        if result is not None:
            rows.append(result)

    if rows:
        print_results_table(rows)
        dump_results_table(run_dir, rows)
    else:
        log.warning("no successful runs to summarize")


if __name__ == "__main__":
    main()
