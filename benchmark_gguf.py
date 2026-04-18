import logging
import os
import sys
import time

import torch
import transformers
from transformers import AutoTokenizer

from gptqmodel import BACKEND, GPTQModel
from gptqmodel.quantization import GGUFConfig
from bench_utils import (
    compute_perplexity,
    dump_results_table,
    evaluate_gsm8k,
    finalize_result,
    free_memory,
    get_peak_memory_mb,
    load_gsm8k_questions,
    load_wikitext2_tokens,
    pick_device,
    print_results_table,
    reset_peak_memory,
    setup_run_logging,
    sync_device,
)

log = logging.getLogger("bench")

# Script-level runtime identity — stamped onto every result.
# Note: gptqmodel is the quant *toolkit* that dispatches kernels; the
# generation loop itself runs inside HuggingFace transformers, so that's the
# true `engine`. The kernel field captures gptqmodel's GGUF_TRITON / GGUF_TORCH
# dispatch choice.
FRAMEWORK = "PyTorch"
ENGINE = "transformers"
ENGINE_VERSION = transformers.__version__

# ── Configuration ──────────────────────────────────────────────────────────────
# Flat list: one entry per benchmark run. FP16 baselines are peers with GGUF
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
        "variant": "GGUF Q4_K_M",
        "quant_method": "k-quant",
        "quant_bits": 4,
        "quant_format": "Q4_K_M",
        "model_id": "models/Qwen2.5-7B-Instruct-GGUF-Q4_K_M",
        "gguf_format": "q4_k_m",
        "source_fp16": "Qwen/Qwen2.5-7B-Instruct",
    },
    {
        "name": "Qwen2.5-7B-Instruct",
        "variant": "GGUF Q2_K",
        "quant_method": "k-quant",
        "quant_bits": 2,
        "quant_format": "Q2_K",
        "model_id": "models/Qwen2.5-7B-Instruct-GGUF-Q2_K",
        "gguf_format": "q2_k",
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
        "variant": "GGUF Q4_K_M",
        "quant_method": "k-quant",
        "quant_bits": 4,
        "quant_format": "Q4_K_M",
        "model_id": "models/Llama-3.1-8B-Instruct-GGUF-Q4_K_M",
        "gguf_format": "q4_k_m",
        "source_fp16": "meta-llama/Llama-3.1-8B-Instruct",
    },
    {
        "name": "Llama-3.1-8B-Instruct",
        "variant": "GGUF Q2_K",
        "quant_method": "k-quant",
        "quant_bits": 2,
        "quant_format": "Q2_K",
        "model_id": "models/Llama-3.1-8B-Instruct-GGUF-Q2_K",
        "gguf_format": "q2_k",
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
        "variant": "GGUF Q4_K_M",
        "quant_method": "k-quant",
        "quant_bits": 4,
        "quant_format": "Q4_K_M",
        "model_id": "models/Gemma-2-9B-it-GGUF-Q4_K_M",
        "gguf_format": "q4_k_m",
        "source_fp16": "google/gemma-2-9b-it",
    },
    {
        "name": "Gemma-2-9B-it",
        "variant": "GGUF Q2_K",
        "quant_method": "k-quant",
        "quant_bits": 2,
        "quant_format": "Q2_K",
        "model_id": "models/Gemma-2-9B-it-GGUF-Q2_K",
        "gguf_format": "q2_k",
        "source_fp16": "google/gemma-2-9b-it",
    },
]

gsm8k_samples    = 10
gsm8k_max_tokens = 512
warmup_runs      = 2
device = pick_device()


def _ensure_local_quant(model_id, source_fp16, gguf_format, gguf_backend):
    """Quantize `source_fp16` to GGUF `model_id` via GPTQModel if the local
    path doesn't exist yet. Mirrors _ensure_local_quant in benchmark_mlx.py."""
    if os.path.isdir(model_id):
        return
    log.info("quantizing %s to GGUF %s at %s...", source_fp16, gguf_format, model_id)
    bits = 2 if "q2" in gguf_format else 4
    qconfig = GGUFConfig(bits=bits, format=gguf_format)
    tokenizer = AutoTokenizer.from_pretrained(source_fp16)
    if not tokenizer.pad_token_id:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model = GPTQModel.load(source_fp16, qconfig)
    t0 = time.perf_counter()
    model.quantize(calibration=None, tokenizer=tokenizer, backend=gguf_backend)
    log.info("quantization took %.1fs", time.perf_counter() - t0)
    model.save(model_id)
    tokenizer.save_pretrained(model_id)
    del model
    free_memory()
    log.info("quantization complete: %s", model_id)


def benchmark_run(run_dir, config, gsm8k_questions, gguf_backend):
    """Run GSM8K + WikiText-2 PPL for a single model variant (FP16 or GGUF).

    Returns the augmented results dict on success, or None on failure (OOM etc.).
    """
    name = config["name"]
    variant = config["variant"]
    model_id = config["model_id"]
    label = f"{variant} ({name})"
    is_gguf = "gguf_format" in config

    log.info("=" * 60)
    log.info("RUN: %s", label)
    log.info("=" * 60)

    if is_gguf and model_id.startswith(("models/", "/", ".")):
        _ensure_local_quant(model_id, config["source_fp16"], config["gguf_format"], gguf_backend)

    model = None
    try:
        log.info("loading model: %s", model_id)
        reset_peak_memory(device)
        t0 = time.perf_counter()
        if is_gguf:
            model = GPTQModel.load(model_id, backend=gguf_backend, device=device)
            # MPS needs explicit .to(device) — GPTQModel leaves some layers on CPU.
            # On CUDA, GPTQModel handles placement via device_map; skip to avoid conflicts.
            if str(device).startswith("mps") and hasattr(model, "model"):
                model.model.to(device)
        else:
            model = GPTQModel.load(model_id, device=device)
        sync_device(device)
        weight_mem = get_peak_memory_mb(device)
        log.info("loaded in %.1fs (weight mem: %.1f MB)", time.perf_counter() - t0, weight_mem)

        tokenizer = AutoTokenizer.from_pretrained(model_id)
        if not tokenizer.pad_token_id:
            tokenizer.pad_token_id = tokenizer.eos_token_id

        log.info("evaluating GSM8K...")
        reset_peak_memory(device)
        results, samples = evaluate_gsm8k(
            model, tokenizer, gsm8k_questions, gsm8k_max_tokens, warmup_runs, device
        )
        log.info("GSM8K accuracy: %.1f%%", results["gsm8k_accuracy"] * 100)

        log.info("computing perplexity...")
        wikitext_tokens = load_wikitext2_tokens(tokenizer)
        results["perplexity"] = compute_perplexity(model, wikitext_tokens, device)
        log.info("perplexity: %.2f", results["perplexity"])

        if is_gguf:
            kernel = "triton" if gguf_backend == BACKEND.GGUF_TRITON else "torch"
        else:
            kernel = "torch"  # FP16 baseline: stock PyTorch matmul via GPTQModel.load

        return finalize_result(
            run_dir, label, results, samples, name, variant, weight_mem,
            framework=FRAMEWORK,
            engine=ENGINE,
            engine_version=ENGINE_VERSION,
            quant_method=config.get("quant_method"),
            quant_bits=config.get("quant_bits"),
            quant_format=config.get("quant_format"),
            kernel=kernel,
        )
    except torch.cuda.OutOfMemoryError as e:
        log.warning("run failed (CUDA OOM): %s — skipping", e)
        return None
    except Exception as e:
        log.warning("run failed: %s — skipping", e)
        return None
    finally:
        model = None
        free_memory()


def _parse_cli(args):
    """Parse optional index args; no args → run all configs."""
    if not args:
        return RUN_CONFIGS
    return [RUN_CONFIGS[int(a)] for a in args]


def main():
    run_dir = setup_run_logging(__file__)
    configs = _parse_cli(sys.argv[1:])

    log.info("device: %s", device)
    log.info("runs: %s", [f"{c['variant']} {c['name']}" for c in configs])

    gguf_backend = BACKEND.GGUF_TRITON if str(device).startswith("cuda") else BACKEND.GGUF_TORCH
    log.info("gguf backend: %s", gguf_backend)

    gsm8k_questions = load_gsm8k_questions(num_samples=gsm8k_samples)
    log.info("loaded %d GSM8K questions", len(gsm8k_questions))

    rows = []
    for cfg in configs:
        result = benchmark_run(run_dir, cfg, gsm8k_questions, gguf_backend)
        if result is not None:
            rows.append(result)

    if rows:
        print_results_table(rows)
        dump_results_table(run_dir, rows)
    else:
        log.warning("no successful runs to summarize")


if __name__ == "__main__":
    main()
