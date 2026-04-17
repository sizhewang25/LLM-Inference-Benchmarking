import logging
import os
import time

import torch
from transformers import AutoTokenizer

from gptqmodel import BACKEND, GPTQModel
from gptqmodel.quantization import GGUFConfig
from bench_utils import (
    compute_perplexity,
    dump_results,
    dump_samples_csv,
    evaluate_gsm8k,
    free_memory,
    get_peak_memory_mb,
    load_gsm8k_questions,
    load_wikitext2_tokens,
    pick_device,
    print_comparison,
    print_results,
    reset_peak_memory,
    setup_run_logging,
    sync_device,
)

log = logging.getLogger("bench")

# ── Configuration ──────────────────────────────────────────────
MODEL_CONFIGS = [
    {
        "name": "Qwen2.5-0.5B-Instruct",
        "fp16_id": "Qwen/Qwen2.5-0.5B-Instruct",
        "quant_path": "models/Qwen2.5-0.5B-Instruct-GGUF-Q4_K_M",
    },
    # {
    #     "name": "Llama-3.1-8B-Instruct",
    #     "fp16_id": "meta-llama/Llama-3.1-8B-Instruct",
    #     "quant_path": "models/Llama-3.1-8B-Instruct-GGUF-Q4_K_M",
    # },
    # {
    #     "name": "Qwen2.5-7B-Instruct",
    #     "fp16_id": "Qwen/Qwen2.5-7B-Instruct",
    #     "quant_path": "models/Qwen2.5-7B-Instruct-GGUF-Q4_K_M",
    # },
    # {
    #     "name": "Gemma-2-9B-it",
    #     "fp16_id": "google/gemma-2-9b-it",
    #     "quant_path": "models/Gemma-2-9B-it-GGUF-Q4_K_M",
    # },
]

gguf_format      = "q4_k_m"
gsm8k_samples    = 50
gsm8k_max_tokens = 512
warmup_runs      = 2
device = pick_device()


def get_weight_memory_mb(device):
    """Snapshot current memory as a proxy for model weight memory."""
    from bench_utils import sync_device as _sync, get_peak_memory_mb as _get
    _sync(device)
    return _get(device)


def benchmark_model_pair(run_dir, config, gsm8k_questions, gguf_backend):
    """Run FP16 vs GGUF Q4_K_M benchmark for a single model config."""
    model_name = config["name"]
    fp16_id = config["fp16_id"]
    quant_path = config["quant_path"]

    log.info("=" * 60)
    log.info("MODEL: %s", model_name)
    log.info("=" * 60)

    tokenizer = AutoTokenizer.from_pretrained(fp16_id)
    if not tokenizer.pad_token_id:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # ── 1. Benchmark FP16 via GPTQModel ──────────────────────────────────
    log.info("loading FP16 model via GPTQModel: %s", fp16_id)
    reset_peak_memory(device)
    t0 = time.perf_counter()
    fp16_model = GPTQModel.load(fp16_id, device=device)
    sync_device(device)
    weight_mem = get_peak_memory_mb(device)
    log.info("loaded in %.1fs (weight mem: %.1f MB)", time.perf_counter() - t0, weight_mem)

    log.info("evaluating GSM8K (FP16)...")
    reset_peak_memory(device)
    fp16_results, fp16_samples = evaluate_gsm8k(fp16_model, tokenizer, gsm8k_questions, gsm8k_max_tokens, warmup_runs, device)
    fp16_results["weight_mem_mb"] = weight_mem
    fp16_results["runtime_mem_mb"] = max(0, fp16_results["peak_mem_mb"] - weight_mem)
    log.info("FP16 GSM8K accuracy: %.1f%%", fp16_results["gsm8k_accuracy"] * 100)

    log.info("computing perplexity (FP16)...")
    wikitext_tokens = load_wikitext2_tokens(tokenizer)
    fp16_results["perplexity"] = compute_perplexity(fp16_model, wikitext_tokens, device)
    log.info("FP16 perplexity: %.2f", fp16_results["perplexity"])

    fp16_label = f"FP16 ({model_name})"
    print_results(fp16_label, fp16_results)
    dump_results(run_dir, fp16_label, fp16_results)
    dump_samples_csv(run_dir, fp16_label, fp16_samples)

    del fp16_model
    free_memory()

    # ── 2. Quantize (GGUF Q4_K_M) — skip if already saved ────────────────
    if os.path.isdir(quant_path):
        log.info("quantized model found at %s, skipping quantization.", quant_path)
    else:
        log.info("quantizing model (GGUF %s)...", gguf_format)
        qconfig = GGUFConfig(bits=4, format=gguf_format)

        quant_model = GPTQModel.load(fp16_id, qconfig)

        t0 = time.perf_counter()
        quant_model.quantize(calibration=None, tokenizer=tokenizer, backend=gguf_backend)
        log.info("quantization took %.1fs", time.perf_counter() - t0)

        quant_model.save(quant_path)
        tokenizer.save_pretrained(quant_path)
        del quant_model
        free_memory()

    # ── 3. Benchmark quantized GGUF ──────────────────────────────────────
    log.info("loading quantized GGUF model...")
    reset_peak_memory(device)
    t0 = time.perf_counter()
    quant_model = GPTQModel.load(quant_path, backend=gguf_backend, device=device)
    if device != "cpu" and hasattr(quant_model, "model"):
        quant_model.model.to(device)
    sync_device(device)
    quant_weight_mem = get_peak_memory_mb(device)
    log.info("loaded in %.1fs (weight mem: %.1f MB)", time.perf_counter() - t0, quant_weight_mem)

    quant_tokenizer = AutoTokenizer.from_pretrained(quant_path)
    if not quant_tokenizer.pad_token_id:
        quant_tokenizer.pad_token_id = quant_tokenizer.eos_token_id

    log.info("evaluating GSM8K (GGUF %s)...", gguf_format)
    reset_peak_memory(device)
    quant_results, quant_samples = evaluate_gsm8k(quant_model, quant_tokenizer, gsm8k_questions, gsm8k_max_tokens, warmup_runs, device)
    quant_results["weight_mem_mb"] = quant_weight_mem
    quant_results["runtime_mem_mb"] = max(0, quant_results["peak_mem_mb"] - quant_weight_mem)
    log.info("GGUF GSM8K accuracy: %.1f%%", quant_results["gsm8k_accuracy"] * 100)

    log.info("computing perplexity (GGUF %s)...", gguf_format)
    quant_wikitext_tokens = load_wikitext2_tokens(quant_tokenizer)
    quant_results["perplexity"] = compute_perplexity(quant_model, quant_wikitext_tokens, device)
    log.info("GGUF perplexity: %.2f", quant_results["perplexity"])

    quant_label = f"GGUF {gguf_format} ({model_name})"
    print_results(quant_label, quant_results)
    dump_results(run_dir, quant_label, quant_results)
    dump_samples_csv(run_dir, quant_label, quant_samples)

    del quant_model
    free_memory()

    # ── 4. Comparison ─────────────────────────────────────────────────────
    print_comparison(f"FP16 {model_name}", fp16_results, f"GGUF-{gguf_format} {model_name}", quant_results)


def main():
    run_dir = setup_run_logging(__file__)
    log.info("device: %s", device)
    log.info("models: %s", [c["name"] for c in MODEL_CONFIGS])

    gguf_backend = BACKEND.GGUF_TRITON if device.startswith("cuda") else BACKEND.GGUF_TORCH
    log.info("gguf backend: %s", gguf_backend)

    gsm8k_questions = load_gsm8k_questions(num_samples=gsm8k_samples)
    log.info("loaded %d GSM8K questions", len(gsm8k_questions))

    for config in MODEL_CONFIGS:
        benchmark_model_pair(run_dir, config, gsm8k_questions, gguf_backend)


if __name__ == "__main__":
    main()
