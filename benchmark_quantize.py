import logging
import time

import torch
from transformers import AutoTokenizer

from gptqmodel import GPTQModel
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
        "quant_id": "Qwen/Qwen2.5-0.5B-Instruct-GPTQ-Int4",
    },
    # {
    #     "name": "Llama-3.1-8B-Instruct",
    #     "fp16_id": "meta-llama/Llama-3.1-8B-Instruct",
    #     "quant_id": "hugging-quants/Meta-Llama-3.1-8B-Instruct-GPTQ-INT4",
    # },
    # {
    #     "name": "Qwen2.5-7B-Instruct",
    #     "fp16_id": "Qwen/Qwen2.5-7B-Instruct",
    #     "quant_id": "Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4",
    # },
    # {
    #     "name": "Gemma-2-9B-it",
    #     "fp16_id": "google/gemma-2-9b-it",
    #     "quant_id": "ModelCloud/gemma-2-9b-it-gptq-4bit",
    # },
]

gsm8k_samples    = 50
gsm8k_max_tokens = 512
warmup_runs      = 2
device = pick_device()


def get_weight_memory_mb(device):
    """Snapshot current memory as a proxy for model weight memory."""
    sync_device(device)
    return get_peak_memory_mb(device)


def benchmark_model_pair(run_dir, config, gsm8k_questions):
    """Run FP16 vs GPTQ 4-bit benchmark for a single model config."""
    model_name = config["name"]
    fp16_id = config["fp16_id"]
    quant_id = config["quant_id"]

    log.info("=" * 60)
    log.info("MODEL: %s", model_name)
    log.info("=" * 60)

    # ── 1. Benchmark FP16 via GPTQModel ──────────────────────────────────
    log.info("loading FP16 model via GPTQModel: %s", fp16_id)
    reset_peak_memory(device)
    t0 = time.perf_counter()
    fp16_model = GPTQModel.load(fp16_id, device=device)
    sync_device(device)
    weight_mem = get_weight_memory_mb(device)
    log.info("loaded in %.1fs (weight mem: %.1f MB)", time.perf_counter() - t0, weight_mem)

    tokenizer = AutoTokenizer.from_pretrained(fp16_id)
    if not tokenizer.pad_token_id:
        tokenizer.pad_token_id = tokenizer.eos_token_id

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

    # ── 2. Benchmark pre-quantized GPTQ model ────────────────────────────
    log.info("loading pre-quantized GPTQ model: %s", quant_id)
    reset_peak_memory(device)
    t0 = time.perf_counter()
    quant_model = GPTQModel.load(quant_id, device=device)
    sync_device(device)
    quant_weight_mem = get_weight_memory_mb(device)
    log.info("loaded in %.1fs (weight mem: %.1f MB)", time.perf_counter() - t0, quant_weight_mem)

    try:
        quant_tokenizer = AutoTokenizer.from_pretrained(quant_id)
        if not quant_tokenizer.pad_token_id:
            quant_tokenizer.pad_token_id = quant_tokenizer.eos_token_id
    except Exception as e:
        log.warning("could not load tokenizer from %s (%s); falling back to base tokenizer", quant_id, e)
        quant_tokenizer = tokenizer

    log.info("evaluating GSM8K (GPTQ 4-bit)...")
    reset_peak_memory(device)
    quant_results, quant_samples = evaluate_gsm8k(quant_model, quant_tokenizer, gsm8k_questions, gsm8k_max_tokens, warmup_runs, device)
    quant_results["weight_mem_mb"] = quant_weight_mem
    quant_results["runtime_mem_mb"] = max(0, quant_results["peak_mem_mb"] - quant_weight_mem)
    log.info("GPTQ GSM8K accuracy: %.1f%%", quant_results["gsm8k_accuracy"] * 100)

    log.info("computing perplexity (GPTQ 4-bit)...")
    quant_wikitext_tokens = load_wikitext2_tokens(quant_tokenizer)
    quant_results["perplexity"] = compute_perplexity(quant_model, quant_wikitext_tokens, device)
    log.info("GPTQ perplexity: %.2f", quant_results["perplexity"])

    quant_label = f"GPTQ 4-bit ({model_name})"
    print_results(quant_label, quant_results)
    dump_results(run_dir, quant_label, quant_results)
    dump_samples_csv(run_dir, quant_label, quant_samples)

    del quant_model
    free_memory()

    # ── 3. Comparison ─────────────────────────────────────────────────────
    print_comparison(f"FP16 {model_name}", fp16_results, f"GPTQ-4bit {model_name}", quant_results)


def main():
    run_dir = setup_run_logging(__file__)
    log.info("device: %s", device)
    log.info("models: %s", [c["name"] for c in MODEL_CONFIGS])

    gsm8k_questions = load_gsm8k_questions(num_samples=gsm8k_samples)
    log.info("loaded %d GSM8K questions", len(gsm8k_questions))

    for config in MODEL_CONFIGS:
        benchmark_model_pair(run_dir, config, gsm8k_questions)


if __name__ == "__main__":
    main()
