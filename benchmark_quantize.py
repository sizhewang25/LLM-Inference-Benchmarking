import logging
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from gptqmodel import GPTQModel
from bench_utils import (
    compute_perplexity,
    dump_results,
    evaluate_gsm8k,
    free_memory,
    load_gsm8k_questions,
    load_wikitext2_tokens,
    pick_device,
    print_comparison,
    print_results,
    setup_run_logging,
)

log = logging.getLogger("bench")

# ── Configuration ──────────────────────────────────────────────
model_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
quant_id = "ModelCloud/DeepSeek-R1-Distill-Qwen-7B-gptqmodel-4bit-vortex-v2"
gsm8k_samples    = 100
gsm8k_max_tokens = 512
warmup_runs      = 2
device = pick_device()


def main():
    run_dir = setup_run_logging(__file__)
    log.info("device: %s", device)
    log.info("model: %s", model_id)
    log.info("quant: %s", quant_id)

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if not tokenizer.pad_token_id:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    gsm8k_questions = load_gsm8k_questions(num_samples=gsm8k_samples)
    log.info("loaded %d GSM8K questions", len(gsm8k_questions))

    # ── 1. Benchmark FP16 ──
    log.info("loading original FP16 model...")
    t0 = time.perf_counter()
    fp16_kwargs = {"torch_dtype": torch.float16}
    if device == "mps":
        # See benchmark_gguf.py for full rationale: MPS needs eager attention
        # for generate(), and `device_map=` triggers an accelerate warmup that
        # blows past Apple's per-buffer size limit on 7B models. Use plain
        # `.to(device)` to load tensor-by-tensor instead.
        fp16_kwargs["attn_implementation"] = "eager"
        orig_model = AutoModelForCausalLM.from_pretrained(model_id, **fp16_kwargs).to(device)
    else:
        fp16_kwargs["device_map"] = device
        orig_model = AutoModelForCausalLM.from_pretrained(model_id, **fp16_kwargs)
    log.info("loaded in %.1fs", time.perf_counter() - t0)

    log.info("evaluating GSM8K (FP16)...")
    fp16_results = evaluate_gsm8k(orig_model, tokenizer, gsm8k_questions, gsm8k_max_tokens, warmup_runs, device)
    log.info("FP16 GSM8K accuracy: %.1f%%", fp16_results["gsm8k_accuracy"] * 100)

    log.info("computing perplexity (FP16)...")
    wikitext_tokens = load_wikitext2_tokens(tokenizer)
    fp16_results["perplexity"] = compute_perplexity(orig_model, wikitext_tokens, device)
    log.info("FP16 perplexity: %.2f", fp16_results["perplexity"])

    print_results("FP16 (Original)", fp16_results)
    dump_results(run_dir, "FP16 (Original)", fp16_results)

    del orig_model
    free_memory()

    # ── 2. Benchmark pre-quantized GPTQ model ──
    log.info("loading pre-quantized GPTQ model: %s", quant_id)
    t0 = time.perf_counter()
    quant_model = GPTQModel.load(quant_id, device=device)
    log.info("loaded in %.1fs", time.perf_counter() - t0)

    # Prefer the tokenizer shipped with the quantized repo so any drift in
    # special tokens or chat template stays consistent with the quant model.
    try:
        quant_tokenizer = AutoTokenizer.from_pretrained(quant_id)
        if not quant_tokenizer.pad_token_id:
            quant_tokenizer.pad_token_id = quant_tokenizer.eos_token_id
    except Exception as e:
        log.warning("could not load tokenizer from %s (%s); falling back to base tokenizer", quant_id, e)
        quant_tokenizer = tokenizer

    log.info("evaluating GSM8K (GPTQ 4-bit)...")
    quant_results = evaluate_gsm8k(quant_model, quant_tokenizer, gsm8k_questions, gsm8k_max_tokens, warmup_runs, device)
    log.info("GPTQ GSM8K accuracy: %.1f%%", quant_results["gsm8k_accuracy"] * 100)

    log.info("computing perplexity (GPTQ 4-bit)...")
    quant_wikitext_tokens = load_wikitext2_tokens(quant_tokenizer)
    quant_results["perplexity"] = compute_perplexity(quant_model, quant_wikitext_tokens, device)
    log.info("GPTQ perplexity: %.2f", quant_results["perplexity"])

    print_results("GPTQ 4-bit (Quantized)", quant_results)
    dump_results(run_dir, "GPTQ 4-bit (Quantized)", quant_results)

    del quant_model
    free_memory()

    # ── 3. Comparison ──
    print_comparison("FP16", fp16_results, "GPTQ-4bit", quant_results)


if __name__ == "__main__":
    main()
