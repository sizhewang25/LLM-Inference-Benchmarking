import logging
import os
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from gptqmodel import BACKEND, GPTQModel
from gptqmodel.quantization import GGUFConfig
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
# Tiny model for local end-to-end smoke runs (already cached on dev Mac).
# For the real 7B comparison, swap back to "Qwen/Qwen2.5-7B-Instruct".
model_id = "Qwen/Qwen2.5-0.5B-Instruct"
quant_path = "models/Qwen2.5-0.5B-Instruct-GGUF-Q4_K_M"
gguf_format = "q4_k_m"
gsm8k_samples    = 100
gsm8k_max_tokens = 512
warmup_runs      = 2
device = pick_device()


def main():
    run_dir = setup_run_logging(__file__)
    log.info("device: %s", device)
    log.info("model: %s", model_id)

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
        # Two MPS-specific quirks:
        #  1. The default SDPA attention path crashes inside generate()
        #     ("mps_matmul: invalid shape") — force eager.
        #  2. Using `device_map="mps"` triggers accelerate's
        #     `caching_allocator_warmup`, which tries to pre-allocate the
        #     entire model footprint as one tensor and fails on Apple's
        #     per-buffer size limit ("Invalid buffer size: 14.19 GiB").
        #     Instead, load to CPU then `.to("mps")` tensor-by-tensor.
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

    # The default GGUF kernel is GGUF_TRITON, which requires `triton` and is
    # CUDA-only. On MPS/CPU fall back to the pure-PyTorch GGUF_TORCH kernel
    # (DEVICE.ALL, PLATFORM.ALL per gptqmodel's qlinear registry).
    gguf_backend = BACKEND.GGUF_TRITON if device.startswith("cuda") else BACKEND.GGUF_TORCH
    log.info("gguf backend: %s", gguf_backend)

    # ── 2. Quantize (GGUF Q4_K_M) — skip if already saved ──
    if os.path.isdir(quant_path):
        log.info("quantized model found at %s, skipping quantization.", quant_path)
    else:
        log.info("quantizing model (GGUF %s)...", gguf_format)
        qconfig = GGUFConfig(bits=4, format=gguf_format)

        quant_model = GPTQModel.load(model_id, qconfig)

        t0 = time.perf_counter()
        quant_model.quantize(calibration=None, tokenizer=tokenizer, backend=gguf_backend)
        log.info("quantization took %.1fs", time.perf_counter() - t0)

        quant_model.save(quant_path)
        tokenizer.save_pretrained(quant_path)
        del quant_model
        free_memory()

    # ── 3. Benchmark quantized ──
    log.info("loading quantized GGUF model...")
    t0 = time.perf_counter()
    quant_model = GPTQModel.load(quant_path, backend=gguf_backend, device=device)
    # GPTQModel.load places the packed quant weights on `device`, but on MPS
    # the unquantized layers (embeddings, layernorms, lm_head) can be left on
    # CPU, causing "Placeholder storage has not been allocated on MPS device"
    # at the first embed_tokens() call. Force the whole HF submodule onto
    # the target device to make sure inputs and weights co-locate.
    if device != "cpu" and hasattr(quant_model, "model"):
        quant_model.model.to(device)
    log.info("loaded in %.1fs", time.perf_counter() - t0)

    # Load the tokenizer from the quantized path so any drift in special
    # tokens, vocab, or chat template stays consistent with the quant model.
    quant_tokenizer = AutoTokenizer.from_pretrained(quant_path)
    if not quant_tokenizer.pad_token_id:
        quant_tokenizer.pad_token_id = quant_tokenizer.eos_token_id

    log.info("evaluating GSM8K (GGUF %s)...", gguf_format)
    quant_results = evaluate_gsm8k(quant_model, quant_tokenizer, gsm8k_questions, gsm8k_max_tokens, warmup_runs, device)
    log.info("GGUF GSM8K accuracy: %.1f%%", quant_results["gsm8k_accuracy"] * 100)

    log.info("computing perplexity (GGUF %s)...", gguf_format)
    quant_wikitext_tokens = load_wikitext2_tokens(quant_tokenizer)
    quant_results["perplexity"] = compute_perplexity(quant_model, quant_wikitext_tokens, device)
    log.info("GGUF perplexity: %.2f", quant_results["perplexity"])

    label = f"GGUF {gguf_format} (Quantized)"
    print_results(label, quant_results)
    dump_results(run_dir, label, quant_results)

    del quant_model
    free_memory()

    # ── 4. Comparison ──
    print_comparison("FP16", fp16_results, f"GGUF-{gguf_format}", quant_results)


if __name__ == "__main__":
    main()
