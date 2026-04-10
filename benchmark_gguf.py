import logging
import os
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from gptqmodel import BACKEND, GPTQModel
from gptqmodel.quantization import GGUFConfig
from bench_utils import (
    PROMPTS,
    benchmark,
    dump_results,
    format_prompts,
    free_memory,
    pick_device,
    print_comparison,
    print_results,
    setup_run_logging,
)

log = logging.getLogger("bench")

# ── Configuration ──────────────────────────────────────────────
model_id = "Qwen/Qwen2.5-7B-Instruct"
quant_path = "Qwen2.5-7B-Instruct-GGUF-Q4_K_M"
gguf_format = "q4_k_m"
num_samples = 10
max_new_tokens = 128
warmup_runs = 2
device = pick_device()


def main():
    run_dir = setup_run_logging(__file__)
    log.info("device: %s", device)
    log.info("model: %s", model_id)

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if not tokenizer.pad_token_id:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    prompts = PROMPTS[:num_samples]
    fp16_prompts = format_prompts(tokenizer, prompts)

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

    log.info("benchmarking FP16 model...")
    fp16_results = benchmark(orig_model, tokenizer, fp16_prompts, max_new_tokens, warmup_runs, device)
    print_results("FP16 (Original)", fp16_results)
    dump_results(run_dir, "FP16 (Original)", fp16_results)

    del orig_model
    free_memory()

    # ── 2. Quantize (GGUF Q4_K_M) — skip if already saved ──
    if os.path.isdir(quant_path):
        log.info("quantized model found at %s, skipping quantization.", quant_path)
    else:
        log.info("quantizing model (GGUF %s)...", gguf_format)
        qconfig = GGUFConfig(bits=4, format=gguf_format)

        quant_model = GPTQModel.load(model_id, qconfig)

        t0 = time.perf_counter()
        quant_model.quantize(calibration=None, tokenizer=tokenizer)
        log.info("quantization took %.1fs", time.perf_counter() - t0)

        quant_model.save(quant_path)
        tokenizer.save_pretrained(quant_path)
        del quant_model
        free_memory()

    # ── 3. Benchmark quantized ──
    log.info("loading quantized GGUF model...")
    t0 = time.perf_counter()
    quant_model = GPTQModel.load(quant_path, backend=BACKEND.GGUF_TRITON, device=device)
    log.info("loaded in %.1fs", time.perf_counter() - t0)

    # Load the tokenizer from the quantized path so any drift in special
    # tokens, vocab, or chat template stays consistent with the quant model.
    quant_tokenizer = AutoTokenizer.from_pretrained(quant_path)
    if not quant_tokenizer.pad_token_id:
        quant_tokenizer.pad_token_id = quant_tokenizer.eos_token_id

    quant_prompts = format_prompts(quant_tokenizer, prompts)
    log.info("benchmarking GGUF %s model...", gguf_format)
    quant_results = benchmark(quant_model, quant_tokenizer, quant_prompts, max_new_tokens, warmup_runs, device)
    label = f"GGUF {gguf_format} (Quantized)"
    print_results(label, quant_results)
    dump_results(run_dir, label, quant_results)

    del quant_model
    free_memory()

    # ── 4. Comparison ──
    print_comparison("FP16", fp16_results, f"GGUF-{gguf_format}", quant_results)


if __name__ == "__main__":
    main()
