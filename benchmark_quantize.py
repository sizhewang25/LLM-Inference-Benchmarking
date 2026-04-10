import logging
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from gptqmodel import GPTQModel
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
model_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
quant_id = "ModelCloud/DeepSeek-R1-Distill-Qwen-7B-gptqmodel-4bit-vortex-v2"
num_samples = 10
max_new_tokens = 128
warmup_runs = 2
device = pick_device()


def main():
    run_dir = setup_run_logging(__file__)
    log.info("device: %s", device)
    log.info("model: %s", model_id)
    log.info("quant: %s", quant_id)

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

    log.info("benchmarking FP16 model...")
    fp16_results = benchmark(orig_model, tokenizer, fp16_prompts, max_new_tokens, warmup_runs, device)
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

    quant_prompts = format_prompts(quant_tokenizer, prompts)
    log.info("benchmarking GPTQ 4-bit model...")
    quant_results = benchmark(quant_model, quant_tokenizer, quant_prompts, max_new_tokens, warmup_runs, device)
    print_results("GPTQ 4-bit (Quantized)", quant_results)
    dump_results(run_dir, "GPTQ 4-bit (Quantized)", quant_results)

    del quant_model
    free_memory()

    # ── 3. Comparison ──
    print_comparison("FP16", fp16_results, "GPTQ-4bit", quant_results)


if __name__ == "__main__":
    main()
