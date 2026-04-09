import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from gptqmodel import GPTQModel
from bench_utils import PROMPTS, benchmark, print_results, print_comparison, free_memory

# ── Configuration ──────────────────────────────────────────────
model_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
quant_id = "ModelCloud/DeepSeek-R1-Distill-Qwen-7B-gptqmodel-4bit-vortex-v2"
num_samples = 10
max_new_tokens = 128
warmup_runs = 2
device = "cuda:0" if torch.cuda.is_available() else "cpu"


def main():
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if not tokenizer.pad_token_id:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    prompts = PROMPTS[:num_samples]

    # ── 1. Benchmark FP16 ──
    print("\n>>> Loading original FP16 model...")
    t0 = time.perf_counter()
    orig_model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float16, device_map=device,
    )
    print(f"    Loaded in {time.perf_counter() - t0:.1f}s")

    print(">>> Benchmarking FP16 model...")
    fp16_results = benchmark(orig_model, tokenizer, prompts, max_new_tokens, warmup_runs, device)
    print_results("FP16 (Original)", fp16_results)

    del orig_model
    free_memory()

    # ── 2. Benchmark pre-quantized GPTQ model ──
    print(f"\n>>> Loading pre-quantized GPTQ model: {quant_id}")
    t0 = time.perf_counter()
    quant_model = GPTQModel.load(quant_id, device=device)
    print(f"    Loaded in {time.perf_counter() - t0:.1f}s")

    print(">>> Benchmarking GPTQ 4-bit model...")
    quant_results = benchmark(quant_model, tokenizer, prompts, max_new_tokens, warmup_runs, device)
    print_results("GPTQ 4-bit (Quantized)", quant_results)

    del quant_model
    free_memory()

    # ── 3. Comparison ──
    print_comparison("FP16", fp16_results, "GPTQ-4bit", quant_results)


if __name__ == "__main__":
    main()
