import os
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from gptqmodel import BACKEND, GPTQModel
from gptqmodel.quantization import GGUFConfig
from bench_utils import PROMPTS, benchmark, print_results, print_comparison, free_memory

# ── Configuration ──────────────────────────────────────────────
model_id = "Qwen/Qwen2.5-7B-Instruct"
quant_path = "Qwen2.5-7B-Instruct-GGUF-Q4_K_M"
gguf_format = "q4_k_m"
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

    # ── 2. Quantize (GGUF Q4_K_M) — skip if already saved ──
    if os.path.isdir(quant_path):
        print(f"\n>>> Quantized model found at {quant_path}, skipping quantization.")
    else:
        print(f"\n>>> Quantizing model (GGUF {gguf_format})...")
        qconfig = GGUFConfig(bits=4, format=gguf_format)

        quant_model = GPTQModel.load(model_id, qconfig)

        t0 = time.perf_counter()
        quant_model.quantize(calibration=None, tokenizer=tokenizer)
        print(f"    Quantization took {time.perf_counter() - t0:.1f}s")

        quant_model.save(quant_path)
        tokenizer.save_pretrained(quant_path)
        del quant_model
        free_memory()

    # ── 3. Benchmark quantized ──
    print("\n>>> Loading quantized GGUF model...")
    t0 = time.perf_counter()
    quant_model = GPTQModel.load(quant_path, backend=BACKEND.GGUF_TRITON, device=device)
    print(f"    Loaded in {time.perf_counter() - t0:.1f}s")

    print(f">>> Benchmarking GGUF {gguf_format} model...")
    quant_results = benchmark(quant_model, tokenizer, prompts, max_new_tokens, warmup_runs, device)
    print_results(f"GGUF {gguf_format} (Quantized)", quant_results)

    del quant_model
    free_memory()

    # ── 4. Comparison ──
    print_comparison("FP16", fp16_results, f"GGUF-{gguf_format}", quant_results)


if __name__ == "__main__":
    main()
