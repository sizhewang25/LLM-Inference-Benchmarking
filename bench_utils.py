import gc
import time
from statistics import mean, stdev

import torch
from transformers import GenerationConfig
from transformers.generation.streamers import BaseStreamer

PROMPTS = [
    "Explain the theory of relativity in simple terms.",
    "Write a Python function to sort a list using merge sort.",
    "What are the main causes of climate change?",
    "Describe the process of photosynthesis step by step.",
    "What is the difference between TCP and UDP?",
    "Summarize the plot of Romeo and Juliet.",
    "How does a neural network learn?",
    "Explain quantum computing to a 10 year old.",
    "What are the benefits of regular exercise?",
    "Describe how a compiler works.",
    "What is the significance of the Turing test?",
    "Explain how encryption protects data.",
    "What causes earthquakes?",
    "How does the internet work?",
    "What is the difference between AI and machine learning?",
]


class TTFTStreamer(BaseStreamer):
    """Records wall-clock time when the first generated token arrives."""

    def __init__(self):
        self.first_token_time = None
        self.start_time = None
        self._first_call = True

    def put(self, _value):
        if self._first_call:
            self._first_call = False
            return
        if self.first_token_time is None:
            self.first_token_time = time.perf_counter()

    def end(self):
        pass


def get_peak_memory_mb(device="cuda:0"):
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated(device) / 1024 / 1024
    return 0.0


def reset_peak_memory(device="cuda:0"):
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)


def free_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def benchmark(model, tokenizer, prompts, max_new_tokens, warmup_runs, device):
    gen_config = GenerationConfig(
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
    )

    # Warmup
    for i in range(min(warmup_runs, len(prompts))):
        inputs = tokenizer(prompts[i], return_tensors="pt").to(device)
        with torch.no_grad():
            model.generate(**inputs, generation_config=gen_config)
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # Measure
    reset_peak_memory(device)

    all_ttft = []
    all_latency = []
    all_tpot = []
    all_tokens = []

    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        input_len = inputs["input_ids"].shape[1]

        streamer = TTFTStreamer()
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        streamer.start_time = time.perf_counter()
        t_start = streamer.start_time

        with torch.no_grad():
            output_ids = model.generate(
                **inputs, generation_config=gen_config, streamer=streamer,
            )

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t_end = time.perf_counter()

        new_tokens = output_ids.shape[1] - input_len
        latency = t_end - t_start

        ttft = (streamer.first_token_time - t_start) if streamer.first_token_time else latency

        decode_tokens = max(new_tokens - 1, 1)
        decode_time = latency - ttft
        tpot = decode_time / decode_tokens

        all_ttft.append(ttft * 1000)
        all_latency.append(latency * 1000)
        all_tpot.append(tpot * 1000)
        all_tokens.append(new_tokens)

    total_tokens = sum(all_tokens)
    total_time = sum(all_latency) / 1000

    return {
        "throughput_tok_s": total_tokens / total_time,
        "ttft_mean_ms": mean(all_ttft),
        "ttft_std_ms": stdev(all_ttft) if len(all_ttft) > 1 else 0,
        "latency_mean_ms": mean(all_latency),
        "latency_std_ms": stdev(all_latency) if len(all_latency) > 1 else 0,
        "tpot_mean_ms": mean(all_tpot),
        "tpot_std_ms": stdev(all_tpot) if len(all_tpot) > 1 else 0,
        "peak_mem_mb": get_peak_memory_mb(device),
        "num_samples": len(prompts),
        "total_tokens": total_tokens,
    }


def print_results(label, r):
    print(f"\n{'=' * 60}")
    print(f"  {label}")
    print(f"{'=' * 60}")
    print(f"  Throughput          : {r['throughput_tok_s']:>8.2f} tokens/s")
    print(f"  TTFT (prefill)      : {r['ttft_mean_ms']:>8.2f} ms  (std {r['ttft_std_ms']:.2f})")
    print(f"  TPOT (decode)       : {r['tpot_mean_ms']:>8.2f} ms  (std {r['tpot_std_ms']:.2f})")
    print(f"  Latency (TTFT+TPOT) : {r['latency_mean_ms']:>8.2f} ms  (std {r['latency_std_ms']:.2f})")
    print(f"  Peak memory         : {r['peak_mem_mb']:>8.1f} MB")
    print(f"  Samples / tokens    : {r['num_samples']} / {r['total_tokens']}")


def print_comparison(label_a, results_a, label_b, results_b):
    def delta(b, a):
        if a == 0:
            return "N/A"
        pct = (b - a) / abs(a) * 100
        sign = "+" if pct >= 0 else ""
        return f"{sign}{pct:.1f}%"

    print(f"\n{'=' * 60}")
    print(f"  COMPARISON  ({label_a} vs {label_b})")
    print(f"{'=' * 60}")
    print(f"  {'Metric':<22} {label_a:>12} {label_b:>12} {'Delta':>10}")
    print(f"  {'-' * 56}")
    print(f"  {'Throughput (tok/s)':<22} {results_a['throughput_tok_s']:>12.2f} {results_b['throughput_tok_s']:>12.2f} {delta(results_b['throughput_tok_s'], results_a['throughput_tok_s']):>10}")
    print(f"  {'TTFT (ms)':<22} {results_a['ttft_mean_ms']:>12.2f} {results_b['ttft_mean_ms']:>12.2f} {delta(results_b['ttft_mean_ms'], results_a['ttft_mean_ms']):>10}")
    print(f"  {'TPOT (ms)':<22} {results_a['tpot_mean_ms']:>12.2f} {results_b['tpot_mean_ms']:>12.2f} {delta(results_b['tpot_mean_ms'], results_a['tpot_mean_ms']):>10}")
    print(f"  {'Latency (ms)':<22} {results_a['latency_mean_ms']:>12.2f} {results_b['latency_mean_ms']:>12.2f} {delta(results_b['latency_mean_ms'], results_a['latency_mean_ms']):>10}")
    print(f"  {'Peak memory (MB)':<22} {results_a['peak_mem_mb']:>12.1f} {results_b['peak_mem_mb']:>12.1f} {delta(results_b['peak_mem_mb'], results_a['peak_mem_mb']):>10}")
    print(f"  {'-' * 56}")
    speedup = results_b["throughput_tok_s"] / results_a["throughput_tok_s"] if results_a["throughput_tok_s"] else 0
    mem_saved = (1 - results_b["peak_mem_mb"] / results_a["peak_mem_mb"]) * 100 if results_a["peak_mem_mb"] else 0
    print(f"  Speedup: {speedup:.2f}x  |  Memory saved: {mem_saved:.1f}%")
    print(f"{'=' * 60}")
