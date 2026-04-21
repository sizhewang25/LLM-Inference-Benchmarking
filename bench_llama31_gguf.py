#!/usr/bin/env python3
"""
Benchmark open-weight LLMs at multiple GGUF quantization levels via GPTQModel.

Speed metrics (matching llama.cpp llama-bench convention):
  Prefill  = PP512  tokens/s
  Decode   = TG128  tokens/s
  TTFT     = time-to-first-token, ms
  TPOT     = time-per-output-token, ms
  Latency  = TTFT + TPOT, ms
  Weight   = VRAM allocated by model weights after load, GiB
  Peak     = max VRAM allocated during benchmark, GiB

Quality metrics:
  PPL         = perplexity on Wikitext-2 test set (lower is better)
  HellaSwag   = accuracy on 400 commonsense-NLI tasks (higher is better)
  Winogrande  = accuracy on 1,267 pronoun-resolution tasks (higher is better)

Usage:
  python scripts/bench_llama31_gguf.py --models llama --device cuda:0
  python scripts/bench_llama31_gguf.py --models llama,qwen,gemma --eval
"""

from __future__ import annotations

import argparse
import gc
import math
import os
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# ── constants ──────────────────────────────────────────────────────────────────
PROMPT_TOKENS = 512
DECODE_TOKENS = 128
LATENCY_RUNS = 3
WARMUP_RUNS = 1

HELLASWAG_SAMPLES = 400
PPL_N_CTX = 512
PPL_MAX_CHUNKS = 400

_FILL_SENTENCE = (
    "Summarize the scientific, historical, and economic significance of the "
    "Atlantic Ocean for intercontinental trade, climate, and biodiversity. "
)

MODELS = {
    "llama": {
        "id": "meta-llama/Llama-3.1-8B-Instruct",
        "name": "Llama-3.1-8B-Instruct",
        "short": "Llama-3.1-8B",
    },
    "qwen": {
        "id": "Qwen/Qwen2.5-7B-Instruct",
        "name": "Qwen2.5-7B-Instruct",
        "short": "Qwen2.5-7B",
    },
    "gemma": {
        "id": "google/gemma-2-9b-it",
        "name": "Gemma-2-9B-IT",
        "short": "Gemma-2-9B",
    },
}


# ── helpers ───────────────────────────────────────────────────────────────────

def _sync(device: str) -> None:
    if torch.cuda.is_available():
        idx = int(device.split(":")[-1]) if ":" in device else 0
        torch.cuda.synchronize(idx)


def _gpu_idx(device: str) -> int:
    return int(device.split(":")[-1]) if ":" in device else 0


def _weight_vram_gib(device: str) -> float:
    return torch.cuda.memory_allocated(_gpu_idx(device)) / 1024**3


def _peak_vram_gib(device: str) -> float:
    return torch.cuda.max_memory_allocated(_gpu_idx(device)) / 1024**3


def _reset_peak(device: str) -> None:
    torch.cuda.reset_peak_memory_stats(_gpu_idx(device))


def build_prompt(tokenizer, target_tokens: int) -> tuple[str, int]:
    prompt = _FILL_SENTENCE
    while True:
        n = len(tokenizer(prompt, return_tensors="pt").input_ids[0])
        if n >= target_tokens:
            return prompt, n
        prompt += _FILL_SENTENCE


def _free_model(*models) -> None:
    for m in models:
        del m
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ── result containers ────────────────────────────────────────────────────────

@dataclass
class BenchResult:
    quant: str
    prefill_toks_s: float
    decode_toks_s: float
    ttft_ms: list[float]
    tpot_ms: list[float]
    latency_ms: list[float]
    weight_vram_gib: float
    peak_vram_gib: float

    @property
    def ttft_mean(self) -> float: return statistics.mean(self.ttft_ms)
    @property
    def ttft_std(self) -> float:
        return statistics.stdev(self.ttft_ms) if len(self.ttft_ms) > 1 else 0.0
    @property
    def tpot_mean(self) -> float: return statistics.mean(self.tpot_ms)
    @property
    def tpot_std(self) -> float:
        return statistics.stdev(self.tpot_ms) if len(self.tpot_ms) > 1 else 0.0
    @property
    def lat_mean(self) -> float: return statistics.mean(self.latency_ms)
    @property
    def lat_std(self) -> float:
        return statistics.stdev(self.latency_ms) if len(self.latency_ms) > 1 else 0.0


@dataclass
class QualityResult:
    quant: str
    ppl: float
    ppl_std: float
    hellaswag: float
    winogrande: float


@dataclass
class ModelResults:
    model_name: str
    model_short: str
    speed: list[BenchResult]
    quality: list[QualityResult]


# ── speed measurement ────────────────────────────────────────────────────────

def _prefill_once(hf_model, enc: dict, device: str) -> float:
    _sync(device)
    t0 = time.perf_counter()
    with torch.inference_mode():
        hf_model(**enc, use_cache=True)
    _sync(device)
    return (time.perf_counter() - t0) * 1e3


def _streaming_once(
    hf_model, enc: dict, decode_tokens: int, device: str
) -> tuple[float, float, float]:
    _sync(device)
    t0 = time.perf_counter()
    with torch.inference_mode():
        out = hf_model(**enc, use_cache=True)
    _sync(device)
    ttft_ms = (time.perf_counter() - t0) * 1e3

    past_kv = out.past_key_values
    next_tok = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)

    t1 = time.perf_counter()
    with torch.inference_mode():
        for _ in range(decode_tokens - 1):
            out = hf_model(
                input_ids=next_tok,
                past_key_values=past_kv,
                use_cache=True,
            )
            past_kv = out.past_key_values
            next_tok = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    _sync(device)
    decode_total_ms = (time.perf_counter() - t1) * 1e3

    tpot_ms = decode_total_ms / (decode_tokens - 1)
    return ttft_ms, tpot_ms, ttft_ms + tpot_ms


def run_benchmark(
    label: str,
    hf_model,
    enc: dict,
    device: str,
    warmup: int = WARMUP_RUNS,
    trials: int = LATENCY_RUNS,
    weight_vram_gib: float = 0.0,
) -> BenchResult:
    n_prompt = enc["input_ids"].shape[1]

    print(f"  [{label}] warming up ({warmup} run(s))…")
    for _ in range(warmup):
        _streaming_once(hf_model, enc, DECODE_TOKENS, device)
    _reset_peak(device)

    print(f"  [{label}] measuring prefill throughput…")
    prefill_times_ms: list[float] = []
    for i in range(trials):
        ms = _prefill_once(hf_model, enc, device)
        prefill_times_ms.append(ms)
        print(f"    run {i+1}/{trials}: {ms:.1f} ms  ({n_prompt/(ms/1e3):.0f} t/s)")

    prefill_mean_ms = statistics.mean(prefill_times_ms)
    prefill_toks_s = n_prompt / (prefill_mean_ms / 1e3)

    print(f"  [{label}] measuring streaming latency ({trials} run(s))…")
    ttft_list: list[float] = []
    tpot_list: list[float] = []
    lat_list: list[float] = []

    for i in range(trials):
        ttft, tpot, total = _streaming_once(hf_model, enc, DECODE_TOKENS, device)
        ttft_list.append(ttft)
        tpot_list.append(tpot)
        lat_list.append(total)
        decode_tps = (DECODE_TOKENS - 1) / (tpot * (DECODE_TOKENS - 1) / 1e3)
        print(
            f"    run {i+1}/{trials}: "
            f"TTFT={ttft:.1f} ms  TPOT={tpot:.2f} ms  Total={total:.1f} ms  "
            f"Decode={decode_tps:.1f} t/s"
        )

    mean_tpot_ms = statistics.mean(tpot_list)
    decode_toks_s = 1e3 / mean_tpot_ms

    peak = _peak_vram_gib(device)
    print(
        f"  [{label}] Weight VRAM={weight_vram_gib:.2f} GiB  "
        f"Peak VRAM={peak:.2f} GiB"
    )

    return BenchResult(
        quant=label,
        prefill_toks_s=prefill_toks_s,
        decode_toks_s=decode_toks_s,
        ttft_ms=ttft_list,
        tpot_ms=tpot_list,
        latency_ms=lat_list,
        weight_vram_gib=weight_vram_gib,
        peak_vram_gib=peak,
    )


# ── quality evaluation ───────────────────────────────────────────────────────

def _preprocess_hellaswag(text: str) -> str:
    import re
    text = text.strip()
    text = text.replace(" [title]", ". ")
    text = re.sub(r"\[.*?\]", "", text)
    text = text.replace("  ", " ")
    return text


def compute_ppl(
    hf_model, tokenizer, device: str,
    n_ctx: int = PPL_N_CTX, max_chunks: int = PPL_MAX_CHUNKS,
) -> tuple[float, float]:
    """Perplexity matching llama.cpp: non-overlapping n_ctx-token chunks,
    scoring only the last half of each chunk."""
    from datasets import load_dataset

    print("  [PPL] loading wikitext-2 test set…")
    test = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(test["text"])
    encodings = tokenizer(text, return_tensors="pt", add_special_tokens=True)
    all_tokens = encodings.input_ids[0]
    seq_len = len(all_tokens)

    n_chunk_max = seq_len // n_ctx
    n_chunk = min(max_chunks, n_chunk_max)
    first = n_ctx // 2
    tokens_per_chunk = n_ctx - first - 1

    print(
        f"  [PPL] {seq_len} tokens, n_ctx={n_ctx}, {n_chunk} chunks, "
        f"scoring last {tokens_per_chunk} tokens per chunk"
    )

    bos_token_id = tokenizer.bos_token_id

    nll_sum = 0.0
    nll2_sum = 0.0
    count = 0

    for i in range(n_chunk):
        start = i * n_ctx
        chunk_tokens = all_tokens[start:start + n_ctx].clone()

        if bos_token_id is not None:
            chunk_tokens[0] = bos_token_id

        input_ids = chunk_tokens.unsqueeze(0).to(device)

        with torch.inference_mode():
            logits = hf_model(input_ids).logits[0]

        target_tokens = all_tokens[start + first + 1:start + n_ctx].to(device)
        score_logits = logits[first:n_ctx - 1]

        log_probs = torch.log_softmax(score_logits, dim=-1)
        token_nlls = -log_probs.gather(1, target_tokens.unsqueeze(1)).squeeze(1)

        nll_sum += token_nlls.double().sum().item()
        nll2_sum += (token_nlls.double() ** 2).sum().item()
        count += len(token_nlls)

        if (i + 1) % 50 == 0 or i == n_chunk - 1:
            running_ppl = math.exp(nll_sum / count)
            print(f"    chunk {i+1}/{n_chunk}: running PPL={running_ppl:.4f}")

    nll_mean = nll_sum / count
    nll2_mean = nll2_sum / count
    ppl = math.exp(nll_mean)

    variance = nll2_mean - nll_mean ** 2
    if variance > 0:
        ppl_std = math.sqrt(variance / (count - 1)) * ppl
    else:
        ppl_std = 0.0

    print(f"  [PPL] Final estimate: PPL = {ppl:.4f} +/- {ppl_std:.5f}")
    return ppl, ppl_std


def compute_hellaswag(
    hf_model, tokenizer, device: str, max_samples: int = HELLASWAG_SAMPLES
) -> float:
    """HellaSwag accuracy using length-normalized log-likelihood (acc_norm),
    matching llama.cpp's scoring methodology."""
    from datasets import load_dataset
    import random as _random

    print(f"  [HellaSwag] loading validation set (max {max_samples} samples)…")
    ds = load_dataset("Rowan/hellaswag", split="validation")

    indices = list(range(len(ds)))
    if max_samples and max_samples < len(ds):
        rng = _random.Random(1)
        rng.shuffle(indices)
        indices = indices[:max_samples]

    correct = 0
    total = len(indices)

    for count_i, idx in enumerate(indices):
        example = ds[idx]

        ctx = _preprocess_hellaswag(
            example["activity_label"] + ": "
            + example["ctx_a"] + " " + example["ctx_b"].capitalize()
        )
        label = int(example["label"])
        endings = example["endings"]

        seq_tokens = []
        for ending in endings:
            full_text = ctx + " " + _preprocess_hellaswag(ending)
            tokens = tokenizer(full_text, return_tensors="pt").input_ids[0]
            seq_tokens.append(tokens)

        min_len = min(len(t) for t in seq_tokens)
        common_prefix = 0
        for k in range(min_len):
            if all(seq_tokens[s][k] == seq_tokens[0][k] for s in range(1, 4)):
                common_prefix += 1
            else:
                break

        best_ll = float("-inf")
        best_idx = -1

        for s in range(4):
            input_ids = seq_tokens[s].unsqueeze(0).to(device)
            seq_len = input_ids.size(1)

            with torch.inference_mode():
                logits = hf_model(input_ids).logits[0]

            score_start = max(common_prefix - 1, 0)
            score_end = seq_len - 1

            if score_end <= score_start:
                continue

            log_probs = torch.log_softmax(logits[score_start:score_end], dim=-1)
            targets = input_ids[0, score_start + 1:score_end + 1]
            token_lls = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)

            avg_ll = token_lls.sum().item() / len(token_lls)

            if avg_ll > best_ll:
                best_ll = avg_ll
                best_idx = s

        if best_idx == label:
            correct += 1

        if (count_i + 1) % 100 == 0:
            print(f"    {count_i+1}/{total}: acc={correct/(count_i+1)*100:.1f}%")

    acc = correct / total * 100
    print(f"  [HellaSwag] accuracy = {acc:.2f}% ({correct}/{total})")
    return acc


def compute_winogrande(hf_model, tokenizer, device: str) -> float:
    """Winogrande accuracy matching llama.cpp: score only the continuation
    after the choice word, length-normalized."""
    from datasets import load_dataset

    print("  [Winogrande] loading validation set…")
    ds = load_dataset("allenai/winogrande", "winogrande_debiased", split="validation")
    total = len(ds)
    print(f"  [Winogrande] {total} samples")

    K_MIN_TRAILING_CTX = 3

    correct = 0
    for i, example in enumerate(ds):
        sentence = example["sentence"]
        option1 = example["option1"]
        option2 = example["option2"]
        answer = int(example["answer"])

        where = sentence.index("_")
        first_part = sentence[:where]
        second_part = sentence[where + 1:]

        choices = [option1, option2]

        seq_tokens = []
        n_bases = []
        for choice in choices:
            full_text = first_part + choice + second_part
            tokens = tokenizer(full_text, return_tensors="pt").input_ids[0]
            seq_tokens.append(tokens)

            base_text = first_part + choice
            base_tokens = tokenizer(base_text, return_tensors="pt").input_ids[0]
            n_bases.append(len(base_tokens))

        min_len = min(len(t) for t in seq_tokens)
        common_prefix = 0
        for k in range(min_len):
            if seq_tokens[0][k] == seq_tokens[1][k]:
                common_prefix += 1
            else:
                break

        skip_choice = (
            len(seq_tokens[0]) - common_prefix > K_MIN_TRAILING_CTX and
            len(seq_tokens[1]) - common_prefix > K_MIN_TRAILING_CTX
        )

        scores = []
        for s in range(2):
            input_ids = seq_tokens[s].unsqueeze(0).to(device)
            seq_len = input_ids.size(1)

            n_base = n_bases[s] if skip_choice else common_prefix
            last = 1 if (seq_len - n_base > 1) else 0

            with torch.inference_mode():
                logits = hf_model(input_ids).logits[0]

            score_start = n_base - 1
            score_end = seq_len - 1 - last

            if score_end <= score_start:
                scores.append(float("-inf"))
                continue

            log_probs = torch.log_softmax(logits[score_start:score_end], dim=-1)
            targets = input_ids[0, score_start + 1:score_end + 1]
            token_lls = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)

            n_scored = seq_len - n_base - last
            avg_ll = token_lls.sum().item() / n_scored
            scores.append(avg_ll)

        result = 1 if scores[0] > scores[1] else 2

        if result == answer:
            correct += 1

        if (i + 1) % 200 == 0:
            print(f"    {i+1}/{total}: acc={correct/(i+1)*100:.1f}%")

    acc = correct / total * 100
    print(f"  [Winogrande] accuracy = {acc:.2f}% ({correct}/{total})")
    return acc


# ── output ────────────────────────────────────────────────────────────────────

def print_speed_results(all_results: list[ModelResults]) -> None:
    for mr in all_results:
        print()
        print("=" * 130)
        print(
            f"  {mr.model_name}  |  Prompt={PROMPT_TOKENS} tokens  |  "
            f"Decode={DECODE_TOKENS} tokens  |  Latency runs={LATENCY_RUNS}"
        )
        print("=" * 130)
        hdr = (
            f"  {'Quant':<10}  {'Prefill':>9}  {'Decode':>9}  "
            f"{'TTFT (ms)':>22}  {'TPOT (ms)':>22}  "
            f"{'Latency (ms)':>25}  {'Weight':>9}  {'Peak':>9}"
        )
        sub = (
            f"  {'':10}  {'(t/s)':>9}  {'(t/s)':>9}  "
            f"{'mean ± std':>22}  {'mean ± std':>22}  "
            f"{'mean ± std':>25}  {'(GiB)':>9}  {'(GiB)':>9}"
        )
        print(hdr)
        print(sub)
        print("-" * 130)
        for r in mr.speed:
            print(
                f"  {r.quant:<10}  {r.prefill_toks_s:>9.0f}  {r.decode_toks_s:>9.1f}  "
                f"  {r.ttft_mean:>8.1f} ± {r.ttft_std:>6.1f}  "
                f"  {r.tpot_mean:>8.2f} ± {r.tpot_std:>6.2f}  "
                f"  {r.lat_mean:>10.1f} ± {r.lat_std:>8.1f}  "
                f"  {r.weight_vram_gib:>9.2f}  {r.peak_vram_gib:>9.2f}"
            )
        print("=" * 130)

    # ── LaTeX speed table ────────────────────────────────────────────────
    print()
    print(r"""\begin{table*}[t]
  \centering
  \caption{%
    Inference speed and memory usage for three open-weight LLMs at multiple
    quantization levels, measured on NVIDIA~L40S GPU
    using GPTQModel (GGUF backend).
    \textbf{Prefill} = prompt-processing throughput at 512 input tokens
    (PP512, tokens/s);
    \textbf{Decode} = text-generation throughput at 128 output tokens
    (TG128, tokens/s);
    \textbf{TTFT} = time-to-first-token;
    \textbf{TPOT} = time-per-output-token;
    \textbf{Latency} = TTFT\,+\,TPOT (mean of """ + str(LATENCY_RUNS) + r""" streaming runs,
    512-token prompt, 128 output tokens, temperature~0);
    \textbf{Weight} = model-weight VRAM footprint;
    \textbf{Peak} = maximum VRAM during benchmark.%
  }
  \label{tab:gptqmodel_speed_memory}
  \resizebox{\linewidth}{!}{%
  \begin{tabular}{@{} l l rr rrr rr @{}}
    \toprule
    \multirow{2}{*}{\textbf{Model}} &
    \multirow{2}{*}{\textbf{Quant}} &
    \multicolumn{2}{c}{\textbf{Throughput (t/s)}} &
    \multicolumn{3}{c}{\textbf{Latency (ms)}} &
    \multicolumn{2}{c}{\textbf{VRAM (GiB)}} \\
    \cmidrule(lr){3-4}\cmidrule(lr){5-7}\cmidrule(lr){8-9}
    & &
    \textbf{Prefill} & \textbf{Decode} &
    \textbf{TTFT}   & \textbf{TPOT}  & \textbf{Total} &
    \textbf{Weight} & \textbf{Peak}  \\""")

    for mi, mr in enumerate(all_results):
        print(r"    \midrule")
        for i, r in enumerate(mr.speed):
            quant = r.quant.replace("_", r"\_")
            name_col = mr.model_name if i == 0 else " " * len(mr.model_name)
            print(
                f"    {name_col} & {quant} & "
                f"{r.prefill_toks_s:.0f} & {r.decode_toks_s:.1f} & "
                f"${r.ttft_mean:.1f}\\pm{r.ttft_std:.1f}$ & "
                f"${r.tpot_mean:.2f}\\pm{r.tpot_std:.2f}$ & "
                f"${r.lat_mean:.1f}\\pm{r.lat_std:.1f}$ & "
                f"{r.weight_vram_gib:.2f} & {r.peak_vram_gib:.2f} \\\\"
            )

    print(r"""    \bottomrule
  \end{tabular}%
  }
\end{table*}""")


def print_quality_results(all_results: list[ModelResults]) -> None:
    has_quality = any(mr.quality for mr in all_results)
    if not has_quality:
        return

    # ── ASCII table ──────────────────────────────────────────────────────
    print()
    print("=" * 80)
    print("  Quality Metrics")
    print("=" * 80)
    print(f"  {'Model':<25}  {'Quant':<10}  {'PPL':>18}  {'HellaSwag':>12}  {'Winogrande':>12}")
    print("-" * 90)
    for mr in all_results:
        for q in mr.quality:
            ppl_str = f"{q.ppl:.4f} ± {q.ppl_std:.5f}"
            print(
                f"  {mr.model_short:<25}  {q.quant:<10}  {ppl_str:>18}  "
                f"{q.hellaswag:>11.2f}%  {q.winogrande:>11.2f}%"
            )
    print("=" * 90)

    # ── LaTeX quality table ──────────────────────────────────────────────
    print()
    print(r"""\begin{table}[t]
  \centering
  \caption{%
    Model quality metrics at multiple quantization levels.
    \textbf{PPL} = perplexity on the Wikitext-2 test set
    (lower is better).
    \textbf{HellaSwag} = accuracy on """ + str(HELLASWAG_SAMPLES) + r""" commonsense-NLI tasks from the
    HellaSwag validation set (higher is better).
    \textbf{Winogrande} = accuracy on 1{,}267 debiased pronoun-resolution
    tasks (higher is better).
    Both accuracy benchmarks are evaluated via log-likelihood ranking.%
  }
  \label{tab:gptqmodel_quality}
  \begin{tabular}{@{} l l r rr @{}}
    \toprule
    \textbf{Model} & \textbf{Quant} &
    \textbf{PPL\,$\downarrow$} &
    \textbf{HellaSwag (\%)\,$\uparrow$} &
    \textbf{Winogrande (\%)\,$\uparrow$} \\""")

    for mi, mr in enumerate(all_results):
        print(r"    \midrule")
        for i, q in enumerate(mr.quality):
            quant = q.quant.replace("_", r"\_")
            name_col = mr.model_short if i == 0 else " " * len(mr.model_short)
            print(
                f"    {name_col} & {quant} & "
                f"${q.ppl:.2f}\\pm{q.ppl_std:.2f}$ & {q.hellaswag:.2f} & {q.winogrande:.2f} \\\\"
            )

    print(r"""    \bottomrule
  \end{tabular}
\end{table}""")


# ── quantization helpers ──────────────────────────────────────────────────────

def _quantize_gguf(
    model_id: str,
    out_dir: Path,
    bits: int,
    fmt: str,
    quant_backend,
) -> None:
    from gptqmodel import BACKEND, GGUFConfig, GPTQModel

    out_dir.mkdir(parents=True, exist_ok=True)
    qcfg = GGUFConfig(bits=bits, format=fmt)
    print(f"  Loading base model for quantization…")
    q_model = GPTQModel.load(model_id, qcfg)
    print(f"  Quantizing (backend={quant_backend.value})…")
    q_model.quantize(calibration=None, backend=quant_backend)
    print(f"  Saving to {out_dir}…")
    q_model.save(str(out_dir))
    del q_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _load_gguf(path: Path, device: str, backend) -> tuple:
    from gptqmodel import GPTQModel

    gptq_model = GPTQModel.load(
        str(path),
        device=device,
        backend=backend,
    )
    return gptq_model, gptq_model.model


# ── per-model benchmark ──────────────────────────────────────────────────────

def benchmark_model(
    model_id: str,
    model_name: str,
    model_short: str,
    work_dir: Path,
    device: str,
    args: argparse.Namespace,
) -> ModelResults:
    from gptqmodel import BACKEND

    print(f"\n{'='*80}")
    print(f"  MODEL: {model_name} ({model_id})")
    print(f"{'='*80}")

    q8_dir = work_dir / "gguf_q8_0"
    q4_dir = work_dir / "gguf_q4_k_m"

    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    prompt, n_prompt = build_prompt(tokenizer, PROMPT_TOKENS)
    print(f"Prompt: {n_prompt} tokens  (target: {PROMPT_TOKENS})")

    speed_results: list[BenchResult] = []
    quality_results: list[QualityResult] = []

    variants = [
        dict(
            label="F16",
            is_f16=True,
        ),
        dict(
            label="Q8_0",
            bits=8,
            fmt="q8_0",
            out_dir=q8_dir,
            quant_backend=BACKEND.GGUF_TORCH,
            infer_backend=BACKEND.GGUF_TRITON,
            is_f16=False,
        ),
        dict(
            label="Q4_K_M",
            bits=4,
            fmt="q4_k_m",
            out_dir=q4_dir,
            quant_backend=BACKEND.GGUF_TORCH,
            infer_backend=BACKEND.GGUF_TRITON,
            is_f16=False,
        ),
    ]

    for v in variants:
        label = v["label"]
        is_f16 = v["is_f16"]

        if is_f16 and args.skip_f16:
            continue

        print(f"\n── {label} {'─'*60}")

        if is_f16:
            torch.cuda.empty_cache()
            _reset_peak(device)

            import transformers.modeling_utils as _tmu
            _orig_warmup = getattr(_tmu, "caching_allocator_warmup", None)
            if _orig_warmup is not None:
                _tmu.caching_allocator_warmup = lambda *a, **kw: None

            hf_model = AutoModelForCausalLM.from_pretrained(
                model_id,
                dtype=torch.bfloat16,
                device_map=device,
            )

            if _orig_warmup is not None:
                _tmu.caching_allocator_warmup = _orig_warmup
            hf_model.eval()
            gptq_model = None
        else:
            out_dir: Path = v["out_dir"]
            quant_config_path = out_dir / "quantize_config.json"
            if not args.skip_quantize or not quant_config_path.exists():
                print(f"  Quantizing to {label}…")
                _quantize_gguf(
                    model_id, out_dir,
                    bits=v["bits"], fmt=v["fmt"],
                    quant_backend=v["quant_backend"],
                )

            print(f"  Loading {label} (inference backend: {v['infer_backend'].value})…")
            torch.cuda.empty_cache()
            _reset_peak(device)
            gptq_model, hf_model = _load_gguf(out_dir, device, v["infer_backend"])

        _sync(device)
        weight_vram = _weight_vram_gib(device)
        print(f"  Weight VRAM after load: {weight_vram:.2f} GiB")

        # Speed benchmark
        if not args.skip_speed:
            enc = tokenizer(prompt, return_tensors="pt").to(device)
            result = run_benchmark(
                label, hf_model, enc, device,
                warmup=args.warmup, trials=args.trials,
                weight_vram_gib=weight_vram,
            )
            speed_results.append(result)

        # Quality evaluation
        if args.eval:
            print(f"\n  ── Quality evaluation for {label} ──")
            ppl, ppl_std = compute_ppl(hf_model, tokenizer, device)
            hellaswag_acc = compute_hellaswag(hf_model, tokenizer, device)
            winogrande_acc = compute_winogrande(hf_model, tokenizer, device)
            quality_results.append(QualityResult(
                quant=label,
                ppl=ppl,
                ppl_std=ppl_std,
                hellaswag=hellaswag_acc,
                winogrande=winogrande_acc,
            ))

        _free_model(gptq_model, hf_model)

    return ModelResults(
        model_name=model_name,
        model_short=model_short,
        speed=speed_results,
        quality=quality_results,
    )


# ── main ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--models", default="llama",
        help="Comma-separated model keys to benchmark: llama,qwen,gemma (default: llama)",
    )
    p.add_argument("--work-dir", default="/tmp/gguf_bench",
                   help="Root directory for quantized model artifacts.")
    p.add_argument("--device", default="cuda:0",
                   help="Torch device for inference (e.g. 'cuda:0').")
    p.add_argument("--warmup", type=int, default=WARMUP_RUNS)
    p.add_argument("--trials", type=int, default=LATENCY_RUNS)
    p.add_argument("--skip-quantize", action="store_true",
                   help="Skip quantization if artifacts already exist.")
    p.add_argument("--skip-f16", action="store_true",
                   help="Skip the F16 baseline benchmark.")
    p.add_argument("--skip-speed", action="store_true",
                   help="Skip speed benchmarks (useful when only running --eval).")
    p.add_argument("--eval", action="store_true",
                   help="Run quality evaluation (PPL, HellaSwag, Winogrande).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = args.device

    if not torch.cuda.is_available():
        print("WARNING: CUDA not available — results will reflect CPU performance.")

    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(_gpu_idx(device))
        sm_major, sm_minor = torch.cuda.get_device_capability(_gpu_idx(device))
        print(f"GPU: {gpu_name}  (sm_{sm_major}{sm_minor})")

    model_keys = [k.strip() for k in args.models.split(",")]
    for k in model_keys:
        if k not in MODELS:
            print(f"ERROR: Unknown model key '{k}'. Available: {', '.join(MODELS)}")
            sys.exit(1)

    all_results: list[ModelResults] = []

    for key in model_keys:
        m = MODELS[key]
        model_work_dir = Path(args.work_dir) / key
        mr = benchmark_model(
            model_id=m["id"],
            model_name=m["name"],
            model_short=m["short"],
            work_dir=model_work_dir,
            device=device,
            args=args,
        )
        all_results.append(mr)

    # ── summary ──────────────────────────────────────────────────────────
    if not args.skip_speed:
        print_speed_results(all_results)
    if args.eval:
        print_quality_results(all_results)


if __name__ == "__main__":
    main()
