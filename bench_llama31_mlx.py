#!/usr/bin/env python3
"""
Benchmark open-weight LLMs at multiple MLX quantization levels on Apple Silicon.

Mirrors bench_llama31_gguf.py section-for-section, replacing the CUDA /
HuggingFace / GPTQModel stack with mlx_lm. Same benchmarking procedures:
llama.cpp-style non-overlapping PPL, length-normalized HellaSwag /
Winogrande scoring, explicit prefill + manual KV-cache decode loop.

Speed metrics (matching llama.cpp llama-bench convention):
  Prefill  = PP512  tokens/s
  Decode   = TG128  tokens/s
  TTFT     = time-to-first-token, ms
  TPOT     = time-per-output-token, ms
  Latency  = TTFT + TPOT, ms
  Weight   = unified memory allocated by model weights after load, GiB
  Peak     = max unified memory allocated during benchmark, GiB

Quality metrics:
  PPL         = perplexity on Wikitext-2 test set (lower is better)
  HellaSwag   = accuracy on 400 commonsense-NLI tasks (higher is better)
  Winogrande  = accuracy on 1,267 pronoun-resolution tasks (higher is better)

Usage:
  python bench_llama31_mlx.py --models llama
  python bench_llama31_mlx.py --models llama,qwen,gemma --eval
"""

from __future__ import annotations

import argparse
import gc
import math
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load as mlx_load
from mlx_lm.models.cache import make_prompt_cache

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
        "name": "Llama-3.1-8B-Instruct",
        "short": "Llama-3.1-8B",
        "fp16_id": "meta-llama/Llama-3.1-8B-Instruct",
        "mlx_8bit_id": "mlx-community/Meta-Llama-3.1-8B-Instruct-8bit",
        "mlx_4bit_id": "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit",
    },
    "qwen": {
        "name": "Qwen2.5-7B-Instruct",
        "short": "Qwen2.5-7B",
        "fp16_id": "Qwen/Qwen2.5-7B-Instruct",
        "mlx_8bit_id": "mlx-community/Qwen2.5-7B-Instruct-8bit",
        "mlx_4bit_id": "mlx-community/Qwen2.5-7B-Instruct-4bit",
    },
    "gemma": {
        "name": "Gemma-2-9B-IT",
        "short": "Gemma-2-9B",
        "fp16_id": "google/gemma-2-9b-it",
        "mlx_8bit_id": "mlx-community/gemma-2-9b-it-8bit",
        "mlx_4bit_id": "mlx-community/gemma-2-9b-it-4bit",
    },
}


# ── helpers ───────────────────────────────────────────────────────────────────

def _sync() -> None:
    """Force all pending Metal compute. MLX is lazy; eval() is the only flush."""
    mx.eval()


def _weight_mem_gib() -> float:
    return mx.get_peak_memory() / 1024**3


def _peak_mem_gib() -> float:
    return mx.get_peak_memory() / 1024**3


def _reset_peak() -> None:
    mx.reset_peak_memory()


def _tokenize_ids(tokenizer, text: str, add_special_tokens: bool = True) -> list[int]:
    """Return raw token ids, handling both HF tokenizers and mlx_lm's TokenizerWrapper."""
    inner = getattr(tokenizer, "_tokenizer", tokenizer)
    try:
        return inner.encode(text, add_special_tokens=add_special_tokens)
    except TypeError:
        out = inner.encode(text)
        return list(out.ids) if hasattr(out, "ids") else list(out)


def build_prompt(tokenizer, target_tokens: int) -> tuple[str, int]:
    prompt = _FILL_SENTENCE
    while True:
        n = len(_tokenize_ids(tokenizer, prompt))
        if n >= target_tokens:
            return prompt, n
        prompt += _FILL_SENTENCE


def _free_model(*models) -> None:
    for _ in models:
        pass  # names go out of scope in caller; keep signature parity with GGUF script
    gc.collect()


# ── result containers ────────────────────────────────────────────────────────

@dataclass
class BenchResult:
    quant: str
    prefill_toks_s: float
    decode_toks_s: float
    ttft_ms: list[float]
    tpot_ms: list[float]
    latency_ms: list[float]
    weight_mem_gib: float
    peak_mem_gib: float

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

def _prefill_once(model, input_ids: mx.array) -> float:
    _sync()
    t0 = time.perf_counter()
    cache = make_prompt_cache(model)
    logits = model(input_ids, cache=cache)
    mx.eval(logits)
    return (time.perf_counter() - t0) * 1e3


def _streaming_once(
    model, input_ids: mx.array, decode_tokens: int
) -> tuple[float, float, float]:
    _sync()
    t0 = time.perf_counter()
    cache = make_prompt_cache(model)
    logits = model(input_ids, cache=cache)
    mx.eval(logits)
    ttft_ms = (time.perf_counter() - t0) * 1e3

    next_tok = mx.argmax(logits[:, -1, :], axis=-1, keepdims=True)
    mx.eval(next_tok)

    t1 = time.perf_counter()
    for _ in range(decode_tokens - 1):
        logits = model(next_tok, cache=cache)
        next_tok = mx.argmax(logits[:, -1, :], axis=-1, keepdims=True)
        mx.eval(next_tok)
    decode_total_ms = (time.perf_counter() - t1) * 1e3

    tpot_ms = decode_total_ms / (decode_tokens - 1)
    return ttft_ms, tpot_ms, ttft_ms + tpot_ms


def run_benchmark(
    label: str,
    model,
    input_ids: mx.array,
    warmup: int = WARMUP_RUNS,
    trials: int = LATENCY_RUNS,
    weight_mem_gib: float = 0.0,
) -> BenchResult:
    n_prompt = input_ids.shape[1]

    print(f"  [{label}] warming up ({warmup} run(s))…")
    for _ in range(warmup):
        _streaming_once(model, input_ids, DECODE_TOKENS)
    _reset_peak()

    print(f"  [{label}] measuring prefill throughput…")
    prefill_times_ms: list[float] = []
    for i in range(trials):
        ms = _prefill_once(model, input_ids)
        prefill_times_ms.append(ms)
        print(f"    run {i+1}/{trials}: {ms:.1f} ms  ({n_prompt/(ms/1e3):.0f} t/s)")

    prefill_mean_ms = statistics.mean(prefill_times_ms)
    prefill_toks_s = n_prompt / (prefill_mean_ms / 1e3)

    print(f"  [{label}] measuring streaming latency ({trials} run(s))…")
    ttft_list: list[float] = []
    tpot_list: list[float] = []
    lat_list: list[float] = []

    for i in range(trials):
        ttft, tpot, total = _streaming_once(model, input_ids, DECODE_TOKENS)
        ttft_list.append(ttft)
        tpot_list.append(tpot)
        lat_list.append(total)
        decode_tps = 1e3 / tpot
        print(
            f"    run {i+1}/{trials}: "
            f"TTFT={ttft:.1f} ms  TPOT={tpot:.2f} ms  Total={total:.1f} ms  "
            f"Decode={decode_tps:.1f} t/s"
        )

    mean_tpot_ms = statistics.mean(tpot_list)
    decode_toks_s = 1e3 / mean_tpot_ms

    peak = _peak_mem_gib()
    print(
        f"  [{label}] Weight Mem={weight_mem_gib:.2f} GiB  "
        f"Peak Mem={peak:.2f} GiB"
    )

    return BenchResult(
        quant=label,
        prefill_toks_s=prefill_toks_s,
        decode_toks_s=decode_toks_s,
        ttft_ms=ttft_list,
        tpot_ms=tpot_list,
        latency_ms=lat_list,
        weight_mem_gib=weight_mem_gib,
        peak_mem_gib=peak,
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
    model, tokenizer,
    n_ctx: int = PPL_N_CTX, max_chunks: int = PPL_MAX_CHUNKS,
) -> tuple[float, float]:
    """Perplexity matching llama.cpp: non-overlapping n_ctx-token chunks,
    scoring only the last half of each chunk."""
    from datasets import load_dataset

    print("  [PPL] loading wikitext-2 test set…")
    test = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(test["text"])
    all_tokens = _tokenize_ids(tokenizer, text, add_special_tokens=True)
    seq_len = len(all_tokens)

    n_chunk_max = seq_len // n_ctx
    n_chunk = min(max_chunks, n_chunk_max)
    first = n_ctx // 2
    tokens_per_chunk = n_ctx - first - 1

    print(
        f"  [PPL] {seq_len} tokens, n_ctx={n_ctx}, {n_chunk} chunks, "
        f"scoring last {tokens_per_chunk} tokens per chunk"
    )

    bos_token_id = getattr(tokenizer, "bos_token_id", None)
    if bos_token_id is None:
        inner = getattr(tokenizer, "_tokenizer", tokenizer)
        bos_token_id = getattr(inner, "bos_token_id", None)

    nll_sum = 0.0
    nll2_sum = 0.0
    count = 0

    for i in range(n_chunk):
        start = i * n_ctx
        chunk_tokens = list(all_tokens[start:start + n_ctx])

        if bos_token_id is not None:
            chunk_tokens[0] = bos_token_id

        input_ids = mx.array(chunk_tokens)[None]
        logits = model(input_ids)
        mx.eval(logits)
        logits = logits[0]

        target_tokens = mx.array(all_tokens[start + first + 1:start + n_ctx])
        score_logits = logits[first:n_ctx - 1]

        log_probs = nn.log_softmax(score_logits, axis=-1)
        token_nlls = -mx.take_along_axis(
            log_probs, target_tokens[:, None], axis=-1
        ).squeeze(-1)
        mx.eval(token_nlls)

        nll_sum += float(mx.sum(token_nlls).item())
        nll2_sum += float(mx.sum(token_nlls * token_nlls).item())
        count += int(target_tokens.shape[0])

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


def _score_logits_sum_ll(
    model, tokens: list[int], score_start: int, score_end: int
) -> tuple[float, int]:
    """Forward-pass `tokens`, return (sum of log-likelihoods, n_scored) for
    predicting tokens[score_start+1 : score_end+1] from logits[score_start:score_end].
    Returns (-inf, 0) if the slice is empty."""
    if score_end <= score_start:
        return float("-inf"), 0

    input_ids = mx.array(tokens)[None]
    logits = model(input_ids)
    mx.eval(logits)
    logits = logits[0]

    score_logits = logits[score_start:score_end]
    targets = mx.array(tokens[score_start + 1:score_end + 1])

    log_probs = nn.log_softmax(score_logits, axis=-1)
    token_lls = mx.take_along_axis(
        log_probs, targets[:, None], axis=-1
    ).squeeze(-1)
    mx.eval(token_lls)

    return float(mx.sum(token_lls).item()), int(targets.shape[0])


def compute_hellaswag(
    model, tokenizer, max_samples: int = HELLASWAG_SAMPLES
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

        seq_tokens: list[list[int]] = []
        for ending in endings:
            full_text = ctx + " " + _preprocess_hellaswag(ending)
            seq_tokens.append(_tokenize_ids(tokenizer, full_text))

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
            seq_len = len(seq_tokens[s])
            score_start = max(common_prefix - 1, 0)
            score_end = seq_len - 1

            sum_ll, n_scored = _score_logits_sum_ll(
                model, seq_tokens[s], score_start, score_end
            )
            if n_scored == 0:
                continue

            avg_ll = sum_ll / n_scored
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


def compute_winogrande(model, tokenizer) -> float:
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

        seq_tokens: list[list[int]] = []
        n_bases: list[int] = []
        for choice in choices:
            full_text = first_part + choice + second_part
            seq_tokens.append(_tokenize_ids(tokenizer, full_text))

            base_text = first_part + choice
            n_bases.append(len(_tokenize_ids(tokenizer, base_text)))

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
            seq_len = len(seq_tokens[s])
            n_base = n_bases[s] if skip_choice else common_prefix
            last = 1 if (seq_len - n_base > 1) else 0

            score_start = n_base - 1
            score_end = seq_len - 1 - last

            sum_ll, _ = _score_logits_sum_ll(
                model, seq_tokens[s], score_start, score_end
            )
            if math.isinf(sum_ll):
                scores.append(float("-inf"))
                continue

            n_scored = seq_len - n_base - last
            avg_ll = sum_ll / n_scored
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
                f"  {r.weight_mem_gib:>9.2f}  {r.peak_mem_gib:>9.2f}"
            )
        print("=" * 130)

    # ── LaTeX speed table ────────────────────────────────────────────────
    print()
    print(r"""\begin{table*}[t]
  \centering
  \caption{%
    Inference speed and memory usage for three open-weight LLMs at multiple
    quantization levels, measured on Apple Silicon
    using mlx-lm.
    \textbf{Prefill} = prompt-processing throughput at 512 input tokens
    (PP512, tokens/s);
    \textbf{Decode} = text-generation throughput at 128 output tokens
    (TG128, tokens/s);
    \textbf{TTFT} = time-to-first-token;
    \textbf{TPOT} = time-per-output-token;
    \textbf{Latency} = TTFT\,+\,TPOT (mean of """ + str(LATENCY_RUNS) + r""" streaming runs,
    512-token prompt, 128 output tokens, temperature~0);
    \textbf{Weight} = model-weight memory footprint;
    \textbf{Peak} = maximum unified memory during benchmark.%
  }
  \label{tab:mlx_speed_memory}
  \resizebox{\linewidth}{!}{%
  \begin{tabular}{@{} l l rr rrr rr @{}}
    \toprule
    \multirow{2}{*}{\textbf{Model}} &
    \multirow{2}{*}{\textbf{Quant}} &
    \multicolumn{2}{c}{\textbf{Throughput (t/s)}} &
    \multicolumn{3}{c}{\textbf{Latency (ms)}} &
    \multicolumn{2}{c}{\textbf{Memory (GiB)}} \\
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
                f"{r.weight_mem_gib:.2f} & {r.peak_mem_gib:.2f} \\\\"
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
    Model quality metrics at multiple MLX quantization levels.
    \textbf{PPL} = perplexity on the Wikitext-2 test set
    (lower is better).
    \textbf{HellaSwag} = accuracy on """ + str(HELLASWAG_SAMPLES) + r""" commonsense-NLI tasks from the
    HellaSwag validation set (higher is better).
    \textbf{Winogrande} = accuracy on 1{,}267 debiased pronoun-resolution
    tasks (higher is better).
    Both accuracy benchmarks are evaluated via log-likelihood ranking.%
  }
  \label{tab:mlx_quality}
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

def _ensure_local_quant(source_fp16: str, out_dir: Path, bits: int) -> None:
    """Self-quantize via mlx_lm.convert if out_dir doesn't exist or is empty."""
    if out_dir.exists() and any(out_dir.iterdir()):
        return
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    print(f"  self-quantizing {source_fp16} to {bits}-bit at {out_dir}…")
    subprocess.run([
        sys.executable, "-m", "mlx_lm", "convert",
        "--hf-path", source_fp16,
        "-q",
        "--q-bits", str(bits),
        "--q-group-size", "64",
        "--mlx-path", str(out_dir),
    ], check=True)


def _resolve_and_load(
    variant_label: str,
    mlx_community_id: str | None,
    fp16_fallback: str | None,
    bits: int | None,
    local_quant_dir: Path | None,
) -> tuple[object, object]:
    """Try mlx-community pre-quantized first; on failure, self-quantize from
    fp16_fallback into local_quant_dir. Returns (model, tokenizer)."""
    if mlx_community_id is not None:
        try:
            print(f"  Loading {variant_label}: {mlx_community_id}")
            return mlx_load(mlx_community_id)
        except Exception as e:
            print(f"  warning: mlx-community load failed ({e}); falling back to self-quantize")

    if fp16_fallback is None or local_quant_dir is None or bits is None:
        raise RuntimeError(f"no fallback available for {variant_label}")

    _ensure_local_quant(fp16_fallback, local_quant_dir, bits)
    print(f"  Loading {variant_label}: {local_quant_dir}")
    return mlx_load(str(local_quant_dir))


# ── per-model benchmark ──────────────────────────────────────────────────────

def benchmark_model(
    model_cfg: dict,
    work_dir: Path,
    args: argparse.Namespace,
) -> ModelResults:
    model_name = model_cfg["name"]
    model_short = model_cfg["short"]

    print(f"\n{'='*80}")
    print(f"  MODEL: {model_name}")
    print(f"{'='*80}")

    fp16_id = model_cfg["fp16_id"]
    q8_dir = work_dir / "mlx_8bit"
    q4_dir = work_dir / "mlx_4bit"

    variants = [
        dict(
            label="FP16",
            is_fp16=True,
            mlx_id=fp16_id,
            fallback_fp16=None,
            bits=None,
            local_dir=None,
        ),
        dict(
            label="MLX_8bit",
            is_fp16=False,
            mlx_id=model_cfg.get("mlx_8bit_id"),
            fallback_fp16=fp16_id,
            bits=8,
            local_dir=q8_dir,
        ),
        dict(
            label="MLX_4bit",
            is_fp16=False,
            mlx_id=model_cfg.get("mlx_4bit_id"),
            fallback_fp16=fp16_id,
            bits=4,
            local_dir=q4_dir,
        ),
    ]

    # Build the prompt once using the FP16 tokenizer (quant variants share the
    # same tokenizer family; tokenization is model-family-stable).
    # We load the tokenizer lazily from the first variant that loads.
    prompt: str | None = None
    n_prompt: int = 0

    speed_results: list[BenchResult] = []
    quality_results: list[QualityResult] = []

    for v in variants:
        label = v["label"]
        if v["is_fp16"] and args.skip_f16:
            continue

        print(f"\n── {label} {'─'*60}")

        _reset_peak()
        model, tokenizer = _resolve_and_load(
            label,
            mlx_community_id=v["mlx_id"],
            fp16_fallback=v["fallback_fp16"],
            bits=v["bits"],
            local_quant_dir=v["local_dir"],
        )
        mx.eval(model.parameters())
        weight_mem = _weight_mem_gib()
        print(f"  Weight memory after load: {weight_mem:.2f} GiB")

        if prompt is None:
            prompt, n_prompt = build_prompt(tokenizer, PROMPT_TOKENS)
            print(f"  Prompt: {n_prompt} tokens  (target: {PROMPT_TOKENS})")

        _reset_peak()

        # Speed benchmark
        if not args.skip_speed:
            input_ids = mx.array(_tokenize_ids(tokenizer, prompt))[None]
            result = run_benchmark(
                label, model, input_ids,
                warmup=args.warmup, trials=args.trials,
                weight_mem_gib=weight_mem,
            )
            speed_results.append(result)

        # Quality evaluation
        if args.eval:
            print(f"\n  ── Quality evaluation for {label} ──")
            ppl, ppl_std = compute_ppl(model, tokenizer)
            hellaswag_acc = compute_hellaswag(model, tokenizer)
            winogrande_acc = compute_winogrande(model, tokenizer)
            quality_results.append(QualityResult(
                quant=label,
                ppl=ppl,
                ppl_std=ppl_std,
                hellaswag=hellaswag_acc,
                winogrande=winogrande_acc,
            ))

        del model, tokenizer
        _free_model()

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
    p.add_argument("--work-dir", default="/tmp/mlx_bench",
                   help="Root directory for self-quantized model artifacts.")
    p.add_argument("--warmup", type=int, default=WARMUP_RUNS)
    p.add_argument("--trials", type=int, default=LATENCY_RUNS)
    p.add_argument("--skip-quantize", action="store_true",
                   help="Skip self-quantization fallback if artifacts already exist. "
                        "(Self-quant already no-ops when the target dir is populated; "
                        "this flag is kept for CLI parity with the CUDA script.)")
    p.add_argument("--skip-f16", action="store_true",
                   help="Skip the FP16 baseline benchmark.")
    p.add_argument("--skip-speed", action="store_true",
                   help="Skip speed benchmarks (useful when only running --eval).")
    p.add_argument("--eval", action="store_true",
                   help="Run quality evaluation (PPL, HellaSwag, Winogrande).")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    try:
        from importlib.metadata import version as _pkg_version
        mlx_lm_version = _pkg_version("mlx-lm")
    except Exception:
        mlx_lm_version = "unknown"
    print(f"MLX default device: {mx.default_device()}  |  mlx-lm version: {mlx_lm_version}")

    model_keys = [k.strip() for k in args.models.split(",")]
    for k in model_keys:
        if k not in MODELS:
            print(f"ERROR: Unknown model key '{k}'. Available: {', '.join(MODELS)}")
            sys.exit(1)

    all_results: list[ModelResults] = []

    for key in model_keys:
        m = MODELS[key]
        model_work_dir = Path(args.work_dir) / key
        mr = benchmark_model(m, model_work_dir, args)
        all_results.append(mr)

    if not args.skip_speed:
        print_speed_results(all_results)
    if args.eval:
        print_quality_results(all_results)


if __name__ == "__main__":
    main()
