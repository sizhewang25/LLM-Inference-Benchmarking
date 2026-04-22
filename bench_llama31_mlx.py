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
  python bench_llama31_mlx.py --models llama,qwen,gemma --variants mxfp8,mxfp4
  python bench_llama31_mlx.py --models llama --variants fp16,8bit,4bit,mxfp8,mxfp4 --eval

Note on mxfp8 / mxfp4:
  Pre-quantize first with quantize_mlx.py; the bench script loads from
  ./models/<model-short>_mxfp8 (or mxfp4) and auto-quantizes on first run
  if the directory is absent.
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import math
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load as mlx_load
from mlx_lm.models.cache import make_prompt_cache

from bench_utils import setup_run_logging

log = logging.getLogger("bench")

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
        "fp16_id": "mlx-community/Meta-Llama-3.1-8B-Instruct-bf16",
        "mlx_8bit_id": "mlx-community/Meta-Llama-3.1-8B-Instruct-8bit",
        "mlx_4bit_id": "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit",
        "mlx_2bit_id": None,
        "mxfp8_id": None,
        "mxfp4_id": None,
    },
    "qwen": {
        "name": "Qwen2.5-7B-Instruct",
        "short": "Qwen2.5-7B",
        "fp16_id": "Qwen/Qwen2.5-7B-Instruct",
        "mlx_8bit_id": "mlx-community/Qwen2.5-7B-Instruct-8bit",
        "mlx_4bit_id": "mlx-community/Qwen2.5-7B-Instruct-4bit",
        "mlx_2bit_id": None,
        "mxfp8_id": None,
        "mxfp4_id": None,
    },
    "gemma": {
        "name": "Gemma-2-9B-IT",
        "short": "Gemma-2-9B",
        "fp16_id": "mlx-community/gemma-2-9b-it-fp16",
        "mlx_8bit_id": "mlx-community/gemma-2-9b-it-8bit",
        "mlx_4bit_id": "mlx-community/gemma-2-9b-it-4bit",
        "mlx_2bit_id": None,
        "mxfp8_id": None,
        "mxfp4_id": None,
    },
}


# ── helpers ───────────────────────────────────────────────────────────────────

def _sync() -> None:
    """Force all pending Metal compute. MLX is lazy; eval() is the only flush."""
    mx.eval()


def _weight_mem_gib() -> float:
    """Active (live) memory right after model load — reflects weight footprint."""
    return mx.get_active_memory() / 1024**3


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
    mx.clear_cache()  # release MLX's Metal buffer pool back to the OS


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

    log.info(f"  [{label}] warming up ({warmup} run(s))…")
    for _ in range(warmup):
        _streaming_once(model, input_ids, DECODE_TOKENS)
    _reset_peak()

    log.info(f"  [{label}] measuring prefill throughput…")
    prefill_times_ms: list[float] = []
    for i in range(trials):
        ms = _prefill_once(model, input_ids)
        prefill_times_ms.append(ms)
        log.info(f"    run {i+1}/{trials}: {ms:.1f} ms  ({n_prompt/(ms/1e3):.0f} t/s)")

    prefill_mean_ms = statistics.mean(prefill_times_ms)
    prefill_toks_s = n_prompt / (prefill_mean_ms / 1e3)

    log.info(f"  [{label}] measuring streaming latency ({trials} run(s))…")
    ttft_list: list[float] = []
    tpot_list: list[float] = []
    lat_list: list[float] = []

    for i in range(trials):
        ttft, tpot, total = _streaming_once(model, input_ids, DECODE_TOKENS)
        ttft_list.append(ttft)
        tpot_list.append(tpot)
        lat_list.append(total)
        decode_tps = 1e3 / tpot
        log.info(
            f"    run {i+1}/{trials}: "
            f"TTFT={ttft:.1f} ms  TPOT={tpot:.2f} ms  Total={total:.1f} ms  "
            f"Decode={decode_tps:.1f} t/s"
        )

    mean_tpot_ms = statistics.mean(tpot_list)
    decode_toks_s = 1e3 / mean_tpot_ms

    peak = _peak_mem_gib()
    log.info(
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

    log.info("  [PPL] loading wikitext-2 test set…")
    test = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(test["text"])
    all_tokens = _tokenize_ids(tokenizer, text, add_special_tokens=True)
    seq_len = len(all_tokens)

    n_chunk_max = seq_len // n_ctx
    n_chunk = min(max_chunks, n_chunk_max)
    first = n_ctx // 2
    tokens_per_chunk = n_ctx - first - 1

    log.info(
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
            log.info(f"    chunk {i+1}/{n_chunk}: running PPL={running_ppl:.4f}")

    nll_mean = nll_sum / count
    nll2_mean = nll2_sum / count
    ppl = math.exp(nll_mean)

    variance = nll2_mean - nll_mean ** 2
    if variance > 0:
        ppl_std = math.sqrt(variance / (count - 1)) * ppl
    else:
        ppl_std = 0.0

    log.info(f"  [PPL] Final estimate: PPL = {ppl:.4f} +/- {ppl_std:.5f}")
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

    log.info(f"  [HellaSwag] loading validation set (max {max_samples} samples)…")
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
            log.info(f"    {count_i+1}/{total}: acc={correct/(count_i+1)*100:.1f}%")

    acc = correct / total * 100
    log.info(f"  [HellaSwag] accuracy = {acc:.2f}% ({correct}/{total})")
    return acc


def compute_winogrande(model, tokenizer) -> float:
    """Winogrande accuracy matching llama.cpp: score only the continuation
    after the choice word, length-normalized."""
    from datasets import load_dataset

    log.info("  [Winogrande] loading validation set…")
    ds = load_dataset("allenai/winogrande", "winogrande_debiased", split="validation")
    total = len(ds)
    log.info(f"  [Winogrande] {total} samples")

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
            log.info(f"    {i+1}/{total}: acc={correct/(i+1)*100:.1f}%")

    acc = correct / total * 100
    log.info(f"  [Winogrande] accuracy = {acc:.2f}% ({correct}/{total})")
    return acc


# ── output ────────────────────────────────────────────────────────────────────

def _write_tex(run_dir: str | None, filename: str, content: str) -> None:
    if run_dir is None:
        return
    out_path = Path(run_dir) / filename
    out_path.write_text(content)
    log.info(f"  wrote {out_path}")


def _dump_variant_json(
    run_dir: str | None,
    model_name: str,
    model_short: str,
    speed: "BenchResult | None",
    quality: "QualityResult | None",
) -> None:
    """Write one JSON file per variant as soon as it finishes."""
    if run_dir is None:
        return
    quant = (speed or quality).quant
    slug = f"{model_short}_{quant}".lower().replace(" ", "_")
    payload: dict = {
        "model": model_name,
        "model_short": model_short,
        "quant": quant,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    if speed is not None:
        payload["speed"] = {
            "prefill_toks_s": round(speed.prefill_toks_s, 2),
            "decode_toks_s": round(speed.decode_toks_s, 2),
            "ttft_mean_ms": round(speed.ttft_mean, 2),
            "ttft_std_ms": round(speed.ttft_std, 2),
            "tpot_mean_ms": round(speed.tpot_mean, 3),
            "tpot_std_ms": round(speed.tpot_std, 3),
            "latency_mean_ms": round(speed.lat_mean, 2),
            "latency_std_ms": round(speed.lat_std, 2),
            "weight_mem_gib": round(speed.weight_mem_gib, 3),
            "peak_mem_gib": round(speed.peak_mem_gib, 3),
        }
    if quality is not None:
        payload["quality"] = {
            "ppl": round(quality.ppl, 4),
            "ppl_std": round(quality.ppl_std, 5),
            "hellaswag_pct": round(quality.hellaswag, 2),
            "winogrande_pct": round(quality.winogrande, 2),
        }
    out_path = Path(run_dir) / f"{slug}.json"
    out_path.write_text(json.dumps(payload, indent=2))
    log.info(f"  wrote {out_path}")


def _build_speed_latex(all_results: list[ModelResults]) -> str:
    lines: list[str] = []
    lines.append(r"\begin{table*}[t]")
    lines.append(r"  \centering")
    lines.append(r"  \caption{%")
    lines.append(r"    Inference speed and memory usage for three open-weight LLMs at multiple")
    lines.append(r"    quantization levels, measured on Apple Silicon")
    lines.append(r"    using mlx-lm.")
    lines.append(r"    \textbf{Prefill} = prompt-processing throughput at 512 input tokens")
    lines.append(r"    (PP512, tokens/s);")
    lines.append(r"    \textbf{Decode} = text-generation throughput at 128 output tokens")
    lines.append(r"    (TG128, tokens/s);")
    lines.append(r"    \textbf{TTFT} = time-to-first-token;")
    lines.append(r"    \textbf{TPOT} = time-per-output-token;")
    lines.append(
        r"    \textbf{Latency} = TTFT\,+\,TPOT (mean of " + str(LATENCY_RUNS)
        + r" streaming runs,"
    )
    lines.append(r"    512-token prompt, 128 output tokens, temperature~0);")
    lines.append(r"    \textbf{Weight} = model-weight memory footprint;")
    lines.append(r"    \textbf{Peak} = maximum unified memory during benchmark.%")
    lines.append(r"  }")
    lines.append(r"  \label{tab:mlx_speed_memory}")
    lines.append(r"  \resizebox{\linewidth}{!}{%")
    lines.append(r"  \begin{tabular}{@{} l l rr rrr rr @{}}")
    lines.append(r"    \toprule")
    lines.append(r"    \multirow{2}{*}{\textbf{Model}} &")
    lines.append(r"    \multirow{2}{*}{\textbf{Quant}} &")
    lines.append(r"    \multicolumn{2}{c}{\textbf{Throughput (t/s)}} &")
    lines.append(r"    \multicolumn{3}{c}{\textbf{Latency (ms)}} &")
    lines.append(r"    \multicolumn{2}{c}{\textbf{Memory (GiB)}} \\")
    lines.append(r"    \cmidrule(lr){3-4}\cmidrule(lr){5-7}\cmidrule(lr){8-9}")
    lines.append(r"    & &")
    lines.append(r"    \textbf{Prefill} & \textbf{Decode} &")
    lines.append(r"    \textbf{TTFT}   & \textbf{TPOT}  & \textbf{Total} &")
    lines.append(r"    \textbf{Weight} & \textbf{Peak}  \\")

    for mr in all_results:
        lines.append(r"    \midrule")
        for i, r in enumerate(mr.speed):
            quant = r.quant.replace("_", r"\_")
            name_col = mr.model_name if i == 0 else " " * len(mr.model_name)
            lines.append(
                f"    {name_col} & {quant} & "
                f"{r.prefill_toks_s:.0f} & {r.decode_toks_s:.1f} & "
                f"${r.ttft_mean:.1f}\\pm{r.ttft_std:.1f}$ & "
                f"${r.tpot_mean:.2f}\\pm{r.tpot_std:.2f}$ & "
                f"${r.lat_mean:.1f}\\pm{r.lat_std:.1f}$ & "
                f"{r.weight_mem_gib:.2f} & {r.peak_mem_gib:.2f} \\\\"
            )

    lines.append(r"    \bottomrule")
    lines.append(r"  \end{tabular}%")
    lines.append(r"  }")
    lines.append(r"\end{table*}")
    return "\n".join(lines) + "\n"


def _build_quality_latex(all_results: list[ModelResults]) -> str:
    lines: list[str] = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"  \centering")
    lines.append(r"  \caption{%")
    lines.append(r"    Model quality metrics at multiple MLX quantization levels.")
    lines.append(r"    \textbf{PPL} = perplexity on the Wikitext-2 test set")
    lines.append(r"    (lower is better).")
    lines.append(
        r"    \textbf{HellaSwag} = accuracy on " + str(HELLASWAG_SAMPLES)
        + r" commonsense-NLI tasks from the"
    )
    lines.append(r"    HellaSwag validation set (higher is better).")
    lines.append(r"    \textbf{Winogrande} = accuracy on 1{,}267 debiased pronoun-resolution")
    lines.append(r"    tasks (higher is better).")
    lines.append(r"    Both accuracy benchmarks are evaluated via log-likelihood ranking.%")
    lines.append(r"  }")
    lines.append(r"  \label{tab:mlx_quality}")
    lines.append(r"  \begin{tabular}{@{} l l r rr @{}}")
    lines.append(r"    \toprule")
    lines.append(r"    \textbf{Model} & \textbf{Quant} &")
    lines.append(r"    \textbf{PPL\,$\downarrow$} &")
    lines.append(r"    \textbf{HellaSwag (\%)\,$\uparrow$} &")
    lines.append(r"    \textbf{Winogrande (\%)\,$\uparrow$} \\")

    for mr in all_results:
        lines.append(r"    \midrule")
        for i, q in enumerate(mr.quality):
            quant = q.quant.replace("_", r"\_")
            name_col = mr.model_short if i == 0 else " " * len(mr.model_short)
            lines.append(
                f"    {name_col} & {quant} & "
                f"${q.ppl:.2f}\\pm{q.ppl_std:.2f}$ & {q.hellaswag:.2f} & {q.winogrande:.2f} \\\\"
            )

    lines.append(r"    \bottomrule")
    lines.append(r"  \end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines) + "\n"


def print_speed_results(
    all_results: list[ModelResults], run_dir: str | None = None
) -> None:
    for mr in all_results:
        log.info("")
        log.info("=" * 130)
        log.info(
            f"  {mr.model_name}  |  Prompt={PROMPT_TOKENS} tokens  |  "
            f"Decode={DECODE_TOKENS} tokens  |  Latency runs={LATENCY_RUNS}"
        )
        log.info("=" * 130)
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
        log.info(hdr)
        log.info(sub)
        log.info("-" * 130)
        for r in mr.speed:
            log.info(
                f"  {r.quant:<10}  {r.prefill_toks_s:>9.0f}  {r.decode_toks_s:>9.1f}  "
                f"  {r.ttft_mean:>8.1f} ± {r.ttft_std:>6.1f}  "
                f"  {r.tpot_mean:>8.2f} ± {r.tpot_std:>6.2f}  "
                f"  {r.lat_mean:>10.1f} ± {r.lat_std:>8.1f}  "
                f"  {r.weight_mem_gib:>9.2f}  {r.peak_mem_gib:>9.2f}"
            )
        log.info("=" * 130)

    # ── LaTeX speed table ────────────────────────────────────────────────
    latex = _build_speed_latex(all_results)
    print()
    print(latex, end="")
    _write_tex(run_dir, "speed_table.tex", latex)


def print_quality_results(
    all_results: list[ModelResults], run_dir: str | None = None
) -> None:
    has_quality = any(mr.quality for mr in all_results)
    if not has_quality:
        return

    # ── ASCII table ──────────────────────────────────────────────────────
    log.info("")
    log.info("=" * 80)
    log.info("  Quality Metrics")
    log.info("=" * 80)
    log.info(f"  {'Model':<25}  {'Quant':<10}  {'PPL':>18}  {'HellaSwag':>12}  {'Winogrande':>12}")
    log.info("-" * 90)
    for mr in all_results:
        for q in mr.quality:
            ppl_str = f"{q.ppl:.4f} ± {q.ppl_std:.5f}"
            log.info(
                f"  {mr.model_short:<25}  {q.quant:<10}  {ppl_str:>18}  "
                f"{q.hellaswag:>11.2f}%  {q.winogrande:>11.2f}%"
            )
    log.info("=" * 90)

    # ── LaTeX quality table ──────────────────────────────────────────────
    latex = _build_quality_latex(all_results)
    print()
    print(latex, end="")
    _write_tex(run_dir, "quality_table.tex", latex)


# ── quantization helpers ──────────────────────────────────────────────────────

def _ensure_local_quant(
    source_fp16: str, out_dir: Path, bits: int,
    group_size: int = 64, q_mode: str | None = None,
) -> None:
    """Self-quantize via mlx_lm.convert if out_dir doesn't exist or is empty."""
    if out_dir.exists():
        if any(out_dir.iterdir()):
            return  # already quantized
        out_dir.rmdir()  # empty dir from a prior failed run — mlx_lm requires it absent
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    mode_str = f" --q-mode {q_mode}" if q_mode else f" gs{group_size}"
    log.info(f"  self-quantizing {source_fp16} → {bits}-bit{mode_str} at {out_dir}…")
    cmd = [
        sys.executable, "-m", "mlx_lm", "convert",
        "--hf-path", source_fp16,
        "-q",
        "--q-bits", str(bits),
        "--mlx-path", str(out_dir),
    ]
    if q_mode is not None:
        # mxfp* modes have fixed group sizes; skip --q-group-size to use their defaults
        cmd += ["--q-mode", q_mode]
    else:
        cmd += ["--q-group-size", str(group_size)]
    subprocess.run(cmd, check=True)


def _resolve_and_load(
    variant_label: str,
    mlx_community_id: str | None,
    fp16_fallback: str | None,
    bits: int | None,
    local_quant_dir: Path | None,
    group_size: int = 64,
    q_mode: str | None = None,
) -> tuple[object, object]:
    """Try mlx-community pre-quantized first; on failure, self-quantize from
    fp16_fallback into local_quant_dir. Returns (model, tokenizer)."""
    if mlx_community_id is not None:
        try:
            log.info(f"  Loading {variant_label}: {mlx_community_id}")
            return mlx_load(mlx_community_id)
        except Exception as e:
            log.warning(f"  mlx-community load failed ({e}); falling back to self-quantize")

    if fp16_fallback is None or local_quant_dir is None or bits is None:
        raise RuntimeError(f"no fallback available for {variant_label}")

    _ensure_local_quant(fp16_fallback, local_quant_dir, bits, group_size, q_mode)
    log.info(f"  Loading {variant_label}: {local_quant_dir}")
    return mlx_load(str(local_quant_dir))


# ── per-model benchmark ──────────────────────────────────────────────────────

def benchmark_model(
    model_cfg: dict,
    work_dir: Path,
    args: argparse.Namespace,
    run_dir: str | None = None,
) -> ModelResults:
    model_name = model_cfg["name"]
    model_short = model_cfg["short"]

    log.info(f"\n{'='*80}")
    log.info(f"  MODEL: {model_name}")
    log.info(f"{'='*80}")

    fp16_id = model_cfg["fp16_id"]

    def _local(label: str) -> Path:
        # e.g. ./models/Llama-3.1-8B_mxfp8  or  ./models/Llama-3.1-8B_mlx_4bit
        return work_dir / f"{model_short}_{label.lower()}"

    variants = [
        dict(
            label="FP16",
            is_fp16=True,
            mlx_id=fp16_id,
            fallback_fp16=None,
            bits=None,
            local_dir=None,
            q_mode=None,
        ),
        dict(
            label="MLX_8bit",
            is_fp16=False,
            mlx_id=model_cfg.get("mlx_8bit_id"),
            fallback_fp16=fp16_id,
            bits=8,
            local_dir=_local("MLX_8bit"),
            q_mode=None,
        ),
        dict(
            label="MLX_4bit",
            is_fp16=False,
            mlx_id=model_cfg.get("mlx_4bit_id"),
            fallback_fp16=fp16_id,
            bits=4,
            local_dir=_local("MLX_4bit"),
            q_mode=None,
        ),
        dict(
            label="MLX_2bit",
            is_fp16=False,
            mlx_id=model_cfg.get("mlx_2bit_id"),
            fallback_fp16=fp16_id,
            bits=2,
            local_dir=_local("MLX_2bit"),
            q_mode=None,
        ),
        dict(
            label="MXFP8",
            is_fp16=False,
            mlx_id=model_cfg.get("mxfp8_id"),
            fallback_fp16=fp16_id,
            bits=8,
            local_dir=_local("MXFP8"),
            q_mode="mxfp8",
        ),
        dict(
            label="MXFP4",
            is_fp16=False,
            mlx_id=model_cfg.get("mxfp4_id"),
            fallback_fp16=fp16_id,
            bits=4,
            local_dir=_local("MXFP4"),
            q_mode="mxfp4",
        ),
    ]

    if args.variants is not None:
        requested = {v.strip().lower() for v in args.variants.split(",")}
        _label_map = {
            "fp16": "FP16",
            "8bit": "MLX_8bit",
            "4bit": "MLX_4bit",
            "2bit": "MLX_2bit",
            "mxfp8": "MXFP8",
            "mxfp4": "MXFP4",
        }
        allowed = {_label_map[r] for r in requested if r in _label_map}
        variants = [v for v in variants if v["label"] in allowed]

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

        log.info(f"\n── {label} {'─'*60}")

        _reset_peak()
        model, tokenizer = _resolve_and_load(
            label,
            mlx_community_id=v["mlx_id"],
            fp16_fallback=v["fallback_fp16"],
            bits=v["bits"],
            local_quant_dir=v["local_dir"],
            group_size=args.quant_group_size,
            q_mode=v["q_mode"],
        )
        mx.eval(model.parameters())
        weight_mem = _weight_mem_gib()
        log.info(f"  Weight memory after load: {weight_mem:.2f} GiB")

        if prompt is None:
            prompt, n_prompt = build_prompt(tokenizer, PROMPT_TOKENS)
            log.info(f"  Prompt: {n_prompt} tokens  (target: {PROMPT_TOKENS})")

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
            log.info(f"\n  ── Quality evaluation for {label} ──")
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

        _dump_variant_json(
            run_dir,
            model_name=model_name,
            model_short=model_short,
            speed=speed_results[-1] if speed_results and speed_results[-1].quant == label else None,
            quality=quality_results[-1] if quality_results and quality_results[-1].quant == label else None,
        )

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
    p.add_argument("--work-dir", default="./models",
                   help="Root directory for quantized model artifacts "
                        "(default: ./models). Each variant is saved as "
                        "<work-dir>/<model-short>_<variant>, e.g. "
                        "models/Llama-3.1-8B_mxfp8.")
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
    p.add_argument(
        "--variants", default=None,
        help=(
            "Comma-separated quant levels to run: fp16,8bit,4bit,2bit,mxfp8,mxfp4 "
            "(default: all). Example: --variants mxfp8,mxfp4"
        ),
    )
    p.add_argument(
        "--quant-group-size", type=int, default=64,
        choices=[32, 64, 128],
        help="Group size for self-quantization fallback (default: 64). "
             "MLX supports 32, 64, 128. Larger = coarser scales, closer to llama.cpp QK_K.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = setup_run_logging(__file__)

    try:
        from importlib.metadata import version as _pkg_version
        mlx_lm_version = _pkg_version("mlx-lm")
    except Exception:
        mlx_lm_version = "unknown"
    log.info(f"MLX default device: {mx.default_device()}  |  mlx-lm version: {mlx_lm_version}")

    model_keys = [k.strip() for k in args.models.split(",")]
    for k in model_keys:
        if k not in MODELS:
            log.error(f"Unknown model key '{k}'. Available: {', '.join(MODELS)}")
            sys.exit(1)

    all_results: list[ModelResults] = []

    for key in model_keys:
        m = MODELS[key]
        mr = benchmark_model(m, Path(args.work_dir), args, run_dir=run_dir)
        all_results.append(mr)

    if not args.skip_speed:
        print_speed_results(all_results, run_dir=run_dir)
    if args.eval:
        print_quality_results(all_results, run_dir=run_dir)


if __name__ == "__main__":
    main()
