import gc
import json
import logging
import math
import os
import re
import sys
import time
from datetime import datetime
from statistics import mean, stdev

import torch
from transformers import GenerationConfig
from transformers.generation.streamers import BaseStreamer

log = logging.getLogger("bench")


def setup_run_logging(script_name):
    """Configure the `bench` logger to write to both stderr and a per-run log file.

    Output layout, relative to the calling script's directory:

        outputs/<script_name>/<timestamp>/run.log
        outputs/<script_name>/<timestamp>/<label>.json   (written by dump_results)

    Returns the absolute path to the run directory so the caller can pass it
    to `dump_results`.
    """
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    base_dir = os.path.dirname(os.path.abspath(script_name))
    run_dir = os.path.join(base_dir, "outputs", os.path.basename(script_name).removesuffix(".py"), timestamp)
    os.makedirs(run_dir, exist_ok=True)

    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%H:%M:%S")

    root = logging.getLogger("bench")
    root.setLevel(logging.INFO)
    # Clear any handlers from a previous setup() call in the same process.
    for h in list(root.handlers):
        root.removeHandler(h)

    file_handler = logging.FileHandler(os.path.join(run_dir, "run.log"))
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    root.addHandler(stream_handler)

    # Route any uncaught exception through the logger so tracebacks land in
    # run.log instead of only reaching the terminal (where they may be lost if
    # the script was launched with output redirected).
    def _excepthook(exc_type, exc_value, exc_tb):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_tb)
            return
        log.error("uncaught exception", exc_info=(exc_type, exc_value, exc_tb))

    sys.excepthook = _excepthook

    log.info("run dir: %s", run_dir)
    return run_dir


def dump_results(run_dir, label, results):
    """Write a benchmark result dict to `<run_dir>/<slug>.json`."""
    slug = label.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("/", "_")
    path = os.path.join(run_dir, f"{slug}.json")
    payload = {
        "label": label,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "results": results,
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    log.info("wrote results json: %s", path)
    return path

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


def pick_device():
    """Return the best available accelerator string: cuda → mps → cpu."""
    if torch.cuda.is_available():
        return "cuda:0"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _is_cuda(device):
    return str(device).startswith("cuda")


def _is_mps(device):
    return str(device).startswith("mps")


def sync_device(device):
    """Block until queued GPU work for the given device is finished."""
    if _is_cuda(device) and torch.cuda.is_available():
        torch.cuda.synchronize()
    elif _is_mps(device) and torch.backends.mps.is_available():
        torch.mps.synchronize()


def get_peak_memory_mb(device="cuda:0"):
    if _is_cuda(device) and torch.cuda.is_available():
        return torch.cuda.max_memory_allocated(device) / 1024 / 1024
    if _is_mps(device) and torch.backends.mps.is_available():
        # MPS has no peak-tracking API; report current driver allocation as a
        # best-effort proxy. Will under-report if the peak occurred earlier.
        return torch.mps.driver_allocated_memory() / 1024 / 1024
    return 0.0


def reset_peak_memory(device="cuda:0"):
    if _is_cuda(device) and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)
    # MPS has no equivalent reset; silently no-op.


def free_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif torch.backends.mps.is_available():
        torch.mps.empty_cache()


def format_prompts(tokenizer, prompts):
    """Apply the tokenizer's chat template if one is set, else return raw prompts.

    Both benchmark scripts target instruct/chat-tuned models. Applying the
    template makes prefill length match realistic serving and is required for
    models like DeepSeek-R1-Distill that emit `<think>` reasoning blocks.
    """
    if not getattr(tokenizer, "chat_template", None):
        return list(prompts)
    return [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": p}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for p in prompts
    ]


def load_wikitext2_tokens(tokenizer):
    """Load WikiText-2 test split, concatenate, and tokenize.

    Returns a plain Python list[int] so the caller can convert to the
    framework-specific tensor type (torch.tensor or mx.array).

    Uses huggingface_hub (already a transitive dependency of transformers)
    to download the raw text, avoiding the heavy `datasets` library which
    requires lzma support that some pyenv Python builds lack.
    """
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(
        repo_id="Salesforce/wikitext",
        repo_type="dataset",
        filename="wikitext-2-raw-v1/test-00000-of-00001.parquet",
    )

    import pyarrow.parquet as pq

    table = pq.read_table(path)
    rows = table.column("text").to_pylist()
    text = "\n\n".join(line for line in rows if line.strip())

    # Handle both HuggingFace tokenizers (callable) and mlx_lm
    # TokenizerWrapper (delegates to ._tokenizer).
    inner = getattr(tokenizer, "_tokenizer", tokenizer)
    return inner.encode(text)


GSM8K_INSTRUCTION = (
    "Solve this math problem step by step. "
    "Put your final answer after ####.\n\n"
)


def load_gsm8k_questions(num_samples=100, seed=42):
    """Load GSM8K test questions via hf_hub_download + pyarrow.

    Returns a list of dicts: [{"question": str, "answer": float}, ...]
    The answer is the numeric value after #### in the original answer string.
    Uses a fixed seed for reproducible subset selection across runs.
    """
    import random

    from huggingface_hub import hf_hub_download
    import pyarrow.parquet as pq

    path = hf_hub_download(
        repo_id="openai/gsm8k",
        repo_type="dataset",
        filename="main/test-00000-of-00001.parquet",
    )
    table = pq.read_table(path)
    questions_raw = table.column("question").to_pylist()
    answers_raw = table.column("answer").to_pylist()

    items = []
    for q, a in zip(questions_raw, answers_raw):
        answer_num = float(a.split("####")[-1].strip().replace(",", ""))
        items.append({"question": q, "answer": answer_num})

    if num_samples < len(items):
        rng = random.Random(seed)
        items = rng.sample(items, num_samples)

    return items


def parse_gsm8k_answer(model_output):
    """Extract the final numeric answer from model output.

    Extraction priority:
      1. #### <number>  (what we instruct the model to produce)
      2. \\boxed{<number>}  (LaTeX, common in reasoning models like QwQ)
      3. Last number in the text  (fallback)

    Returns float or None if no number found.
    """
    # Priority 1: #### pattern
    match = re.search(r"####\s*([+-]?[\d,]+\.?\d*)", model_output)
    if match:
        return float(match.group(1).replace(",", ""))

    # Priority 2: \boxed{} pattern
    match = re.search(r"\\boxed\{([+-]?[\d,]+\.?\d*)\}", model_output)
    if match:
        return float(match.group(1).replace(",", ""))

    # Priority 3: last number in text
    numbers = re.findall(r"[+-]?[\d,]+\.?\d*", model_output)
    if numbers:
        return float(numbers[-1].replace(",", ""))

    return None


def compute_perplexity(model, token_ids, device, max_length=2048, stride=512):
    """Sliding-window perplexity over pre-tokenized text (PyTorch models).

    Works with both AutoModelForCausalLM and GPTQModel — both support
    forward(input_ids, labels=...) returning an output with .loss.
    """
    input_ids = torch.tensor(token_ids, dtype=torch.long).unsqueeze(0)
    seq_len = input_ids.size(1)

    nlls = []

    for begin in range(0, seq_len, stride):
        end = min(begin + max_length, seq_len)
        chunk = input_ids[:, begin:end].to(device)
        target = chunk.clone()

        # Mask the overlap region so tokens aren't double-counted.
        # Keep only the last `stride` tokens (or fewer for the final chunk).
        if begin > 0:
            num_keep = min(stride, end - begin)
            target[:, : end - begin - num_keep] = -100

        with torch.no_grad():
            outputs = model(chunk, labels=target)
            nll = outputs.loss.float().item()

        num_scored = (target != -100).sum().item()
        nlls.append((nll, num_scored))

        if end == seq_len:
            break

    total_nll = sum(nll * n for nll, n in nlls)
    total_tokens = sum(n for _, n in nlls)
    avg_nll = total_nll / total_tokens
    return math.exp(avg_nll)


def compute_perplexity_mlx(model, token_ids, max_length=2048, stride=512):
    """Sliding-window perplexity for MLX models (Apple Silicon).

    MLX models don't support a labels= shortcut, so we compute
    cross-entropy manually from logits.
    """
    import mlx.core as mx
    import mlx.nn as nn

    input_ids = mx.array(token_ids)
    seq_len = len(input_ids)

    nlls = []

    for begin in range(0, seq_len, stride):
        end = min(begin + max_length, seq_len)
        chunk = input_ids[begin:end]

        logits = model(chunk[None])  # (1, chunk_len, vocab_size)

        # Shift: logits[:, :-1] predict tokens at positions 1..end
        shift_logits = logits[:, :-1, :]
        shift_labels = chunk[1:]

        # Only score the stride portion (skip overlap context)
        if begin > 0:
            score_start = max_length - stride - 1
        else:
            score_start = 0

        score_logits = shift_logits[:, score_start:, :]
        score_labels = shift_labels[score_start:]

        score_logits_2d = score_logits.reshape(-1, score_logits.shape[-1])
        loss = nn.losses.cross_entropy(score_logits_2d, score_labels, reduction="sum")
        mx.eval(loss)

        num_scored = score_labels.size
        nlls.append((loss.item(), num_scored))

        if end == seq_len:
            break

    total_nll = sum(nll for nll, _ in nlls)
    total_tokens = sum(n for _, n in nlls)
    avg_nll = total_nll / total_tokens
    return math.exp(avg_nll)


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
    sync_device(device)

    # Measure
    reset_peak_memory(device)

    all_ttft = []
    all_latency = []
    all_tpot = []
    all_tokens = []
    tpot_skipped = 0

    for idx, prompt in enumerate(prompts):
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        input_len = inputs["input_ids"].shape[1]

        streamer = TTFTStreamer()
        sync_device(device)

        streamer.start_time = time.perf_counter()
        t_start = streamer.start_time

        with torch.no_grad():
            output_ids = model.generate(
                **inputs, generation_config=gen_config, streamer=streamer,
            )

        sync_device(device)
        t_end = time.perf_counter()

        new_tokens = output_ids.shape[1] - input_len
        latency = t_end - t_start

        ttft = (streamer.first_token_time - t_start) if streamer.first_token_time else latency

        all_ttft.append(ttft * 1000)
        all_latency.append(latency * 1000)
        all_tokens.append(new_tokens)

        if new_tokens < 2:
            tpot_skipped += 1
            log.warning(
                "skipped TPOT for prompt index %d (only %d token(s) generated)",
                idx, new_tokens,
            )
            continue

        decode_tokens = new_tokens - 1
        decode_time = latency - ttft
        tpot = decode_time / decode_tokens
        all_tpot.append(tpot * 1000)

    total_tokens = sum(all_tokens)
    total_time = sum(all_latency) / 1000

    return {
        "throughput_tok_s": total_tokens / total_time,
        "ttft_mean_ms": mean(all_ttft),
        "ttft_std_ms": stdev(all_ttft) if len(all_ttft) > 1 else 0,
        "latency_mean_ms": mean(all_latency),
        "latency_std_ms": stdev(all_latency) if len(all_latency) > 1 else 0,
        "tpot_mean_ms": mean(all_tpot) if all_tpot else 0,
        "tpot_std_ms": stdev(all_tpot) if len(all_tpot) > 1 else 0,
        "tpot_skipped": tpot_skipped,
        "peak_mem_mb": get_peak_memory_mb(device),
        "num_samples": len(prompts),
        "total_tokens": total_tokens,
    }


def evaluate_gsm8k(model, tokenizer, questions, max_new_tokens, warmup_runs, device):
    """Run GSM8K evaluation: speed metrics + accuracy in a single pass.

    Mirrors the timing structure of benchmark() but formats GSM8K prompts,
    decodes generated text, and scores accuracy via parse_gsm8k_answer().

    Returns the standard 11-key speed dict plus:
        gsm8k_accuracy: float (0.0 to 1.0)
        gsm8k_correct: int
        gsm8k_total: int
    """
    raw_prompts = [GSM8K_INSTRUCTION + q["question"] for q in questions]
    prompts = format_prompts(tokenizer, raw_prompts)

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
    sync_device(device)

    # Measure
    reset_peak_memory(device)

    all_ttft = []
    all_latency = []
    all_tpot = []
    all_tokens = []
    tpot_skipped = 0
    correct = 0

    for idx, (prompt, item) in enumerate(zip(prompts, questions)):
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        input_len = inputs["input_ids"].shape[1]

        streamer = TTFTStreamer()
        sync_device(device)

        streamer.start_time = time.perf_counter()
        t_start = streamer.start_time

        with torch.no_grad():
            output_ids = model.generate(
                **inputs, generation_config=gen_config, streamer=streamer,
            )

        sync_device(device)
        t_end = time.perf_counter()

        # Decode only the new tokens for accuracy scoring
        new_token_ids = output_ids[0, input_len:]
        model_output = tokenizer.decode(new_token_ids, skip_special_tokens=True)

        predicted = parse_gsm8k_answer(model_output)
        if predicted is not None and abs(predicted - item["answer"]) < 1e-3:
            correct += 1

        # Timing (identical to benchmark())
        new_tokens = output_ids.shape[1] - input_len
        latency = t_end - t_start
        ttft = (streamer.first_token_time - t_start) if streamer.first_token_time else latency

        all_ttft.append(ttft * 1000)
        all_latency.append(latency * 1000)
        all_tokens.append(new_tokens)

        if new_tokens < 2:
            tpot_skipped += 1
            log.warning(
                "skipped TPOT for prompt index %d (only %d token(s) generated)",
                idx, new_tokens,
            )
            continue

        decode_tokens = new_tokens - 1
        decode_time = latency - ttft
        all_tpot.append((decode_time / decode_tokens) * 1000)

    total_tokens = sum(all_tokens)
    total_time = sum(all_latency) / 1000

    return {
        "throughput_tok_s": total_tokens / total_time,
        "ttft_mean_ms": mean(all_ttft),
        "ttft_std_ms": stdev(all_ttft) if len(all_ttft) > 1 else 0,
        "latency_mean_ms": mean(all_latency),
        "latency_std_ms": stdev(all_latency) if len(all_latency) > 1 else 0,
        "tpot_mean_ms": mean(all_tpot) if all_tpot else 0,
        "tpot_std_ms": stdev(all_tpot) if len(all_tpot) > 1 else 0,
        "tpot_skipped": tpot_skipped,
        "peak_mem_mb": get_peak_memory_mb(device),
        "num_samples": len(questions),
        "total_tokens": total_tokens,
        "gsm8k_accuracy": correct / len(questions),
        "gsm8k_correct": correct,
        "gsm8k_total": len(questions),
    }


def print_results(label, r):
    lines = [
        "",
        "=" * 60,
        f"  {label}",
        "=" * 60,
        f"  Throughput          : {r['throughput_tok_s']:>8.2f} tokens/s",
        f"  TTFT (prefill)      : {r['ttft_mean_ms']:>8.2f} ms  (std {r['ttft_std_ms']:.2f})",
        f"  TPOT (decode)       : {r['tpot_mean_ms']:>8.2f} ms  (std {r['tpot_std_ms']:.2f})",
        f"  Latency (TTFT+TPOT) : {r['latency_mean_ms']:>8.2f} ms  (std {r['latency_std_ms']:.2f})",
        f"  Peak memory         : {r['peak_mem_mb']:>8.1f} MB",
        f"  Samples / tokens    : {r['num_samples']} / {r['total_tokens']}",
    ]
    if "perplexity" in r:
        lines.append(f"  Perplexity          : {r['perplexity']:>8.2f}")
    if "gsm8k_accuracy" in r:
        lines.append(
            f"  GSM8K accuracy      : {r['gsm8k_accuracy']:>7.1%}  "
            f"({r['gsm8k_correct']}/{r['gsm8k_total']})"
        )
    if r.get("tpot_skipped", 0) > 0:
        lines.append(f"  TPOT skipped        : {r['tpot_skipped']} prompt(s) (<2 tokens generated)")
    log.info("\n".join(lines))


def print_comparison(label_a, results_a, label_b, results_b):
    def delta(b, a):
        if a == 0:
            return "N/A"
        pct = (b - a) / abs(a) * 100
        sign = "+" if pct >= 0 else ""
        return f"{sign}{pct:.1f}%"

    speedup = results_b["throughput_tok_s"] / results_a["throughput_tok_s"] if results_a["throughput_tok_s"] else 0
    mem_saved = (1 - results_b["peak_mem_mb"] / results_a["peak_mem_mb"]) * 100 if results_a["peak_mem_mb"] else 0
    lines = [
        "",
        "=" * 60,
        f"  COMPARISON  ({label_a} vs {label_b})",
        "=" * 60,
        f"  {'Metric':<22} {label_a:>12} {label_b:>12} {'Delta':>10}",
        f"  {'-' * 56}",
        f"  {'Throughput (tok/s)':<22} {results_a['throughput_tok_s']:>12.2f} {results_b['throughput_tok_s']:>12.2f} {delta(results_b['throughput_tok_s'], results_a['throughput_tok_s']):>10}",
        f"  {'TTFT (ms)':<22} {results_a['ttft_mean_ms']:>12.2f} {results_b['ttft_mean_ms']:>12.2f} {delta(results_b['ttft_mean_ms'], results_a['ttft_mean_ms']):>10}",
        f"  {'TPOT (ms)':<22} {results_a['tpot_mean_ms']:>12.2f} {results_b['tpot_mean_ms']:>12.2f} {delta(results_b['tpot_mean_ms'], results_a['tpot_mean_ms']):>10}",
        f"  {'Latency (ms)':<22} {results_a['latency_mean_ms']:>12.2f} {results_b['latency_mean_ms']:>12.2f} {delta(results_b['latency_mean_ms'], results_a['latency_mean_ms']):>10}",
        f"  {'Peak memory (MB)':<22} {results_a['peak_mem_mb']:>12.1f} {results_b['peak_mem_mb']:>12.1f} {delta(results_b['peak_mem_mb'], results_a['peak_mem_mb']):>10}",
    ]
    if "perplexity" in results_a and "perplexity" in results_b:
        lines.append(
            f"  {'Perplexity':<22} {results_a['perplexity']:>12.2f} {results_b['perplexity']:>12.2f} {delta(results_b['perplexity'], results_a['perplexity']):>10}"
        )
    if "gsm8k_accuracy" in results_a and "gsm8k_accuracy" in results_b:
        lines.append(
            f"  {'GSM8K accuracy':<22} {results_a['gsm8k_accuracy']:>11.1%} {results_b['gsm8k_accuracy']:>11.1%} {delta(results_b['gsm8k_accuracy'], results_a['gsm8k_accuracy']):>10}"
        )
    lines += [
        f"  {'-' * 56}",
        f"  Speedup: {speedup:.2f}x  |  Memory saved: {mem_saved:.1f}%",
        "=" * 60,
    ]
    log.info("\n".join(lines))


def benchmark_mlx_model(model, tokenizer, prompts, max_new_tokens, warmup_runs):
    """Benchmark an MLX model loaded via mlx_lm.load().

    Uses mlx_lm.stream_generate instead of HuggingFace .generate(), so this
    function is the MLX counterpart of benchmark(). Returns the same 11-key
    dict so print_results(), print_comparison(), and dump_results() work
    without modification.

    Requires mlx and mlx-lm to be installed (Apple Silicon only). The imports
    are deferred so bench_utils remains importable on CUDA-only machines.
    """
    try:
        import mlx.core as mx
        from mlx_lm import stream_generate
    except ImportError as exc:
        raise ImportError(
            "mlx and mlx-lm are required for benchmark_mlx_model(). "
            "Install with: pip install mlx-lm"
        ) from exc

    # ── Warmup ────────────────────────────────────────────────────────────────
    for i in range(min(warmup_runs, len(prompts))):
        for _ in stream_generate(model, tokenizer, prompts[i], max_tokens=max_new_tokens):
            pass
    mx.eval()  # flush any deferred Metal compute from warmup

    # ── Measure ───────────────────────────────────────────────────────────────
    mx.reset_peak_memory()

    all_ttft = []
    all_latency = []
    all_tpot = []
    all_tokens = []
    tpot_skipped = 0

    for idx, prompt in enumerate(prompts):
        t_start = time.perf_counter()
        first_token = None
        yield_count = 0
        final_gen_tokens = 0

        for response in stream_generate(model, tokenizer, prompt, max_tokens=max_new_tokens):
            if first_token is None:
                first_token = time.perf_counter()
            yield_count += 1
            # generation_tokens is the running output-token count in mlx-lm;
            # fall back to counting yields in case the attribute name drifts
            # across mlx-lm versions.
            final_gen_tokens = getattr(response, "generation_tokens", yield_count)

        mx.eval()  # flush deferred Metal work before recording end time
        t_end = time.perf_counter()

        new_tokens = final_gen_tokens
        latency_ms = (t_end - t_start) * 1000
        ttft_ms = ((first_token - t_start) * 1000) if first_token is not None else latency_ms

        all_ttft.append(ttft_ms)
        all_latency.append(latency_ms)
        all_tokens.append(new_tokens)

        if new_tokens < 2:
            tpot_skipped += 1
            log.warning(
                "skipped TPOT for prompt index %d (only %d token(s) generated)",
                idx, new_tokens,
            )
            continue

        decode_tokens = new_tokens - 1
        decode_time_ms = latency_ms - ttft_ms
        all_tpot.append(decode_time_ms / decode_tokens)

    total_tokens = sum(all_tokens)
    total_time_s = sum(all_latency) / 1000

    return {
        "throughput_tok_s": total_tokens / total_time_s,
        "ttft_mean_ms": mean(all_ttft),
        "ttft_std_ms": stdev(all_ttft) if len(all_ttft) > 1 else 0,
        "latency_mean_ms": mean(all_latency),
        "latency_std_ms": stdev(all_latency) if len(all_latency) > 1 else 0,
        "tpot_mean_ms": mean(all_tpot) if all_tpot else 0,
        "tpot_std_ms": stdev(all_tpot) if len(all_tpot) > 1 else 0,
        "tpot_skipped": tpot_skipped,
        "peak_mem_mb": mx.get_peak_memory() / 1024 / 1024,
        "num_samples": len(prompts),
        "total_tokens": total_tokens,
    }


def evaluate_gsm8k_mlx(model, tokenizer, questions, max_new_tokens, warmup_runs):
    """MLX counterpart of evaluate_gsm8k(). Returns same dict format.

    Uses mlx_lm.stream_generate for timing and collects output text
    for accuracy scoring.
    """
    try:
        import mlx.core as mx
        from mlx_lm import stream_generate
    except ImportError as exc:
        raise ImportError(
            "mlx and mlx-lm are required for evaluate_gsm8k_mlx(). "
            "Install with: pip install mlx-lm"
        ) from exc

    raw_prompts = [GSM8K_INSTRUCTION + q["question"] for q in questions]
    prompts = format_prompts(tokenizer, raw_prompts)

    # Warmup
    for i in range(min(warmup_runs, len(prompts))):
        for _ in stream_generate(model, tokenizer, prompts[i], max_tokens=max_new_tokens):
            pass
    mx.eval()

    # Measure
    mx.reset_peak_memory()

    all_ttft = []
    all_latency = []
    all_tpot = []
    all_tokens = []
    tpot_skipped = 0
    correct = 0

    for idx, (prompt, item) in enumerate(zip(prompts, questions)):
        t_start = time.perf_counter()
        first_token = None
        yield_count = 0
        final_gen_tokens = 0
        output_chunks = []

        for response in stream_generate(model, tokenizer, prompt, max_tokens=max_new_tokens):
            if first_token is None:
                first_token = time.perf_counter()
            yield_count += 1
            final_gen_tokens = getattr(response, "generation_tokens", yield_count)
            # Collect text — response.text may be cumulative or per-token
            # depending on mlx-lm version, so we take the last value.
            output_chunks.append(getattr(response, "text", ""))

        mx.eval()
        t_end = time.perf_counter()

        # Use the last chunk as the full output (cumulative in current mlx-lm)
        model_output = output_chunks[-1] if output_chunks else ""

        predicted = parse_gsm8k_answer(model_output)
        if predicted is not None and abs(predicted - item["answer"]) < 1e-3:
            correct += 1

        new_tokens = final_gen_tokens
        latency_ms = (t_end - t_start) * 1000
        ttft_ms = ((first_token - t_start) * 1000) if first_token is not None else latency_ms

        all_ttft.append(ttft_ms)
        all_latency.append(latency_ms)
        all_tokens.append(new_tokens)

        if new_tokens < 2:
            tpot_skipped += 1
            log.warning(
                "skipped TPOT for prompt index %d (only %d token(s) generated)",
                idx, new_tokens,
            )
            continue

        decode_time_ms = latency_ms - ttft_ms
        all_tpot.append(decode_time_ms / (new_tokens - 1))

    total_tokens = sum(all_tokens)
    total_time_s = sum(all_latency) / 1000

    return {
        "throughput_tok_s": total_tokens / total_time_s,
        "ttft_mean_ms": mean(all_ttft),
        "ttft_std_ms": stdev(all_ttft) if len(all_ttft) > 1 else 0,
        "latency_mean_ms": mean(all_latency),
        "latency_std_ms": stdev(all_latency) if len(all_latency) > 1 else 0,
        "tpot_mean_ms": mean(all_tpot) if all_tpot else 0,
        "tpot_std_ms": stdev(all_tpot) if len(all_tpot) > 1 else 0,
        "tpot_skipped": tpot_skipped,
        "peak_mem_mb": mx.get_peak_memory() / 1024 / 1024,
        "num_samples": len(questions),
        "total_tokens": total_tokens,
        "gsm8k_accuracy": correct / len(questions),
        "gsm8k_correct": correct,
        "gsm8k_total": len(questions),
    }
