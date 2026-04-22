#!/usr/bin/env python3
"""
Quantize a HuggingFace model to an MLX format and save it locally for benchmarking.

Supported modes:
  mxfp8   -- MX float8  (E4M3, block-scaled)   → --q-mode mxfp8  (gs=32 fixed)
  mxfp4   -- MX float4  (E2M1, block-scaled)   → --q-mode mxfp4  (gs=32 fixed)
  8bit    -- Integer 8-bit affine quantization
  4bit    -- Integer 4-bit affine quantization  (default mlx-community format)
  2bit    -- Integer 2-bit affine quantization

Usage:
  python quantize_mlx.py --model mlx-community/Meta-Llama-3.1-8B-Instruct-bf16 --mode mxfp8
  python quantize_mlx.py --model Qwen/Qwen2.5-7B-Instruct --mode mxfp4 --out-dir ./models/qwen_mxfp4
  python quantize_mlx.py --model meta-llama/Meta-Llama-3.1-8B-Instruct --mode 4bit --group-size 32
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

# ── mode registry ─────────────────────────────────────────────────────────────

# Each entry: (q_bits, q_mode or None)
# mxfp* modes have fixed group sizes in mlx_lm (mxfp8→32, mxfp4→32);
# passing --q-group-size for those modes would override them incorrectly.
_MODES: dict[str, tuple[int, str | None]] = {
    "mxfp8": (8, "mxfp8"),
    "mxfp4": (4, "mxfp4"),
    "8bit":  (8, None),
    "4bit":  (4, None),
    "2bit":  (2, None),
}


# ── helpers ───────────────────────────────────────────────────────────────────

def _default_out_dir(model_id: str, mode: str) -> Path:
    slug = model_id.split("/")[-1]
    return Path("./models") / f"{slug}_{mode}"


def _build_convert_cmd(
    model_id: str,
    out_dir: Path,
    bits: int,
    group_size: int,
    q_mode: str | None,
) -> list[str]:
    cmd = [
        sys.executable, "-m", "mlx_lm", "convert",
        "--hf-path", model_id,
        "-q",
        "--q-bits", str(bits),
        "--mlx-path", str(out_dir),
    ]
    if q_mode is not None:
        # mxfp* modes have fixed group sizes; skip --q-group-size to use their defaults
        cmd += ["--q-mode", q_mode]
    else:
        cmd += ["--q-group-size", str(group_size)]
    return cmd


def _verify(out_dir: Path) -> None:
    print(f"Verifying: loading model from {out_dir} …")
    from mlx_lm import load as mlx_load
    model, tokenizer = mlx_load(str(out_dir))
    import mlx.core as mx
    mx.eval(model.parameters())
    mem_gib = mx.get_active_memory() / 1024 ** 3
    print(f"  OK — weight memory: {mem_gib:.2f} GiB")
    del model, tokenizer
    import gc
    gc.collect()
    mx.clear_cache()


# ── main ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--model", required=True,
        help="HuggingFace model ID or local path to the source (fp16/bf16) model.",
    )
    p.add_argument(
        "--mode", required=True, choices=sorted(_MODES),
        help="Quantization mode.",
    )
    p.add_argument(
        "--out-dir", default=None,
        help="Output directory (default: ./models/<model-slug>_<mode>).",
    )
    p.add_argument(
        "--group-size", type=int, default=64, choices=[32, 64, 128],
        help="Group size for integer affine quantization (ignored for mxfp* modes). "
             "Default: 64.",
    )
    p.add_argument(
        "--force", action="store_true",
        help="Overwrite output directory if it already exists.",
    )
    p.add_argument(
        "--verify", action="store_true",
        help="Load the saved model after quantization to confirm it is valid.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    bits, q_mode = _MODES[args.mode]
    out_dir = Path(args.out_dir) if args.out_dir else _default_out_dir(args.model, args.mode)

    if out_dir.exists():
        if not any(out_dir.iterdir()):
            out_dir.rmdir()  # empty dir from a prior failed run — remove silently
        elif args.force:
            print(f"--force: removing existing {out_dir}")
            shutil.rmtree(out_dir)
        else:
            print(f"Output directory already exists and is non-empty: {out_dir}")
            print("Use --force to overwrite, or choose a different --out-dir.")
            sys.exit(1)

    out_dir.parent.mkdir(parents=True, exist_ok=True)

    print(f"Model  : {args.model}")
    print(f"Mode   : {args.mode}  ({bits}-bit)")
    if q_mode is None:
        print(f"Group  : {args.group_size}")
    print(f"Output : {out_dir}")
    print()

    cmd = _build_convert_cmd(args.model, out_dir, bits, args.group_size, q_mode)
    print("Running:", " ".join(cmd))
    print()
    subprocess.run(cmd, check=True)

    print(f"\nSaved to: {out_dir}")

    if args.verify:
        _verify(out_dir)


if __name__ == "__main__":
    main()
