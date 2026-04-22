# MLX Quantization Methods

Source: `mlx.core.quantize` API — https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.quantize.html

---

## Function Signature

```python
mlx.core.quantize(w, group_size=None, bits=None, mode='affine', global_scale=None, stream=None)
```

---

## Supported Modes

| Mode | Default group size | Bits | Scale format | Has zero-point bias | Calibration data |
|---|---|---|---|---|---|
| `affine` | 64 | 2–8 | same as input dtype | Yes | No |
| `mxfp4` | 32 | 4 (fixed) | E8M0 (microscaling) | No | No |
| `mxfp8` | 32 | 8 (fixed) | E8M0 (microscaling) | No | No |
| `nvfp4` | 16 | 4 (fixed) | E4M3 (NVIDIA format) | No | No |

All four modes are **data-free** — scales are computed purely from weight tensor values (max absolute value per group), requiring no forward passes or input samples.

---

## Mode Details

### `affine` (default)
Round-to-nearest linear quantization with a scale and zero-point per group. Flexible bit depth (2–8). At 4-bit and 8-bit it produces usable models; at 2-bit the quantization grid is too coarse and quality collapses (see [2bit-mlx-vs-llama-cpp-q2k.md](2bit-mlx-vs-llama-cpp-q2k.md)).

### `mxfp4` / `mxfp8`
[OCP Microscaling (MX)](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf) format backed by Microsoft, AMD, Intel, and NVIDIA. Uses an E8M0 block scale — a floating-point exponent-only value — which is a more principled representation than affine's integer scale. Fixed at 4-bit or 8-bit respectively. Expected to outperform `affine` at the same bit width because the floating-point scale better handles weight distributions with high dynamic range.

### `nvfp4`
NVIDIA's FP4 variant using E4M3 block scales. Less relevant for Apple Silicon; included for completeness.

---

## Calibration: MLX vs GPTQ/AWQ

| Method | Calibration data? | How scales are determined |
|---|---|---|
| MLX affine | No | min/max per group → linear scale + zero |
| MLX mxfp4/mxfp8/nvfp4 | No | max abs per group → floating-point block scale |
| GPTQ | Yes (~128 samples) | Hessian-based weight update minimizing output error |
| AWQ | Yes | Activation magnitude stats to protect salient weights |
| llama.cpp imatrix | Yes | Per-layer importance scores from calibration text |

Calibrated methods (GPTQ, AWQ) produce better quality at low bit widths (2–3 bit) because they adjust weights to compensate for quantization error based on how activations flow through the model. Data-free methods minimize weight reconstruction error in isolation, which is sufficient at 4-bit+ but insufficient at 2-bit.

---

## Self-Quantizing with `mlx_lm convert`

```bash
python -m mlx_lm convert \
  --hf-path <model_id_or_local_path> \
  -q \
  --q-bits 4 \
  --q-group-size 32 \
  --mlx-path ./models/my-model-4bit-gs32
```

The `--q-bits` and `--q-group-size` flags map to `mlx.core.quantize`'s `bits` and `group_size`. The `mode` parameter is not yet exposed as a CLI flag in `mlx_lm convert` as of April 2026 — `mxfp4`/`mxfp8` would need to be invoked programmatically via the Python API.

---

## Practical Recommendations

| Goal | Method |
|---|---|
| Best quality, don't care about size | FP16 / BF16 |
| Best size/quality tradeoff | `affine` 4-bit, group size 32–64 |
| Potentially better 4-bit quality (untested in this repo) | `mxfp4`, group size 32 |
| Usable 2-bit on MLX | Not possible with built-in methods — use GPTQ externally then convert |

For a usable 2-bit model on MLX: quantize with `gptqmodel` or `autoawq`, then convert the resulting weights with `mlx_lm convert --hf-path <gptq_model>`.

---

## Relationship to llama.cpp Formats

MLX affine and llama.cpp K-quants (Q2_K, Q4_K_M, etc.) are not equivalent at the same nominal bit width. See [2bit-mlx-vs-llama-cpp-q2k.md](2bit-mlx-vs-llama-cpp-q2k.md) for a detailed comparison of 2-bit specifically.
