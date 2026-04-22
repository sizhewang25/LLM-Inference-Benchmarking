# 2-bit Quantization: MLX Affine vs llama.cpp Q2_K

## Summary

MLX's built-in `mlx_lm convert` 2-bit quantization and llama.cpp's Q2_K format are **not equivalent** and should not be compared directly. The differences are algorithmic, not just a parameter mismatch.

---

## Algorithm Comparison

| Property | MLX affine gs64 | llama.cpp Q2_K |
|---|---|---|
| Group / block size | 64 weights per group | 256-weight super-block, 16-weight sub-blocks (QK_K=256) |
| Scale levels | 1 (one scale + one zero per group) | 2 (sub-block scale/min + super-block scale/min) |
| Scale storage | fp16 per group | 4-bit per sub-block, fp16 super-scale/min |
| Quantization algorithm | Round-to-nearest affine | K-quant: exhaustive min-search per sub-block |
| Calibration data | None | None (but min-search compensates) |
| Effective bits/weight | ~2.5 | ~2.625 |
| Practical quality | Unusable (PPL ~1300+) | Usable (PPL ~6–10 for 7–8B models) |

### llama.cpp QK_K constant

```c
// ggml/src/ggml-quants.c
#ifdef GGML_QKK_64
#define QK_K 64     // only if explicitly compiled with this flag
#else
#define QK_K 256    // default
#endif
```

The default `QK_K=256` is almost universal. The 64-element variant requires a non-default compile flag.

---

## Why Round-to-Nearest Breaks at 2-bit

At *n*-bit affine quantization the maximum per-weight rounding error is `½ × (group_range / 2ⁿ)`:

| Bits | Discrete levels | Max rounding error (% of group range) | Practical impact |
|---|---|---|---|
| 8 | 256 | 0.2% | Negligible — PPL delta vs FP16 < 0.01 |
| 4 | 16 | 3.1% | Small — PPL within ~0.1–0.5 of FP16 |
| **2** | **4** | **12.5%** | **Severe — errors compound across layers** |

At 2 bits the quantization grid is so coarse that many weights land far from any grid point. Errors accumulate across attention and MLP layers and the model loses coherent output. This is why calibrated methods (GPTQ, AWQ) or algorithm-level compensation (K-quants' min-search) are required for usable 2-bit models.

---

## Observed Results (MLX affine gs64, Llama-3.1-8B-Instruct)

From `bench_llama31_mlx.py` run on Apple M2 Max, 2026-04-22:

| Metric | Value |
|---|---|
| Weight memory | 2.34 GiB |
| Peak memory | 3.16 GiB |
| Prefill (PP512) | ~280 t/s |
| Decode (TG128) | ~65 t/s |
| Wikitext-2 PPL | **1345.75 ± 14.02** |

For reference, llama.cpp Q2_K on the same model typically yields PPL ~6–8.

---

## Implications for This Benchmark

- **Q8 and Q4** results from `bench_llama31_mlx.py` are valid and comparable to equivalent llama.cpp quantization levels (with the caveat that group sizes and scale algorithms still differ slightly).
- **2-bit MLX affine** results capture speed and memory characteristics accurately but quality metrics (PPL, HellaSwag, Winogrande) are not meaningful for comparison against Q2_K.
- The format label `"affine-gs64"` used in `benchmark_mlx.py` correctly distinguishes these from GGUF-style formats. The `bench_llama31_mlx.py` variant label `"MLX_2bit"` should be interpreted accordingly.

---

## Recommendation

To produce a 2-bit model on MLX that is closer to Q2_K quality:

1. Use a larger group size (`--q-group-size 256`) to match llama.cpp's super-block granularity — but note the algorithm is still different.
2. Use a calibrated quantization tool (e.g. GPTQ via `gptqmodel`, or AWQ) and convert the resulting weights to MLX format — this is the closest achievable equivalent.
3. There is currently no MLX-native implementation of K-quant style min-search quantization.
