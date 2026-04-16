# Pre-Quantized Model Catalog

Researched 2026-04-16. Focus: GPTQ 4-bit, GGUF Q4_K_M, MLX 4-bit across
Llama, Qwen, Gemma, GPT-2, and MiniMax families.

## Llama 3.1 / 3.2

| Model | GPTQ 4-bit | GGUF Q4_K_M | MLX 4-bit |
|-------|-----------|-------------|-----------|
| Llama 3.2 1B-Instruct | `ModelCloud/Llama-3.2-1B-Instruct-gptqmodel-4bit-vortex-v2.5` | `bartowski/Llama-3.2-1B-Instruct-GGUF` | `mlx-community/Llama-3.2-1B-Instruct-4bit` |
| Llama 3.2 3B-Instruct | `ModelCloud/Llama-3.2-3B-Instruct-gptqmodel-4bit-vortex-v3` | `bartowski/Llama-3.2-3B-Instruct-GGUF` | `mlx-community/Llama-3.2-3B-Instruct-4bit` |
| Llama 3.1 8B-Instruct | `hugging-quants/Meta-Llama-3.1-8B-Instruct-GPTQ-INT4` (24K dl) / `ModelCloud/Meta-Llama-3.1-8B-Instruct-gptq-4bit` | `bartowski/Meta-Llama-3.1-8B-Instruct-GGUF` | `mlx-community/Meta-Llama-3.1-8B-Instruct-4bit` |

## Qwen 2.5

| Model | GPTQ 4-bit | GGUF Q4_K_M | MLX 4-bit |
|-------|-----------|-------------|-----------|
| Qwen2.5-0.5B-Instruct | `Qwen/Qwen2.5-0.5B-Instruct-GPTQ-Int4` (official) | `Qwen/Qwen2.5-0.5B-Instruct-GGUF` (official) | `mlx-community/Qwen2.5-0.5B-Instruct-4bit` |
| Qwen2.5-1.5B-Instruct | `Qwen/Qwen2.5-1.5B-Instruct-GPTQ-Int4` | `bartowski/Qwen2.5-1.5B-Instruct-GGUF` | `mlx-community/Qwen2.5-1.5B-Instruct-4bit` |
| Qwen2.5-3B-Instruct | `Qwen/Qwen2.5-3B-Instruct-GPTQ-Int4` | `bartowski/Qwen2.5-3B-Instruct-GGUF` | `mlx-community/Qwen2.5-3B-Instruct-4bit` |
| Qwen2.5-7B-Instruct | `Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4` (official, 62K dl) | `bartowski/Qwen2.5-7B-Instruct-GGUF` (142K dl) | `mlx-community/Qwen2.5-7B-Instruct-4bit` |

## Gemma 2 / 3

| Model | GPTQ 4-bit | GGUF Q4_K_M | MLX 4-bit |
|-------|-----------|-------------|-----------|
| Gemma 3 1B-it | -- | `ggml-org/gemma-3-1b-it-GGUF` (official) | `mlx-community/gemma-3-1b-it-qat-4bit` (QAT) |
| Gemma 2 2B-it | `shuyuej/gemma-2-2b-it-GPTQ` (community) | `bartowski/gemma-2-2b-it-GGUF` (1.2M dl) | `mlx-community/gemma-2-2b-it-4bit` |
| Gemma 3 4B-it | `ISTA-DASLab/gemma-3-4b-it-GPTQ-4b-128g` * | `bartowski/google_gemma-3-4b-it-GGUF` | `mlx-community/gemma-3-4b-it-qat-4bit` (1M+ dl) |
| Gemma 2 9B-it | `ModelCloud/gemma-2-9b-it-gptq-4bit` | `bartowski/gemma-2-9b-it-GGUF` | `mlx-community/gemma-2-9b-it-4bit` |
| Gemma 3 12B-it | `ISTA-DASLab/gemma-3-12b-it-GPTQ-4b-128g` * | `bartowski/google_gemma-3-12b-it-GGUF` | `mlx-community/gemma-3-12b-it-qat-4bit` |

\* ISTA-DASLab Gemma 3 GPTQ models use `compressed-tensors` format (vLLM-oriented).
May not load directly via GPTQModel without conversion.

## GPT-2

| Model | GPTQ 4-bit | GGUF Q4_K_M | MLX |
|-------|-----------|-------------|-----|
| GPT-2 124M | `mlabonne/gpt2-GPTQ-4bit` (community) | `RichardErkhov/openai-community_-_gpt2-gguf` | `mlx-community/gpt2-base-mlx` (fp32 only) |
| GPT-2 XL 1.5B | -- | `RichardErkhov/openai-community_-_gpt2-xl-gguf` | `MCES10/gpt2-xl-mlx-fp16` (fp16 only) |

GPT-2 is a base model (no instruct fine-tune), so it cannot be used for
GSM8K accuracy evaluation. Useful only for perplexity and raw speed tests.
Quant availability is sparse; consider OPT-125M (`ModelCloud/Opt-125-GPTQ-4bit-10-25-2024`)
as a GPTQ alternative at this scale.

## MiniMax

| Model | GPTQ 4-bit | GGUF Q4_K_M | MLX 4-bit |
|-------|-----------|-------------|-----------|
| M2.7 (~229B MoE) | `ModelCloud/MiniMax-M2-GPTQMODEL-W4A16` | `bartowski/MiniMaxAI_MiniMax-M2.7-GGUF` (4-part split) | `mlx-community/MiniMax-M2.7-4bit` |

MiniMax only has very large MoE models (~229B total params). Even at 4-bit,
M2.7 requires ~60-130GB RAM. No small MiniMax variants exist.

---

## MXFP4 / NVFP4 (Newer Quantization Formats)

### NVFP4 (NVIDIA FP4)

CUDA-only, requires TensorRT-LLM. Published by `nvidia` org on HuggingFace.

- `nvidia/Llama-3.1-8B-Instruct-NVFP4` (114K dl)
- `nvidia/Gemma-4-31B-IT-NVFP4` (1M dl)
- `nvidia/DeepSeek-R1-0528-NVFP4-v2` (669K dl)

### MXFP4 (Microscaling FP4)

Available as MLX-native from `mlx-community`:

- `mlx-community/gemma-4-26b-a4b-mxfp4` (7.6K dl)
- `mlx-community/gemma-4-31b-it-mxfp4` (6.2K dl)
- `mlx-community/MiniMax-M2.7-4bit-mxfp4` (3.2K dl)

AMD publishes GPU-targeted MXFP4 models (not MLX-compatible):

- `amd/DeepSeek-R1-MXFP4` (93K dl)
- `amd/Qwen3.5-397B-A17B-MXFP4` (11K dl)

---

## Quantizer Orgs

| Org | Format | Quality | Notes |
|-----|--------|---------|-------|
| **ModelCloud** | GPTQ (vortex) | High | GPTQModel maintainers. Vortex = optimized GPTQ v2 format. Best GPTQModel compatibility. |
| **Qwen** (official) | GPTQ, GGUF | High | Official quantizations for Qwen 2.5 family. |
| **bartowski** | GGUF | High | Most prolific GGUF quantizer. Consistent quality across all model families. |
| **ggml-org** | GGUF | High | llama.cpp maintainers. Reference GGUFs (Gemma 3 1B). |
| **mlx-community** | MLX 4-bit | High | Official MLX ecosystem hub. All models pre-converted for `mlx_lm.load()`. |
| **ISTA-DASLab** | GPTQ (compressed-tensors) | High | Academic group. Uses compressed-tensors format for vLLM — may need conversion for GPTQModel. |
| **hugging-quants** | GPTQ | Good | Popular Llama 3.1 8B GPTQ (24K dl). |
| **RichardErkhov** | GGUF | Okay | Community quantizer. Only option for GPT-2 GGUF. |

---

## Recommendations

### Best-covered families (all 3 quant methods available)

**Llama 3.1/3.2** and **Qwen 2.5** have official or high-quality quants
across GPTQ, GGUF, and MLX in all sizes.

### Suggested benchmark matrix

| Tier | Models | Purpose |
|------|--------|---------|
| Smoke test | Qwen2.5-0.5B-Instruct, Llama-3.2-1B-Instruct | Fast iteration, <1 min per run |
| Small | Qwen2.5-3B-Instruct, Gemma-3-4B-it | Mid-tier, fits Mac 24GB easily |
| Real bench | Llama-3.1-8B-Instruct, Qwen2.5-7B-Instruct, Gemma-2-9B-it | Primary results, fits Mac 24GB and A100 80GB |

### Models to drop

- **GPT-2**: Base model, no instruct, sparse quants, can't do GSM8K.
- **MiniMax**: Only huge MoE models exist. Requires multi-GPU or 192GB Mac.

### Compatibility warnings

1. Gemma 3 GPTQ (ISTA-DASLab) uses `compressed-tensors` — test GPTQModel loading before committing.
2. NVFP4 models are CUDA/TensorRT-LLM only — not usable with our PyTorch or MLX pipelines.
3. MXFP4 is MLX-only for now — limited to Apple Silicon benchmarks.
