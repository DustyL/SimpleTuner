# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SimpleTuner is a comprehensive diffusion model training framework supporting image, video, and audio generative models. It provides LoRA, LyCORIS, and full-rank training for models including Flux.1/2, SDXL, SD3, and many others.

## Development Commands

```bash
# Activate venv (required before any commands)
source /root/SimpleTuner/venv/bin/activate

# Run training with a config directory
simpletuner train env=config/your-config-dir

# Run training with direct config file
simpletuner train --config_path=/path/to/config

# Run tests (unittest, not pytest)
python -m unittest -v -f

# Run a single test
python -m unittest tests.test_module.TestClass.test_method -v

# List available example configurations
simpletuner examples list

# Copy an example configuration
simpletuner examples copy flux-lora ./my-config
```

## Architecture Overview

### Entry Points
- `simpletuner/__main__.py` - Module entry (`python -m simpletuner`)
- `simpletuner/cli.py` - CLI commands (`simpletuner train`, `simpletuner examples`)
- `simpletuner/train.py` - Main training script with Trainer initialization

### Core Training Flow (train.py)
1. `trainer.init_preprocessing_models()` - Load VAE, text encoders
2. `trainer.init_data_backend()` - Configure datasets with BatchFetcher
3. `trainer.init_load_base_model()` - Load transformer/UNet
4. `trainer.init_trainable_peft_adapter()` - Initialize LoRA/LyCORIS
5. `trainer.resume_and_prepare()` - Accelerate setup + checkpointing
6. `trainer.train()` - Main training loop

### Key Directories
```
simpletuner/
├── helpers/
│   ├── models/           # Model implementations (flux/, flux2/, sdxl/, etc.)
│   ├── training/         # Trainer, optimizer, attention backends
│   ├── data_backend/     # Dataset loading, aspect bucketing, caching
│   ├── configuration/    # Config loading (cmd_args.py, loader.py)
│   └── utils/            # RamTorch, offloading, precision utilities
├── simpletuner_sdk/      # Web UI server and field registry
└── examples/             # Example configurations
config/                   # User training configurations
documentation/            # All documentation files
tests/                    # Unit tests
```

### Model-Specific Files (Flux2 Example)
- `helpers/models/flux2/model.py` - Core Flux2 class, latent packing, Mistral encoding
- `helpers/models/flux2/transformer.py` - Flux2Transformer2DModel architecture
- `helpers/models/flux2/autoencoder.py` - Custom VAE with batch normalization
- `helpers/models/flux2/pipeline.py` - Inference pipeline

### Configuration System
SimpleTuner supports three config backends:
1. **JSON** (`config.json`) - Preferred, supports nested structures
2. **TOML** (`config.toml`) - Alternative structured format
3. **ENV** (`config.env`) - Legacy shell variable format

Config loading hierarchy: CLI args > config file > defaults

Key config files in a training directory:
- `config.json` - Main training parameters
- `multidatabackend.json` - Dataset definitions
- `lycoris_config.json` - LyCORIS algorithm settings (if using LyCORIS)
- `user_prompt_library.json` - Validation prompts

## Current Server Environment

**Hardware:**
- GPU: NVIDIA B300 SXM6 AC (275GB VRAM)
- CPU: AMD EPYC 9575F 64-Core (30 vCPUs available)
- RAM: 270GB system memory
- CUDA: 13.0, Driver: 580.105.08

**Software:**
- Python: 3.12.12
- PyTorch: 2.9.1+cu130
- SimpleTuner: 3.2.0 (from venv)
- Venv: `/root/SimpleTuner/venv`

**Local Model Path:**
- Flux.2-dev: `/root/FLUX.2-dev` (HF repo with LFS pointers, weights downloaded on-demand)

---

## Flux.2-dev Model Specifications

### CRITICAL: Actual Model Sizes (from index.json metadata)

| Component | Parameters | Storage (bf16) | Notes |
|-----------|------------|----------------|-------|
| **Transformer** | ~32B | **60.0 GB** | 8 DoubleStreamBlocks + 48 SingleStreamBlocks |
| **Text Encoder (Mistral-3)** | 24.01B | **44.7 GB** | Mistral3ForConditionalGeneration |
| **VAE** | ~150M | ~0.3 GB | AutoencoderKLFlux2 |
| **TOTAL** | **~56B** | **~105 GB** | Just weights, no optimizer/gradients |

### Transformer Architecture
- Hidden dimension: 6,144 (48 attention heads × 128 head dim)
- Joint attention dimension: 15,360 (stacked Mistral layers 10, 20, 30)
- MLP ratio: 3.0
- Latent channels: 128 (after 2×2 pixel shuffle from 32-channel VAE output)

### Text Encoder (Mistral-Small-3.1-24B)
- Hidden size: 5,120
- 40 transformer layers
- 32 attention heads, 8 KV heads
- Context: 131,072 tokens (but SimpleTuner caps at 512)
- Includes Pixtral vision encoder (24 layers, 1024 hidden)

---

## VRAM Requirements for Training (CRITICAL)

### Full bf16 Training (No Quantization) - IMPOSSIBLE ON SINGLE GPU

```
Model Weights (bf16):           ~105 GB
Optimizer States (AdamW):       ~210 GB  (2× weights for momentum + variance)
Gradients:                      ~105 GB
Activations (batch=1):          ~50-100 GB
Framework Overhead:             ~10 GB
─────────────────────────────────────────
TOTAL:                          ~480-530 GB  ❌ EXCEEDS 275GB
```

**Full-precision training requires distributed setup (FSDP2/DeepSpeed) across multiple GPUs.**

### LoRA/LyCORIS Training - More Feasible

For LoRA training, only adapter weights need optimizer states:

```
Base Model Weights (frozen):    ~105 GB  (no gradients needed)
LoRA Adapter Weights:           ~1-2 GB  (depends on rank)
LoRA Optimizer States:          ~4-8 GB  (2× adapter weights)
LoRA Gradients:                 ~1-2 GB
Activations (batch=1):          ~50-100 GB  (still need full forward pass)
Framework Overhead:             ~10 GB
─────────────────────────────────────────
TOTAL:                          ~170-230 GB  ✓ FITS (tight)
```

### Realistic Configurations for B300 (275GB)

#### Option 1: LoRA with int8 Text Encoder (Recommended)
```json
{
    "model_type": "lora",
    "base_model_precision": "no_change",
    "text_encoder_1_precision": "int8-quanto",
    "mixed_precision": "bf16",
    "gradient_checkpointing": true,
    "train_batch_size": 1,
    "gradient_accumulation_steps": 4,
    "attention_mechanism": "flash-attn-3"
}
```
**Estimated VRAM: ~180-200 GB** - Safe margin for batch_size=1

#### Option 2: LyCORIS with Aggressive Quantization
```json
{
    "model_type": "lora",
    "lora_type": "lycoris",
    "base_model_precision": "int8-quanto",
    "text_encoder_1_precision": "int8-quanto",
    "mixed_precision": "bf16",
    "gradient_checkpointing": true,
    "train_batch_size": 2,
    "gradient_accumulation_steps": 2
}
```
**Estimated VRAM: ~150-180 GB** - May allow batch_size=2

#### Option 3: Maximum Throughput (Gradient Checkpointing OFF)
```json
{
    "model_type": "lora",
    "base_model_precision": "int8-quanto",
    "text_encoder_1_precision": "int8-quanto",
    "gradient_checkpointing": false,
    "train_batch_size": 1,
    "gradient_accumulation_steps": 4
}
```
**Estimated VRAM: ~220-250 GB** - Tight, but faster per-step

### What batch_size=32+ Requires

Your existing configs show batch_size=34-36. This is only possible with:
1. **Heavy quantization** (int8-quanto on both transformer AND text encoder)
2. **Gradient checkpointing ON**
3. **LyCORIS** (smaller trainable parameter footprint than full LoRA)
4. **Possibly group offloading** to stream layers from CPU

---

## VRAM Optimization Techniques

### Quantization Options (Largest Impact)
| Precision | VRAM Savings | Quality Impact |
|-----------|-------------|----------------|
| `int8-quanto` | 50% of component | Minimal |
| `int4-quanto` | 75% of component | Some quality loss |
| `fp8-torchao` | 50% (H100+ only) | Minimal |

### Other Techniques
| Technique | VRAM Savings | Speed Impact |
|-----------|-------------|--------------|
| Gradient checkpointing | 40-60% activations | 15-25% slower |
| Group offloading | 20-40% | 10-20% slower |
| RamTorch | 30-50% | 20-40% slower |
| FlashAttention-3 | 10-30% | 5-20% faster |
| EMA on CPU | 100% of EMA copy | Minimal |

### Key Settings for B300
```json
{
    "attention_mechanism": "flash-attn-3",
    "fuse_qkv_projections": true,
    "keep_vae_loaded": true,
    "dataloader_prefetch": true,
    "quantize_via": "cpu"
}
```

---

## Flux2 Training Specifics

### LoRA Targets
```python
DEFAULT_LORA_TARGET = [
    "attn.to_q", "attn.to_k", "attn.to_v", "attn.to_out.0",  # Double stream
    "attn.to_qkv_mlp_proj",  # Single stream parallel attention+FF
]
```
Presets via `--flux_lora_target`: `all`, `attention`, `mlp`, `tiny`

### LyCORIS Configuration
```json
{
    "algo": "lokr",
    "multiplier": 1.0,
    "linear_dim": 128,
    "linear_alpha": 128,
    "factor": 16,
    "apply_preset": {
        "target_module": ["Flux2TransformerBlock", "Flux2SingleTransformerBlock"]
    }
}
```

### Flux2-Specific Parameters
- `flux_guidance_mode`: `constant` or `random-range`
- `flux_guidance_value`: Fixed at 1.0 for training (differs from inference 3.0-5.0)
- `flow_schedule_shift`: Flow matching schedule adjustment
- `aspect_bucket_alignment`: Must be 16 for Flux2

---

## Existing Training Configurations

### `/root/SimpleTuner/config/flux2-dlay-b300-fast`
Fast LyCORIS (LoKr) training:
- batch_size: 36, learning_rate: 1e-4, max_train_steps: 500
- **Requires**: gradient_checkpointing=true + quantization for batch_size=36
- LyCORIS LoKr with factor=16, dim=128
- Masked loss training enabled (probability: 1.0)

### `/root/SimpleTuner/config/flux2-dlay-b300`
Standard LyCORIS training:
- batch_size: 34, learning_rate: 5e-5, max_train_steps: 4000
- Dataset repeats: 30

### `/root/SimpleTuner/config/flux2-dlay-lycoris`
FSDP2-enabled training (for distributed):
- fsdp_enable: true, fsdp_version: 2
- batch_size: 2, gradient_accumulation: 4
- EMA enabled with decay 0.9999

---

## Dataset Setup for Masked Loss Training

```json
[
    {
        "id": "subject-data",
        "type": "local",
        "dataset_type": "image",
        "conditioning_data": "mask-data",
        "instance_data_dir": "/root/DATASETS/DLAY/flux2",
        "caption_strategy": "textfile"
    },
    {
        "id": "mask-data",
        "type": "local",
        "dataset_type": "conditioning",
        "conditioning_type": "mask",
        "instance_data_dir": "/root/DATASETS/DLAY/mask"
    },
    {
        "id": "text-embeds",
        "type": "local",
        "dataset_type": "text_embeds",
        "default": true,
        "cache_dir": "/root/SimpleTuner/cache/text/flux2/dlay"
    }
]
```

Current dataset: 269 images with captions and B&W masks at `/root/DATASETS/DLAY/flux2` and `/root/DATASETS/DLAY/mask`

---

## Code Style (from AGENTS.md)

- Defensive programming must be justified and expected by users
- Never hide import failures unless logic requires it
- Use `# type: ignore` only when absolutely necessary
- No rambling comments - be concise or leave no comment
- Let code be self-explanatory
- Do not remove untracked files unless instructed
- Problems should be provable through tests or logging
- Use available accelerator (cuda, mps) opportunistically

## Testing

```bash
# Full test suite (~300 seconds)
python -m unittest -v -f

# Single test
python -m unittest tests.test_ramtorch.TestRamTorch -v

# With debug logging
SIMPLETUNER_LOG_LEVEL=DEBUG python -m unittest -v -f
```

## Multi-GPU Training Options

SimpleTuner supports several multi-GPU strategies with different tradeoffs:

### 1. Standard DDP (DistributedDataParallel)

**Best for:** LoRA/LyCORIS training when each GPU can hold the full model

```bash
# Set in config.env or config.json
TRAINING_NUM_PROCESSES=4  # Number of GPUs
# OR
"num_processes": 4
```

- Each GPU holds complete model copy
- Gradients synchronized across GPUs
- Most compatible with all training modes

### 2. FSDP2 (Fully Sharded Data Parallel v2)

**Documentation says:** "FSDP2 can only be enabled when `model_type` is `full`"

**However, code supports FSDP2 + LoRA/LyCORIS!** (See `save_hooks.py:564-571`)

The base model can be sharded across GPUs while LoRA adapters are saved/loaded normally. This reduces VRAM by distributing frozen weights.

```json
{
    "model_type": "lora",
    "lora_type": "lycoris",
    "fsdp_enable": true,
    "fsdp_version": 2,
    "fsdp_reshard_after_forward": true,
    "fsdp_state_dict_type": "SHARDED_STATE_DICT",
    "fsdp_auto_wrap_policy": "TRANSFORMER_BASED_WRAP"
}
```

**Key FSDP2 Settings:**
| Setting | Description | Recommended |
|---------|-------------|-------------|
| `fsdp_reshard_after_forward` | Release shards after forward pass | `true` |
| `fsdp_cpu_offload` | Offload parameters to CPU | `false` (unless needed) |
| `fsdp_activation_checkpointing` | FSDP-level checkpointing | Replaces `gradient_checkpointing` |
| `fsdp_cpu_ram_efficient_loading` | Reduce host memory spikes | `true` for resume |

**For 4xB200/B300 with Flux2 + LyCORIS:**
```json
{
    "model_type": "lora",
    "lora_type": "lycoris",
    "fsdp_enable": true,
    "fsdp_version": 2,
    "base_model_precision": "bf16",
    "text_encoder_1_precision": "bf16",
    "train_batch_size": 4,
    "gradient_accumulation_steps": 2,
    "num_processes": 4
}
```
Each GPU: ~60-80GB, Effective batch: 32

### 3. DeepSpeed ZeRO

**IMPORTANT: DeepSpeed does NOT support LoRA/LyCORIS** (see `DEEPSPEED.md:41-45`)

Only use DeepSpeed for full fine-tuning:
- ZeRO Stage 1-2: Optimizer state sharding
- ZeRO Stage 3: Full parameter sharding + offload to CPU/NVMe

```json
{
    "model_type": "full",
    "deepspeed_config": {
        "zero_optimization": {
            "stage": 2,
            "offload_optimizer": {"device": "cpu"}
        }
    }
}
```

### Multi-GPU Summary for Flux2

| Strategy | LoRA/LyCORIS | Full Fine-tune | VRAM per GPU | Notes |
|----------|--------------|----------------|--------------|-------|
| DDP | Yes | No (too large) | Full model | Simple, recommended for LoRA |
| FSDP2 | Yes* | Yes | Sharded | *Code supports it despite docs |
| DeepSpeed | No | Yes | Sharded + offload | Most memory efficient for full |

### Dataset Sizing for Multi-GPU

```
effective_batch_size = train_batch_size x num_gpus x gradient_accumulation_steps
```

Each aspect bucket must have at least `effective_batch_size` samples (or use `--allow_dataset_oversubscription`).

With 269 images and 4 GPUs:
- batch=2, grad_accum=2, gpus=4 -> effective=16 samples (OK)
- batch=4, grad_accum=4, gpus=4 -> effective=64 samples (may need more repeats)

---

## Key Documentation Files

- `/root/SimpleTuner/documentation/OPTIONS.md` - All configuration options
- `/root/SimpleTuner/documentation/quickstart/FLUX2.md` - Flux2-specific guide
- `/root/SimpleTuner/documentation/LYCORIS.md` - LyCORIS training guide
- `/root/SimpleTuner/documentation/DREAMBOOTH.md` - Dreambooth/persona training
- `/root/SimpleTuner/documentation/FSDP2.md` - Distributed training with FSDP2
- `/root/SimpleTuner/documentation/DEEPSPEED.md` - DeepSpeed integration
- `/root/SimpleTuner/documentation/DISTRIBUTED.md` - Multi-node cluster setup

## Local Modifications (Uncommitted)

The following customizations exist in the working tree:
1. **factory.py**: Dataloader worker parallelization (`dataloader_num_workers`, `prefetch_factor`)
2. **flux2/model.py**: RamTorch integration before device placement
3. **data.py**: SDK fields for dataloader worker settings
4. **flux2/transformer.py**: Added `__len__` method for PyTorch 2.11+ torch.compile compatibility

These enhance throughput on systems with fast CPUs and may be worth committing upstream.

---

## Known Issues & Fixes

### PyTorch 2.11+ torch.compile Validation Error

**Issue**: Validation fails with `Flux2Transformer2DModel does not support len()` when using torch.compile with PyTorch 2.11.0a0 or later.

**Cause**: PyTorch 2.11 changed `OptimizedModule` to delegate `len()` calls to the wrapped model.

**Fix**: Added `__len__` method to `Flux2Transformer2DModel` in `simpletuner/helpers/models/flux2/transformer.py`.

**Full Documentation**: `/root/SimpleTuner/documentation/PYTORCH_211_TORCH_COMPILE_LEN_FIX.md`
