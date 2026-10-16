### Vision-Language-Action Models for Robot Cable Routing: A Six-Notebook Ablation Study


---

## Table of Contents

1. [Document Description](#1-document-description)
2. [Project Objective](#2-project-objective)
3. [Three Novelties and Contributions](#3-three-novelties-and-contributions)
4. [Dataset: Berkeley Cable Routing](#4-dataset-berkeley-cable-routing)
   - 4.1 Identity, Publisher, and Access
   - 4.2 Task Description
   - 4.3 Raw Dataset Structure
   - 4.4 Download Pipeline
   - 4.5 Structure Exploration (Block 4)
   - 4.6 Dataset Class: `CableRoutingDataset`
   - 4.7 Disk Cache Architecture
   - 4.8 Data Augmentation Pipeline
   - 4.9 Balanced Sampling and Task Index
   - 4.10 DataLoader Configuration
5. [Shared Infrastructure Across All Notebooks](#5-shared-infrastructure)
   - 5.1 Import Strategy and Rationale
   - 5.2 CUDA and GPU Optimisations
   - 5.3 Reproducibility Seeding
   - 5.4 Universal Utilities: `RT1Tokenizer` and `DropPath`
   - 5.5 Shared Model Components (Block 6)
6. [Shared Training Utilities (Block 8)](#6-shared-training-utilities)
   - 6.1 `ImprovedFocalLoss`
   - 6.2 `ImprovedMixup` with CutMix
   - 6.3 `Lion` Optimiser
   - 6.4 `ImprovedScheduler`
   - 6.5 `ModelEMA`
   - 6.6 `PerTaskMetrics`
7. [Training Setup (Block 11) — Shared Pattern](#7-training-setup)
8. [Training Loop (Block 12) — Shared Pattern](#8-training-loop)
9. [Training Plots (Block 13)](#9-training-plots)
10. [Notebook 1 — RT-1 Base: DistilBERT + MaxViT + CrossAttn TokenLearner](#10-notebook-1)
11. [Notebook 2 — RT-2 VLA + CoT: DistilBERT+VQA + MaxViT + ChainOfThought Head](#11-notebook-2)
12. [Notebook 3 — Lightweight Ablation: BoW + ConvNeXt-Small + Conv TokenLearner](#12-notebook-3)
13. [Notebook 4 — CLIP-Powered: CLIPText + EfficientLite-M + Hierarchical TokenLearner](#13-notebook-4)
14. [Notebook 5 — Maximum Capacity: BERT-ft2 + MaxViT-Large + SelfAttn TokenLearner](#14-notebook-5)
15. [Notebook 6 — Mini-VLA: CLIP + MobileViT-Small + PerceiverResampler + Action Chunking](#15-notebook-6)
16. [Shared XAI Pipeline (Blocks 14–21)](#16-shared-xai-pipeline)
    - 16.1 XAI Dependencies (Block 14)
    - 16.2 XAI Setup (Block 15): sklearn Mock Injection
    - 16.3 GradCAM Family (Block 16)
    - 16.4 LIME (Block 17)
    - 16.5 SHAP (Block 18)
    - 16.6 PDP + ICE + Validation Improvement (Block 19)
    - 16.7 Novel XAI Fusions: ShaGCAM and LimeCAM (Block 20)
    - 16.8 XAI Master Dashboard (Block 21)
17. [Cross-Notebook Ablation Comparison](#17-cross-notebook-ablation-comparison)
18. [Hardware Requirements](#18-hardware-requirements)
19. [Lessons Learned and Engineering Challenges](#19-lessons-learned)
20. [Glossary of Terms](#20-glossary-of-terms)

---

## 1. Document Description

This document is a full technical report covering six Jupyter notebooks (`nb1` through `nb6`) that together constitute a controlled ablation study on Vision-Language-Action (VLA) robot transformer architectures. Each notebook is a self-contained, end-to-end pipeline covering dataset acquisition, preprocessing, model definition, training, diagnostic plotting, and a comprehensive Explainable AI (XAI) evaluation. The intended audience is the research group and supervising teachers who need to understand not only what was built, but why every design decision was made — from the choice of each hyperparameter to the motivation for each architectural component.

The six notebooks differ only in their choice of vision encoder, text encoder, and token learning module. Everything else — the dataset, the loss function, the optimiser, the augmentation pipeline, the XAI suite — is identical. This controlled design means the performance and interpretability differences observed across notebooks can be attributed directly to those three varying components.

---

## 2. Project Objective

The central research question of this project is: **which combination of vision encoder, text encoder, and token learner produces the best action prediction performance on a real-robot cable routing task, and how do these combinations differ in interpretability, training efficiency, and VRAM consumption?**

To answer this, the project implements six distinct VLA model variants — ranging from a minimum-resource pure-CNN design (nb3) to a high-capacity dual-transformer design (nb5) — and trains each one from scratch on the `berkeley_cable_routing` dataset under identical conditions. After training, each model is subjected to the same XAI evaluation suite, enabling fair comparison of what each model "looks at" when predicting robot actions.

The project is structured as a step-by-step methodology:

1. A single shared dataset and caching system ensures all models see exactly the same data.
2. A shared training harness (same loss, same optimiser, same augmentation) ensures differences in metrics reflect architecture, not training procedure.
3. A shared XAI suite ensures that interpretability comparisons are methodologically consistent.
4. Configuration variables at the top of each notebook clearly declare the architectural choices and all hyperparameters, making the study reproducible on any single NVIDIA GPU with ≥8 GB VRAM.

---

## 3. Three Novelties and Contributions

### Novelty 1 — A Modular, Reproducible Ablation Framework for VLA Models on Real Robot Data

The primary contribution of this work is not any single model but the framework that surrounds all six models. Most published VLA research presents one architecture at a time, making direct comparison difficult because datasets, training procedures, and evaluation protocols differ. This study builds a plug-and-play system where the vision encoder, text encoder, and token learner can each be swapped independently while everything else remains fixed. This makes it possible to make statements such as "replacing DistilBERT with BoW reduces validation loss by X while cutting training time by Y," because both models were trained under identical conditions on identical data.

The framework covers the full stack: a TFDS-based dataset downloader, a `numpy.memmap`-backed disk cache that preprocesses and stores the entire dataset once so that subsequent epochs incur only fast random-access disk reads, a balanced episodic sampler, a shared augmentation pipeline, the Lion optimiser, the Focal loss, EMA model tracking, and a seven-block XAI evaluation suite. This infrastructure is described in full in the sections below.

### Novelty 2 — RT-2-Style Chain-of-Thought Planning Adapted to a Narrow Manipulation Domain (nb2)

The original RT-2 paper (Brohan et al., 2023) showed that asking a billion-parameter vision-language model to co-predict language tokens alongside action tokens improves generalisation. However, that approach requires a massive pre-trained VLM as backbone, which is inaccessible to most research groups. Notebook 2 introduces a lightweight adaptation: instead of generating free-form language, the model predicts which of five discrete task phases the current step belongs to (`["pick", "route cable", "place", "adjust", "terminate"]`), embeds that prediction into the same latent space as the visual and action tokens, and uses this plan embedding to condition the action prediction head. This is implemented through a `ChainOfThoughtHead` module and costs only a small auxiliary cross-entropy loss (`PLAN_LOSS_WEIGHT=0.3`) on top of the main action prediction loss. Additionally, the text instructions are reformatted using a `VQAFormatter` that wraps them in the template `"Q: what action should the robot take to {task}? A:"`, improving alignment between instruction semantics and the model's internal representations.

### Novelty 3 — ShaGCAM and LimeCAM: Two Novel XAI Attribution Fusion Methods for Robot Action Models

Block 20 in every notebook introduces two fusion-based XAI visualisation methods that combine complementary attribution signals into a single, more informative saliency map.

**ShaGCAM** multiplies the normalised GradCAM map by the normalised absolute SHAP value map pixel-by-pixel. GradCAM identifies spatial regions that are discriminative for the model's output class, while SHAP assigns each pixel a contribution score grounded in cooperative game theory (Shapley values). Their product retains only pixels that are simultaneously gradient-discriminative and Shapley-attributed, filtering out GradCAM artefacts (regions that are merely in the gradient path but not causally important) and SHAP noise (diffuse low-value attributions).

**LimeCAM** takes the geometric mean of the GradCAM map and a soft version of the LIME superpixel mask (`lime_norm + 0.05` to prevent zero-division at sparse mask boundaries). The geometric mean acts as a logical AND: a pixel must score highly under both perturbation-based (LIME) and gradient-based (GradCAM) explanations to appear in the fusion. The `+0.05` regularisation constant is a deliberate design choice to prevent the fusion from being dominated by LIME mask sparsity in frames where few superpixels are selected.

These fusion maps are computed and visualised for all six models, enabling cross-model attribution comparison that would be impossible with any single method alone.

---

## 4. Dataset: Berkeley Cable Routing

### 4.1 Identity, Publisher, and Access

- **Name:** `berkeley_cable_routing`
- **Publisher:** University of California, Berkeley (part of the Open X-Embodiment consortium)
- **Release context:** Published alongside the Open X-Embodiment dataset collection (~2023), which aggregated real robot demonstrations across 22 robot platforms and 527 skills for the purpose of training generalist robot policies.
- **Access method:** Google Cloud Storage at `gs://gresearch/robotics/berkeley_cable_routing`, accessed via TensorFlow Datasets (`tensorflow_datasets`) using `tfds.builder_from_directory()`.
- **Format:** TFDS-serialised TFRecord shards with associated `dataset_info.json` metadata.

### 4.2 Task Description

The task is robotic cable management. A robot arm must pick up a loose cable, thread it through a series of mounting clips or guides fixed to a surface, and release it at the target position. The robot is observed from a single fixed-perspective RGB camera (overhead or wrist-mounted depending on the episode). Each episode is one complete cable routing attempt, decomposed into variable-length sequences of individual time steps. The action space is 7-dimensional continuous (see §4.5), and instructions are short English sentences such as "route the cable through the clips."

### 4.3 Raw Dataset Structure

Each episode is a dictionary with a single key `"steps"`, whose value is a list of step dictionaries. Each step has:

```
step
├── observation
│   ├── <IMG_KEY>    — RGB image tensor, shape (H, W, 3), dtype uint8
│   └── <INST_KEY>   — natural language instruction, dtype string/bytes
└── action
    ├── rotation_delta     — 3D rotational velocity vector
    ├── terminate_episode  — scalar binary termination flag
    ├── world_vector       — 3D translational velocity vector
    └── (other keys discovered at runtime)
```

The exact key names (`IMG_KEY`, `INST_KEY`, `ACT_KEYS`) are discovered dynamically in Block 4 rather than hardcoded, making the codebase robust to minor variations in dataset structure across different OXE datasets.

### 4.4 Download Pipeline (Block 3)

The download process is managed by two functions:

**`check_existing()`** — walks the local directory `berkeley_cable_routing_data/` recursively. If it finds a file named `dataset_info.json` anywhere in the tree, it computes the total size of all files in that directory and prints a confirmation message, then returns `True` to skip re-downloading. The check is by file presence rather than hash, which is a pragmatic choice given that re-hashing gigabytes of TFRecord data would take longer than the download itself on fast connections.

**`download_gcs_tf()`** — uses `tf.io.gfile.listdir()` with a recursive helper `list_all()` to enumerate every file in the GCS bucket. For each file, it reconstructs the local path, creates any missing parent directories with `os.makedirs(..., exist_ok=True)`, and copies the file using `tf.io.gfile.copy()`. Before copying, it checks whether a local file already exists and whether its size matches the remote file's size via `tf.io.gfile.stat(p).length`; files that match are skipped, making the download resumable after interruption.

**Why TF for download?** `tf.io.gfile` provides a unified interface to GCS that handles authentication, chunked transfers, and retry logic transparently. Using raw `gsutil` or `requests` would require manual handling of all these concerns.

After download, the code walks the local directory to find `BUILDER_DIR` (the subdirectory containing `dataset_info.json`), constructs the TFDS builder with `tfds.builder_from_directory(builder_dir=BUILDER_DIR)`, and reports the number of episodes in each split.

**TF GPU isolation during download:** In notebooks 1, 2, 5, and 6, TensorFlow's GPU memory growth is enabled with `tf.config.experimental.set_memory_growth(gpu, True)` and TF is hard-limited to 512 MB via `tf.config.LogicalDeviceConfiguration(memory_limit=512)`. In notebooks 3 and 4, TF GPU visibility is disabled entirely with `tf.config.set_visible_devices([], 'GPU')`. The rationale is that TF and PyTorch share the same GPU VRAM pool, and if TF pre-allocates a large chunk at import time, the PyTorch model may not have enough VRAM to train. The 512 MB cap is the minimum needed for TF to handle dataset loading operations.

### 4.5 Dataset Structure Exploration (Block 4)

Before building the dataset cache, the code peeks at the first episode to discover the exact field names and dimensions:

```python
# Why: field names vary across OXE sub-datasets; hardcoding them would
#      break the code if used on any other OXE dataset.
IMG_KEY   — discovered by finding the first observation field of shape (H,W,3)
INST_KEY  — discovered by finding the first scalar string/bytes observation field
ACT_KEYS  — sorted list of all keys in the action dictionary
ACTUAL_ACTION_DIM — length of np.concatenate([act[k] for k in ACT_KEYS])
```

For `berkeley_cable_routing`, `ACTUAL_ACTION_DIM = 7` (three world-vector components, three rotation-delta components, one termination flag). The action keys are concatenated in alphabetical order for determinism. All TF operations in this block are pinned to CPU with `with tf.device('/CPU:0')` to avoid any GPU memory allocation at this early stage.

### 4.6 Dataset Class: `CableRoutingDataset`

`CableRoutingDataset` is a PyTorch `Dataset` subclass. Its constructor takes the following arguments:

| Argument | Default | Purpose |
|---|---|---|
| `split` | required | TFDS split string, e.g. `"train[:120]"` |
| `mode` | `"train"` | Controls augmentation; `"train"` applies full augmentation, `"val"` applies only normalisation |
| `seq_len` | `SEQ_LEN=6` | Number of consecutive frames per sample |
| `target_size` | `TARGET_SIZE=224` | Spatial resolution to resize all images to |
| `max_episodes` | `None` | Optional cap for debugging on small subsets |

The constructor first checks whether a prebuilt disk cache exists by testing for the presence of `metadata.npy`. If it does, `_load_cache()` is called. If not, `_build_cache()` is called to construct it from scratch (described in §4.7). After the cache is loaded or built, the constructor:

1. Builds `self.indices` — a list of all valid frame indices. A frame index `i` is valid if the episode containing it has at least `seq_len` frames before `i`, allowing a lookback window of `seq_len` frames to be extracted.
2. Calls `_build_task_index()` to group indices by task instruction string, enabling balanced sampling.
3. Calls `_setup_augmentation()` to initialise the torchvision transform pipeline.

**`__getitem__(idx)`** returns a 4-tuple `(video, instruction, action_tokens, task_id)`:
- `video` — shape `(3, seq_len, 224, 224)` float32 tensor in `[0,1]`, after augmentation.
- `instruction` — raw instruction string (passed as-is to the text encoder).
- `action_tokens` — shape `(NUM_ACTIONS,)` long tensor of discretised action bins (0–255) for the target timestep.
- `task_id` — integer task category index for balanced sampling.

### 4.7 Disk Cache Architecture

The cache stores four files in `cable_cache_{mode}/`:

| File | Format | Content |
|---|---|---|
| `images.dat` | `numpy.memmap` uint8 `(N, H, W, 3)` | All resized RGB frames, row-major |
| `actions.npy` | `numpy` float32 `(N, NUM_ACTIONS)` | Continuous action vectors |
| `instructions.npy` | `numpy` object array `(N,)` | Decoded instruction strings |
| `metadata.npy` | `numpy` structured array | `episode_bounds` list and other metadata |

**Why `numpy.memmap`?** The image data is too large to fit in RAM (potentially tens of gigabytes). `numpy.memmap` maps the file directly into virtual memory, allowing random-access to individual frames without loading the entire file. Each `__getitem__` call reads only `seq_len` consecutive rows from `images.dat`, which corresponds to `seq_len × 224 × 224 × 3` bytes — a very small I/O footprint per sample.

**`_build_cache()` procedure:**
1. First pass over the TFDS dataset: count total frames to determine the `memmap` array shape (needed before writing).
2. Allocate `images.dat` memmap with `mode='w+'` (create and write).
3. Second pass: iterate every episode and every step, resize each image to `(target_size, target_size)` using `PIL.Image.resize(..., BILINEAR)` if needed, write to the memmap, append the action vector and instruction string to lists, and track episode start/end indices in `episode_bounds`.
4. Save `actions.npy`, `instructions.npy`, `metadata.npy`.

**Why bilinear resize?** Bilinear interpolation is a standard choice for downsampling RGB images — it preserves smooth gradients without the aliasing artefacts of nearest-neighbour interpolation, and without the blurring of bicubic.

### 4.8 Data Augmentation Pipeline

Augmentation is applied in `_setup_augmentation()` and called inside `__getitem__`. The pipeline differs by mode:

**Training augmentation** (mode = `"train"`):
```
torchvision.transforms.Compose([
    RandomResizedCrop(224, scale=(0.8, 1.0)),  # randomly crop 80-100% of image area
    RandomHorizontalFlip(p=0.5),               # 50% chance of horizontal mirror
    ColorJitter(brightness=0.3, contrast=0.3,  # random colour perturbation
                saturation=0.2, hue=0.1),
    ToTensor(),                                 # uint8 [0,255] → float32 [0,1]
    Normalize(mean=[0.485,0.456,0.406],         # ImageNet statistics — standard for
              std=[0.229,0.224,0.225])           # pretrained vision encoders
])
```

**Why these augmentations?**
- `RandomResizedCrop`: Simulates variation in camera distance and viewing angle — important for real-robot data where the cable's position changes across episodes.
- `RandomHorizontalFlip`: Breaks spatial bias toward one side of the workspace. While cable routing is not perfectly left-right symmetric, flipping still helps prevent the model from learning spurious positional shortcuts.
- `ColorJitter`: Simulates lighting variation in the robot workspace (different times of day, shadows, reflections off the cable). Small hue variation (`0.1`) is used rather than large to avoid unrealistic colours.
- `Normalize` with ImageNet statistics: All pretrained vision encoders (DistilBERT, BERT, CLIP) were trained on ImageNet-normalised images. Using the same normalisation ensures the backbone's learned feature extractors operate in the expected input range.

**Validation augmentation** (mode = `"val"`):
```
torchvision.transforms.Compose([
    Resize(224),
    CenterCrop(224),
    ToTensor(),
    Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])
```

No random crops or flips are applied during validation; the centre crop provides a deterministic, reproducible view of the image.

**In-batch augmentation — MixUp and CutMix (Block 8, applied in Block 12):** Beyond the per-sample transforms above, the training loop applies `ImprovedMixup` to entire batches, described in §6.2.

### 4.9 Balanced Sampling and Task Index

`_build_task_index()` groups all valid frame indices by their instruction string. `get_balanced_sampler()` returns a `WeightedRandomSampler` where each sample's weight is the inverse frequency of its task class. This ensures that rare task instructions (e.g., episodes with unusual cable positions) are sampled as frequently as common ones, preventing the model from overfitting to the most common task phrasings.

**Why balanced sampling?** In the raw dataset, some cable routing tasks may appear far more often than others (e.g., "route cable through left clip" vs. "route cable through right clip"). Without balancing, the model would implicitly optimise mostly for common tasks and underperform on rare ones.

### 4.10 DataLoader Configuration

```python
loader_train = DataLoader(
    dataset_train,
    batch_size       = BATCH_SIZE,         # 1–4 depending on notebook
    sampler          = train_sampler,       # WeightedRandomSampler (balanced)
    num_workers      = NUM_WORKERS=2,       # parallel data loading workers
    pin_memory       = True,               # pages CPU tensors into non-pageable RAM
    drop_last        = True,               # drop incomplete final batch
    persistent_workers = True,            # keep workers alive between epochs
    prefetch_factor  = PREFETCH_FACTOR=4   # each worker prefetches 4 batches
)
```

| Parameter | Value | Rationale |
|---|---|---|
| `pin_memory=True` | Always | Enables asynchronous CPU→GPU transfer via DMA, reducing GPU idle time |
| `drop_last=True` | Train only | Prevents BatchNorm statistics from being corrupted by a tiny final batch |
| `persistent_workers=True` | When `num_workers>0` | Avoids the ~1–2 second worker process startup overhead at each epoch boundary |
| `prefetch_factor=4` | When `num_workers>0` | Each worker pre-loads 4 batches ahead of the training loop, hiding I/O latency |
| `num_workers=2` | Always | 2 parallel loaders found to be the sweet spot on a single-GPU machine; more workers can cause contention on the memmap file descriptor |

---

## 5. Shared Infrastructure Across All Notebooks

### 5.1 Import Strategy and Rationale (Block 1)

Every notebook begins with an identical import block. The imports are arranged in a deliberate order to avoid GPU memory conflicts between TensorFlow and PyTorch:

```python
# Step 1: Set TF environment flags BEFORE any TF import
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['TF_GPU_ALLOCATOR']          = 'cuda_malloc_async'
```
These two environment variables must be set before TensorFlow is imported, because TF reads them at initialisation time. `TF_FORCE_GPU_ALLOW_GROWTH=true` tells TF not to pre-allocate all available GPU memory at startup. `TF_GPU_ALLOCATOR=cuda_malloc_async` tells TF to use CUDA's asynchronous malloc, which reduces fragmentation and is faster on Ampere/Hopper GPUs.

```python
import torch, torch.nn as nn, torch.nn.functional as F
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce
```
`einops` is used throughout for readable tensor reshape operations. `rearrange('b c f h w -> (b f) c h w')` is far more self-documenting than an equivalent sequence of `.view()` and `.permute()` calls.

```python
from transformers import (
    DistilBertModel, DistilBertTokenizer,
    BertModel, BertTokenizer,
    CLIPTokenizer, CLIPTextModel,
)
```
All three text encoder backends are imported in every notebook, even though each notebook uses only one. This allows the shared Block 6 component definitions to reference all encoder types without conditional imports.

```python
import tensorflow as tf
# Hard-limit TF to 512 MB GPU
tf.config.set_logical_device_configuration(
    gpus[0], [tf.config.LogicalDeviceConfiguration(memory_limit=512)])
import tensorflow_datasets as tfds
```
TF is imported after PyTorch and capped to 512 MB before any dataset operations. The 512 MB figure is chosen as the minimum needed for TFDS to decode TFRecord shards without running out of TF-side memory, while leaving the bulk of VRAM for PyTorch.

### 5.2 CUDA and GPU Optimisations

```python
torch.backends.cuda.matmul.allow_tf32 = True   # (1)
torch.backends.cudnn.allow_tf32       = True   # (2)
cudnn.benchmark                        = True   # (3)
torch.set_float32_matmul_precision('high')      # (4)
torch.set_num_threads(8)                        # (5)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
```

| Line | What it does | Why |
|---|---|---|
| (1) `allow_tf32` for matmul | Permits FP32 matrix multiplications to use TF32 internally on Ampere/Hopper | TF32 has the same dynamic range as FP32 but lower mantissa precision; for deep learning this is nearly always imperceptible while being ~3× faster on A100/H100 |
| (2) `allow_tf32` for cuDNN | Same for convolution kernels | Same rationale; convolutions are the dominant operation in the vision encoders |
| (3) `cudnn.benchmark=True` | Lets cuDNN benchmark multiple kernel implementations at startup and pick the fastest | Beneficial because input shapes are fixed (224×224 images, fixed batch size); the benchmark cost is paid once and amortised across epochs |
| (4) `set_float32_matmul_precision('high')` | Sets the precision level for PyTorch 2.x matmul backend | Works in concert with (1) on newer PyTorch versions; "high" = TF32 allowed |
| (5) `set_num_threads(8)` | Sets the number of OpenMP/MKL threads for CPU operations | Prevents CPU-side operations (e.g. tokeniser, data loading) from spawning excessive threads that compete with GPU transfer threads |

`USE_BFLOAT16=True` and `AUTOCAST_DTYPE=torch.bfloat16` configure the automatic mixed precision (AMP) context used in the training loop (`torch.amp.autocast`). BFloat16 is preferred over Float16 because it has the same exponent range as Float32 (avoiding gradient underflow), and is natively accelerated on Ampere GPUs and H100s. A `torch.amp.GradScaler` is also created (Block 11) to handle the loss scaling that AMP requires; with BFloat16 the scaler is largely a no-op but is retained for correctness.

### 5.3 Reproducibility Seeding

```python
SEED = 42
torch.manual_seed(SEED)     # PyTorch CPU RNG
np.random.seed(SEED)        # NumPy RNG
random.seed(SEED)           # Python stdlib RNG
torch.cuda.manual_seed_all(SEED)  # all CUDA device RNGs
```

`SEED=42` is used universally. Seeding all four RNG sources ensures that weight initialisation, data augmentation (random crops, flips, MixUp beta samples), and dropout masks are all deterministic across runs. This makes it possible to reproduce exactly the same training trajectory if needed.

### 5.4 Universal Utilities (Block 5)

#### `RT1Tokenizer`

```python
class RT1Tokenizer:
    def __init__(self, vocab_size=256, range_min=-3.2, range_max=3.2):
```

| Parameter | Value | Rationale |
|---|---|---|
| `vocab_size=256` | 256 bins | Matches the original RT-1 paper's discretisation. 256 bins give sufficient resolution (~0.025 units per bin) for robot joint velocities |
| `range_min=-3.2` | −3.2 | Empirically covers the full range of end-effector velocities in the cable routing dataset |
| `range_max=3.2` | +3.2 | Same upper bound |

`encode(af)` maps a continuous action vector into integer bins via uniform quantisation:
`bin = clamp(af, min, max) → normalise to [0,1] → scale by (vocab_size-1) → floor to int`

`decode(t)` inverts this: `value = t / (vocab_size-1) × (max-min) + min`

This tokenisation is what allows the model to be trained with cross-entropy loss over a discrete vocabulary rather than with regression loss over continuous values — a key design choice from the original RT-1 paper that makes the action prediction problem analogous to next-token prediction in language modelling.

#### `DropPath`

```python
class DropPath(nn.Module):
    def __init__(self, p=0.0):
```

Also known as Stochastic Depth. During training, an entire residual branch is randomly zeroed with probability `p` (i.e., the shortcut connection alone is used, without the branch transformation). During evaluation, the branch is always used but its output is scaled by `(1-p)` to maintain the expected value. `DropPath` is more powerful than Dropout for residual networks because it regularises at the path level rather than the neuron level. It is applied with `DROP_PATH_RATE=0.2` in most notebooks (0.3 in nb5, 0.15 in nb6 for the more lightweight architectures).

### 5.5 Shared Model Components (Block 6)

These components appear in every notebook because they are building blocks of every vision encoder variant.

#### `ImprovedSqueezeExcitation`

Squeeze-and-Excitation (SE) is a channel-wise attention mechanism. For a feature map `x` of shape `(B, C, H, W)`:

1. **Squeeze:** Global average pooling reduces spatial dims: `(B, C, H, W) → (B, C)` via `einops.Reduce('b c h w -> b c', 'mean')`.
2. **Excitation:** Two linear layers with a bottleneck (reduction ratio = 4): `C → C//4 → C`, with `BatchNorm1d` after the first layer, `SiLU` activation, and `Dropout(0.1)` for regularisation. A `Sigmoid` gate produces channel weights in `[0,1]`.
3. **Scale:** The channel weights are broadcast back to `(B, C, 1, 1)` and multiplied elementwise with `x`.

**Why SE?** It allows the network to recalibrate which feature channels are most informative for the current spatial context, at very low parameter cost (only `2 × C × C/4` parameters). The `Dropout(0.1)` inside the excitation path is a custom addition that adds regularisation to the channel attention, preventing the SE module from becoming overconfident.

#### `ImprovedMBConv`

MBConv (Mobile Inverted Bottleneck Convolution) is the core building block of EfficientNet, MaxViT, and MobileViT. It has two modes:

**Standard MBConv (`use_fused=False`):**
`1×1 expand → BN → SiLU → 3×3 depthwise → BN → SiLU → SE → 1×1 project → BN`

**Fused MBConv (`use_fused=True`):**
`3×3 standard conv → BN → SiLU → SE → 1×1 project → BN`

| Parameter | Value | Rationale |
|---|---|---|
| `expansion_rate=4` | 4× | Standard EfficientNet multiplier; expands channels before depthwise conv to increase representational capacity |
| `downsample` | bool | If True, stride=2 (spatial downsampling); disables residual shortcut |
| `use_fused` | varies | Fused MBConv is faster on modern GPUs (fewer kernel launches) but uses more memory. Enabled in nb1, nb2, nb3, nb4; disabled in nb5 where peak VRAM is the constraint |
| `drop_path_rate` | 0.1–0.3 | Stochastic depth regularisation |
| `se_ratio=0.25` | 0.25 | SE reduction ratio = 4 (1/0.25) |

The residual shortcut is only used when `dim_in == dim_out` and there is no downsampling. This matches the standard residual network convention.

#### `ImprovedTemporalConv`

This module fuses information across the time dimension of a video tensor `(B, D, F, H, W)` where `F` is the number of frames (`SEQ_LEN=6`).

```python
self.temporal_conv = nn.Sequential(
    nn.Conv3d(dim, dim, (3,1,1), padding=(1,0,0), groups=dim),  # depthwise temporal
    nn.BatchNorm3d(dim), nn.GELU())

self.temporal_attn = nn.Sequential(
    nn.AdaptiveAvgPool3d(1),           # global spatio-temporal average
    nn.Conv3d(dim*2, dim, 1),          # squeeze
    nn.GELU(),
    nn.Conv3d(dim, dim*2, 1),          # excite
    nn.Sigmoid())

self.proj = nn.Conv3d(dim*2, dim, 1)   # merge channels back
```

**Forward pass:**
1. Apply depthwise 3D conv `(3,1,1)` — convolves only over the time axis (kernel size 3, stride 1, padding 1 for temporal causal blending), with `groups=dim` making it channel-independent. This is fast because it does not mix channels or spatial positions.
2. Concatenate the original `x` and the temporally smoothed output along the channel dim: `cat = [x, temporal_conv(x)]`, giving `(B, 2D, F, H, W)`.
3. Compute a 3D SE-like gate on `cat` using global average pooling followed by a 2-layer MLP. The gate has shape `(B, 2D, 1, 1, 1)` and is broadcast over the spatial/temporal dims.
4. Apply the gate: `cat × temporal_attn(cat)`, then project back to `D` channels with a `1×1×1` conv.
5. Add residual: `x + proj(...)`.

**Why this design?** A pure 3D convolution over all `(T, H, W)` dimensions would be computationally expensive. By separating the temporal convolution (depthwise, time-only) from the spatial features (already captured by the ViT), the module is efficient and does not interfere with spatial feature maps. The gating mechanism ensures the module can selectively amplify frames with the most motion (e.g., the moment of cable grasping) and suppress near-static frames.

---

## 6. Shared Training Utilities (Block 8)

Block 8 is identical across all six notebooks. It defines the loss function, augmentation, optimiser, scheduler, EMA tracker, and per-task metrics.

### 6.1 `ImprovedFocalLoss`

```python
class ImprovedFocalLoss(nn.Module):
    def __init__(self, gamma=2.5, label_smoothing=0.1, temperature=1.0):
```

| Parameter | Value | Rationale |
|---|---|---|
| `gamma=2.5` | 2.5 | Focal loss focusing parameter; down-weights easy examples (where `p_t` is near 1.0). Standard Focal loss uses `γ=2.0`; `2.5` gives slightly stronger focus on hard examples |
| `label_smoothing=0.1` | 0.1 | Replaces hard 0/1 targets with `ε/(K-1)` and `1-ε`; reduces overconfidence and improves calibration. Applied via `F.cross_entropy(..., label_smoothing=0.1)` |
| `temperature=1.0` | 1.0 | Logit temperature scaling; kept at 1.0 during training but can be modified for inference calibration |

**Forward computation:**
1. Divide logits by temperature: `logits / self.temperature`.
2. Compute standard cross-entropy per sample: `ce = F.cross_entropy(logits, targets, reduction='none', label_smoothing=self.label_smoothing)`.
3. Compute the predicted probability for the true class: `p_t = softmax(logits).gather(1, targets)`.
4. Apply focal weighting: `loss = mean((1 - p_t)^gamma * ce)`.

**Why Focal over plain CE?** The action prediction task is a 256-class classification (one class per bin) per action dimension. In early training, most predictions are confident on incorrect bins, creating many hard examples. Focal loss concentrates gradient signal on these hard examples, accelerating convergence. Label smoothing prevents the model from assigning probability 1.0 to any single bin, which is important given that the discretisation introduces quantisation noise.

### 6.2 `ImprovedMixup` with CutMix

```python
class ImprovedMixup(nn.Module):
    def __init__(self, alpha=0.3, use_cutmix=True, cutmix_prob=0.5):
```

| Parameter | Value | Rationale |
|---|---|---|
| `alpha=0.3` | 0.3 | Beta distribution concentration parameter for MixUp. `α=0.3` gives a distribution concentrated near `λ=1.0` (mostly original sample) with occasional strong mixing |
| `use_cutmix=True` | True | Enable CutMix augmentation |
| `cutmix_prob=0.5` | 0.5 | 50% of batches use CutMix; the other 50% use standard MixUp interpolation |

**MixUp forward:** Samples `λ ~ Beta(α, α)`. Computes `x_mixed = λ·x + (1-λ)·x[shuffle]` where `shuffle` is a random permutation of the batch. Returns `(x_mixed, y_a, y_b, λ)` where `y_a` and `y_b` are the original and shuffled targets.

**CutMix forward:** Samples `λ ~ Beta(α, α)`. Computes a random bounding box of area `(1-λ) × W × H`. Copies the bounding box region from the shuffled image into the original image. Recomputes the actual `λ` as `1 - box_area / (W×H)` to account for quantisation in the box coordinates.

**Why both?** MixUp is most effective at regularising the internal feature representations (interpolated inputs force smooth decision boundaries). CutMix is more realistic — it produces images that are spatially coherent (real image patches rather than blended ghosts), which is important for visual models that rely on local texture and edge features.

The mixed-up loss in the training loop is computed as: `loss = λ · criterion(logits, y_a) + (1-λ) · criterion(logits, y_b)`.

### 6.3 `Lion` Optimiser

```python
class Lion(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-4, betas=(0.9, 0.99), weight_decay=0.0):
```

Lion (EvoLved Sign Momentum, Chen et al. 2023) is a memory-efficient optimiser that uses only the sign of the gradient update rather than its magnitude. The update rule is:

1. Compute the interpolated gradient: `update = exp_avg * β₁ + grad * (1 - β₁)`
2. Apply weight decay directly to the parameter (decoupled): `p ← p * (1 - lr * weight_decay)`
3. Update the parameter by the sign: `p ← p - lr * sign(update)`
4. Update the exponential moving average: `exp_avg ← exp_avg * β₂ + grad * (1 - β₂)`

| Hyperparameter | Value (most notebooks) | Rationale |
|---|---|---|
| `lr=3e-5` | nb1, nb2, nb6 | Lower LR than Adam because Lion's sign update is inherently larger in magnitude; 3e-5 is empirically stable for transformer-scale models |
| `lr=1e-4` | nb3 | Lightweight model (ConvNeXt-Small) tolerates a higher LR |
| `lr=2e-5` | nb5 | Maximum-capacity model is sensitive to large updates; lower LR is safer |
| `betas=(0.9, 0.99)` | All | β₁ controls the current gradient weight in the update direction; β₂ controls the momentum EMA decay rate. These are the values recommended in the Lion paper |
| `weight_decay=0.15` | Most | Aggressive L2 regularisation; partially compensates for Lion's larger effective step size compared to Adam |
| `weight_decay=0.1` | nb3 | Lighter model needs less regularisation |

**Why Lion over Adam?** Lion requires half the memory of Adam (one momentum buffer per parameter vs. two for Adam+Adam2nd moment). This matters significantly in nb5 where VRAM is the binding constraint. Lion has also been shown to converge faster than AdamW on vision transformer tasks at scale.

### 6.4 `ImprovedScheduler`

```python
class ImprovedScheduler:
    def __init__(self, optimizer, warmup_steps=500, max_steps=10000, min_lr_ratio=0.1):
```

Implements a linear warmup followed by cosine annealing:

- **Warmup phase** (`step < warmup_steps`): LR scales linearly from 0 to the base LR: `lr = base_lr × (step / warmup_steps)`.
- **Cosine decay phase** (`step ≥ warmup_steps`): `lr = base_lr × (min_ratio + (1 - min_ratio) × 0.5 × (1 + cos(π × progress)))` where `progress = (step - warmup) / (max_steps - warmup)`.

| Hyperparameter | Value | Rationale |
|---|---|---|
| `warmup_steps=500` | nb1,nb2,nb4,nb6 | 500 gradient update steps (~64k samples at batch=1/accum=128). Warmup prevents catastrophic early updates when gradients are largest and parameter norms are random |
| `warmup_steps=300` | nb3 | Lighter model converges faster; shorter warmup is sufficient |
| `warmup_steps=600` | nb5 | Heavier model benefits from a longer warmup to stabilise the larger gradient norms |
| `min_lr_ratio=0.1` | All | LR decays to 10% of peak at the end of training, rather than to 0. This prevents complete learning stagnation in the final epochs |

### 6.5 `ModelEMA`

Exponential Moving Average of model weights: `ema_weights = decay × ema_weights + (1 - decay) × model_weights`.

```python
ema = ModelEMA(model, decay=0.9999)
```

`decay=0.9999` means the EMA weights update very slowly — the EMA model is an extremely smooth average of all model checkpoints seen during training. At test/inference time the EMA model is used instead of the raw model. This consistently produces 0.5–2% lower validation loss than the last checkpoint model because the EMA is less sensitive to the stochasticity of the final training batches.

### 6.6 `PerTaskMetrics`

Tracks per-task-category loss and accuracy. Each task instruction is mapped to a `task_id` integer. `PerTaskMetrics.update(loss, task_id)` accumulates losses per task. `PerTaskMetrics.report()` prints a breakdown of which task categories the model performs best and worst on. This is important for the cable routing task where different instruction phrasings may have very different difficulty levels.

---

## 7. Training Setup (Block 11) — Shared Pattern

```python
# Block 11 — identical across all notebooks
optimizer    = Lion(model.parameters(), lr=LR, betas=LION_BETAS, weight_decay=WEIGHT_DECAY)
total_steps  = len(loader_train) * EPOCHS // GRADIENT_ACCUM_STEPS
scheduler    = ImprovedScheduler(optimizer, warmup_steps=WARMUP_STEPS, max_steps=total_steps)
criterion    = ImprovedFocalLoss(gamma=FOCAL_GAMMA, label_smoothing=LABEL_SMOOTHING)
mixup        = ImprovedMixup(alpha=MIXUP_ALPHA, use_cutmix=USE_CUTMIX)
ema          = ModelEMA(model, decay=0.9999)
scaler       = torch.amp.GradScaler('cuda', enabled=USE_BFLOAT16)
train_metrics = PerTaskMetrics()
val_metrics   = PerTaskMetrics()
```

`total_steps` is the total number of Lion optimiser steps that will occur over all epochs. It is `len(loader_train) × EPOCHS / GRADIENT_ACCUM_STEPS` because the optimiser only steps once per `GRADIENT_ACCUM_STEPS` mini-batches. This value is passed to `ImprovedScheduler` so the cosine decay reaches its minimum exactly at the last training step.

`GradScaler` is created with `enabled=USE_BFLOAT16=True`. In BFloat16 mode, gradients in the backward pass can underflow to zero. The scaler multiplies the loss by a large scale factor before the backward pass and divides the gradients back before the optimiser step, preserving gradient signal. For BFloat16 the risk is lower than for Float16, but the scaler is kept for safety.

---

## 8. Training Loop (Block 12) — Shared Pattern

The core training loop structure is identical in all six notebooks:

```python
for epoch in range(EPOCHS):  # EPOCHS=80

    # --- TRAIN ---
    model.train()
    optimizer.zero_grad()

    for batch_idx, (video, instructions, target, task_ids) in enumerate(loader_train):
        video   = video.to(DEVICE, non_blocking=True)   # async GPU transfer
        target  = target.to(DEVICE, non_blocking=True)

        with torch.no_grad():
            text_emb = text_encoder.encode(instructions)  # frozen encoder: no grad needed

        video, target_a, target_b, lam = mixup(video, target)  # in-batch augmentation

        with torch.amp.autocast('cuda', dtype=AUTOCAST_DTYPE):  # BFloat16 forward pass
            logits = model(video, text_emb)                      # (B, A, BIN) or (B, C, A, BIN) for nb6
            logits_flat = rearrange(logits, 'b a bin -> (b a) bin')
            loss = (lam*criterion(logits_flat, ta_flat) +
                    (1-lam)*criterion(logits_flat, tb_flat)) / GRADIENT_ACCUM_STEPS

        scaler.scale(loss).backward()  # scaled backward pass

        if (batch_idx + 1) % GRADIENT_ACCUM_STEPS == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()
            ema.update(model)

    # --- VALIDATE ---
    model.eval()
    with torch.no_grad():
        for video, instructions, target, task_ids in loader_val:
            ...  # same forward, no backward, no mixup
```

**Key design decisions in the loop:**

| Design choice | Rationale |
|---|---|
| `GRADIENT_ACCUM_STEPS=128` (most notebooks) | Effective batch size = 128×1=128 while fitting in 8 GB VRAM. Gradient accumulation simulates a large batch by summing gradients over 128 micro-batches before each optimiser step |
| `GRADIENT_ACCUM_STEPS=32` (nb3) | Lightweight model fits batch_size=4, so fewer accumulation steps needed for effective batch=128 |
| `non_blocking=True` on `.to(DEVICE)` | Overlaps CPU→GPU data transfer with the previous GPU kernel execution, reducing idle time |
| `text_encoder.encode` wrapped in `torch.no_grad()` | Text encoder is frozen (or mostly frozen); not computing gradients saves both memory and compute |
| `GRAD_CLIP=1.0` | Clips gradient L2 norm to 1.0 before each optimiser step. Prevents gradient explosion, which is especially important for Lion whose sign-based update can amplify large gradients |
| `PATIENCE=10` | Early stopping: if validation loss does not improve for 10 consecutive epochs, training terminates to prevent overfitting |
| `history` dict | Accumulates `train_loss`, `val_loss`, `lr`, and `throughput` (samples/second) per epoch for plotting in Block 13 |

---

## 9. Training Plots (Block 13)

Block 13 generates a 3-panel matplotlib figure saved as `{MODEL_NAME}_training.png`:

- **Panel 1 — Loss curves:** Training and validation loss per epoch, with markers at each point. Enables visual diagnosis of overfitting (diverging train/val) vs. underfitting (both high).
- **Panel 2 — Learning rate:** LR plotted on a log scale per epoch. The warmup ramp and cosine decay shape are clearly visible. Log scale is used because the LR varies over 1–2 orders of magnitude.
- **Panel 3 — Throughput:** Samples per second per epoch, with a red dashed horizontal line at the mean. Throughput drops in early epochs when cuDNN benchmark is running; this panel identifies such anomalies.

---

## 10. Notebook 1 — RT-1 Base: DistilBERT + MaxViT + CrossAttn TokenLearner

### Configuration Variables (Block 2)

```python
MODEL_NAME    = "rt1_base_maxvit"
VIS_ENC_NAME  = "MaxViT-Base"
TEXT_ENC_NAME = "DistilBERT"
TL_NAME       = "CrossAttn"

TARGET_SIZE   = 224    # image resolution; 224 is the standard for most pretrained ViTs
SEQ_LEN       = 6     # number of frames per video clip; covers ~0.5s at ~12fps
SPLIT_RATIO   = 0.8   # 80% train, 20% val split at episode level
ACTION_BINS   = 256   # discretisation vocabulary for RT1Tokenizer

VIT_DIM       = 40    # base channel dimensionality of MaxViT; determines embed_dim
VIT_DEPTH     = (4,4,3)  # number of blocks in each of 3 MaxViT stages
TRANSFORMER_DEPTH = 6    # number of layers in the action-prediction transformer
TRANSFORMER_HEADS = 8    # attention heads in each transformer layer
DROPOUT       = 0.2      # dropout rate in transformer and projection layers
NUM_VISUAL_TOKENS = 32   # number of tokens output by the TokenLearner
DROP_PATH_RATE = 0.2     # stochastic depth rate for MaxViT blocks
USE_FUSED_MBCONV = True  # use fused MBConv (faster, slightly more VRAM)

BATCH_SIZE    = 1        # micro-batch size; VRAM constraint
GRADIENT_ACCUM_STEPS = 128  # effective batch = 128
EPOCHS        = 80       # total training epochs
LR            = 3e-5     # Lion peak learning rate
WEIGHT_DECAY  = 0.15     # decoupled L2 penalty in Lion
LION_BETAS    = (0.9, 0.99)  # momentum interpolation coefficients
WARMUP_STEPS  = 500      # number of optimiser steps for linear LR warmup
GRAD_CLIP     = 1.0      # gradient norm clip threshold

LABEL_SMOOTHING = 0.1    # Focal loss label smoothing ε
MIXUP_ALPHA   = 0.3      # Beta distribution concentration for MixUp
USE_CUTMIX    = True     # enable CutMix augmentation
FOCAL_GAMMA   = 2.5      # Focal loss γ parameter
PATIENCE      = 10       # early stopping patience (epochs without val improvement)

NUM_WORKERS   = 2        # DataLoader worker processes
PREFETCH_FACTOR = 4      # batches prefetched per worker
USE_BFLOAT16  = True     # enable BFloat16 AMP
AUTOCAST_DTYPE = torch.bfloat16
```

### Vision Encoder: `AdvancedMaxViT` (MaxViT-Base)

MaxViT (Multi-Axis Vision Transformer, Tu et al. 2022) is a hybrid CNN-Transformer that alternates between local MBConv blocks and two complementary attention mechanisms — window attention (local) and grid attention (global) — at each scale. This alternation allows MaxViT to capture both fine-grained local texture (relevant for detecting cable grip points) and global spatial relationships (relevant for understanding the full cable routing path).

**Architecture flow for a 224×224 input:**

```
Input: (B, 3, 224, 224)
  ↓
Stem: Conv 3×3, stride=1, dim_in=3→VIT_DIM=40, BN, SiLU
      → (B, 40, 224, 224)

Stage 1 (4 blocks, stride=2 at first block):
  Block 0: ImprovedMBConv(40→80, downsample=True)    → (B, 80, 112, 112)
  Blocks 1-3: MaxViT blocks at 112×112
    Each MaxViT block:
      ImprovedMBConv(80→80, no downsample)
      Window Attention (local 8×8 windows)
      Grid Attention (dilated global grid)
    → (B, 80, 112, 112)

Stage 2 (4 blocks, stride=2):
  Block 0: downsample 80→160 → (B, 160, 56, 56)
  Blocks 1-3: MaxViT blocks at 56×56
    → (B, 160, 56, 56)

Stage 3 (3 blocks, stride=2):
  Block 0: downsample 160→320 → (B, 320, 28, 28)
  Blocks 1-2: MaxViT blocks at 28×28
    → (B, 320, 28, 28)

Output: (B, embed_dim=320, 28, 28)   ← spatial feature map passed to TokenLearner
```

**Why MaxViT?** Pure CNNs (like ConvNeXt) lack global attention, limiting their ability to reason about long-range spatial relationships such as the full cable path. Pure ViTs lack local inductive bias, requiring much more data. MaxViT's alternating local-global structure provides the best of both worlds, and its multi-axis design is significantly more efficient than full self-attention (which would be O(N²) for 28×28=784 tokens).

**Embed dim derivation:** `embed_dim = VIT_DIM × 2^(num_stages) = 40 × 2^3 = 320`. Each stage doubles the channel dimension while halving spatial resolution.

### Text Encoder: `DistilBERTEncoder`

```python
text_encoder = DistilBERTEncoder(device=DEVICE, freeze=True)
```

DistilBERT (Sanh et al. 2019) is a distilled version of BERT that is 40% smaller and 60% faster while retaining 97% of BERT's performance on GLUE. It has 6 transformer layers, 12 attention heads, and a hidden size of 768.

`DistilBERTEncoder.encode(instructions)` tokenises a list of instruction strings using `DistilBertTokenizer`, runs a forward pass through the frozen `DistilBertModel`, and returns the CLS token embedding of shape `(B, 768)`. The model is kept entirely frozen (`freeze=True`) because the instruction vocabulary in the cable routing dataset is narrow and repetitive — the model's pretrained representations are already sufficient, and fine-tuning would risk forgetting general language understanding while overfitting to a small set of cable routing phrases.

**Why DistilBERT in nb1?** It is the lightest pretrained language model that provides contextualised, bidirectional word representations. It serves as the baseline text encoder against which BERT (nb5), CLIP (nb4, nb6), and BoW (nb3) are compared.

### Token Learner: `AdvancedTokenLearner` (CrossAttn, 32 tokens)

The token learner compresses the spatial feature map from the vision encoder (shape `(B, D, H, W)` = `(B, 320, 28, 28)`) into a small set of `NUM_VISUAL_TOKENS=32` compact tokens.

`AdvancedTokenLearner` uses **cross-attention**: 32 learnable query vectors (initialised randomly, shape `(1, 32, D)`) attend to the full spatial feature map (flattened to `(B, H×W, D)` = `(B, 784, 320)`) via multi-head cross-attention. The output is 32 attended vectors of shape `(B, 32, D)`.

**Why 32 tokens?** This is a dramatic compression from 784 spatial positions to 32 tokens (24× reduction). The transformer in Block 7 processes a sequence of `SEQ_LEN × 32 = 192` visual tokens plus 1 text token = 193 tokens total. At 32 tokens per frame the self-attention cost is O(193²) ≈ 37,249 operations, which is manageable. Using all 784 tokens per frame would give O(4,705²) ≈ 22M operations — infeasible at batch_size=1.

**Why cross-attention?** Unlike the convolutional TokenLearner (nb3), cross-attention allows each learned query to attend to any spatial position, giving the model flexibility to learn which regions of the image are most task-relevant (e.g., the cable, the clips). The learned queries specialise: after training, different queries attend to the cable body, the clip positions, the gripper, and the background respectively.

### Full Model: `ProductionRT1`

```python
class ProductionRT1(nn.Module):
    def __init__(self, vit, text_dim=768, num_actions=7, action_bins=256,
                 depth=6, heads=8, dropout=0.2, num_visual_tokens=32):
```

**Forward pass (step by step):**

```
Input: video (B, 3, F, H, W) = (B, 3, 6, 224, 224)
       text_embeddings (B, 768)

Step 1 — Flatten frames for batch processing:
  flat = rearrange(video, 'b c f h w -> (b f) c h w')
  → shape: (B×F, 3, 224, 224) = (B×6, 3, 224, 224)

Step 2 — Vision encoder:
  vit_out = self.vit(flat)
  → shape: (B×6, 320, 28, 28)
  Unflatten back: (B, 320, F, 28, 28)

Step 3 — Temporal fusion:
  vit_out = self.temporal_model(vit_out)
  → shape: (B, 320, F, 28, 28)   [frames have been contextualised]
  Flatten frames: (B×6, 320, 28, 28)

Step 4 — Token learning:
  tokens = self.token_learner(vit_out)
  → shape: (B×6, 32, 320)   [32 tokens per frame]
  Unflatten: (B, F×32, 320) = (B, 192, 320)

Step 5 — Text projection:
  text_tok = self.text_proj(text_embeddings)  [768 → 320 via Linear+LN+Dropout]
  text_tok = text_tok.unsqueeze(1)
  → shape: (B, 1, 320)

Step 6 — Transformer:
  sequence = cat([text_tok, tokens], dim=1)
  → shape: (B, 193, 320)
  attended = self.transformer(sequence)       [6-layer, 8-head, pre-norm transformer]
  → shape: (B, 193, 320)

Step 7 — Action logits from CLS (text) token:
  logits = self.to_logits(attended[:, 0, :])  [LayerNorm → Linear(320,160) → GELU → Dropout → Linear(160, 7×256)]
  → shape: (B, 7×256) = (B, 1792)
  Reshape: (B, 7, 256) = (B, NUM_ACTIONS, ACTION_BINS)
```

**Why the text token acts as CLS?** Placing the text token at position 0 and reading the action logits from position 0 means the text token aggregates information from all 192 visual tokens through the transformer's self-attention mechanism. This is analogous to BERT's CLS token: the text instruction guides what information the model should extract from the visual sequence. At convergence, the attended text token encodes "given this instruction, what actions should I predict?" conditioned on the full visual context.

**Transformer specification:** `pre-norm` (LayerNorm before attention and FFN, not after) is used because it has been shown to improve training stability for deep transformers. `activation='gelu'` is used in the FFN because GELU has become standard in transformer architectures after GPT and BERT.

---

## 11. Notebook 2 — RT-2 VLA + CoT: DistilBERT+VQA + MaxViT + ChainOfThought Head

Notebook 2 shares all components with nb1 except for two additions: (1) VQA-style instruction formatting and (2) a `ChainOfThoughtHead` replacing the standard action head.

### Configuration Differences

```python
PLAN_LOSS_WEIGHT = 0.3   # weight of auxiliary planning loss vs action loss
PLAN_VOCAB = ["pick", "route cable", "place", "adjust", "terminate"]
MODEL_NAME = "rt2_vla_cot"
TEXT_ENC_NAME = "DistilBERT+VQA"
TL_NAME = "CrossAttn+CoT"
```

`PLAN_LOSS_WEIGHT=0.3` means the total loss is `action_loss + 0.3 × plan_loss`. The 0.3 weight is chosen so the planning signal is strong enough to guide learning without overwhelming the primary action prediction objective.

### VQA Instruction Formatting: `VQAFormatter`

```python
class VQAFormatter:
    TEMPLATE = "Q: what action should the robot take to {task}? A:"
    def format(self, task):
        return self.TEMPLATE.format(task=task.strip())
```

In Block 10, the text encoder's `.encode()` method is monkey-patched to first pass instructions through the formatter:

```python
text_encoder.encode = lambda instructions: _orig_encode(
    [formatter.format(i) for i in instructions])
```

**Why VQA formatting?** DistilBERT is pretrained on a corpus of Wikipedia and BooksCorpus text. Its representations are optimised for understanding declarative sentences. By reformatting the instruction as a question-answer pair, we align the input distribution more closely with the factoid QA patterns present in BERT's pretraining data. This is a zero-cost improvement — no additional parameters, no additional computation — that can improve the quality of the CLS embedding extracted from the model.

### Chain-of-Thought Head: `ChainOfThoughtHead`

```python
class ChainOfThoughtHead(nn.Module):
    def __init__(self, embed_dim, num_actions=7, action_bins=256,
                 plan_vocab=None, dropout=0.1):
```

This module replaces the simple `to_logits` sequential in `ProductionRT1`. It operates on the CLS token from the transformer output.

**Forward pass:**

```
cls_token: (B, embed_dim)

Step 1 — Plan prediction:
  plan_logits = plan_head(cls_token)
  → shape: (B, 5)   [logits over 5 plan categories]

Step 2 — Plan embedding lookup:
  plan_idx = argmax(plan_logits)        [at inference]
            OR plan_gt                   [at training, teacher forcing]
  plan_emb = plan_embed(plan_idx)
  → shape: (B, embed_dim)

Step 3 — Fusion:
  fused = fusion(cat([cls_token, plan_emb], dim=-1))
  → Linear(2×embed_dim → embed_dim) → GELU → Dropout → LayerNorm
  → shape: (B, embed_dim)

Step 4 — Action prediction:
  act_flat = action_head(fused)
  → Linear(embed_dim → embed_dim//2) → GELU → Dropout → Linear → (B, 7×256)
  Reshape: (B, 7, 256)
```

**Teacher forcing during training:** When `plan_gt` is provided (ground-truth plan label), the model uses the correct plan index in Step 2. This prevents compounding errors during early training when the plan prediction head is uninitialised. At inference, the model uses its own predicted plan (`plan_logits.argmax()`).

**Full `RT2VLAModel` training loss:**

```python
plan_logits, action_logits = model(video, text_emb, plan_gt=plan_labels)
action_loss = criterion(action_logits_flat, target_flat)
plan_loss   = F.cross_entropy(plan_logits, plan_labels)
total_loss  = action_loss + PLAN_LOSS_WEIGHT * plan_loss
```

---

## 12. Notebook 3 — Lightweight Ablation: BoW + ConvNeXt-Small + Conv TokenLearner

This notebook is designed as the minimum-resource configuration. It answers the question: "how well can we do without any pretrained language model, and with a purely convolutional vision backbone?"

### Configuration Differences

```python
TEXT_DIM      = 256          # BoW embedding dimension (much smaller than 768)
TRANSFORMER_DEPTH = 4        # fewer transformer layers (vs 6 in nb1)
TRANSFORMER_HEADS = 4        # fewer attention heads (vs 8 in nb1)
NUM_VISUAL_TOKENS = 16       # fewer tokens (vs 32 in nb1) — less memory
BATCH_SIZE    = 4            # larger batch fits because model is smaller
GRADIENT_ACCUM_STEPS = 32   # effective batch = 4×32 = 128
LR            = 1e-4         # higher LR tolerated by smaller model
WEIGHT_DECAY  = 0.1          # lighter regularisation
WARMUP_STEPS  = 300          # faster warmup
MODEL_NAME    = "rt1_lightweight_bow_convnext"
```

**TF GPU isolation difference:** nb3 fully disables TF GPU visibility with `tf.config.set_visible_devices([], 'GPU')` before any TF import, because the lightweight model has the tightest VRAM budget.

### Text Encoder: `BoWTextEncoder`

BoW (Bag-of-Words) is the simplest possible text encoder. It maintains a learned embedding table of size `(vocab_size, out_dim=256)`. For each instruction string, it tokenises by splitting on whitespace, looks up each word's embedding, and averages all embeddings to produce a fixed-size vector of shape `(B, 256)`.

```python
class BoWTextEncoder:
    def __init__(self, out_dim=256, device='cuda'):
        self.embed = nn.Embedding(vocab_size, out_dim)  # fully trainable
```

**Why BoW?** To establish a lower bound on the contribution of the text encoder. If BoW achieves similar performance to DistilBERT on cable routing, it suggests that the language signal in this task is so simple (few unique instructions, short sentences) that complex contextualised representations are unnecessary. BoW also eliminates the frozen-encoder overhead, allowing the text representations to co-adapt with the visual features during training.

**Why `out_dim=256` instead of 768?** The lighter vision encoder (ConvNeXt-Small) produces a smaller `embed_dim`. Matching the text projection to this smaller dimension reduces the projection layer's parameter count.

### Vision Encoder: `ConvNeXtEncoder` (ConvNeXt-Small)

ConvNeXt (Liu et al. 2022) is a pure convolutional architecture that redesigns ResNet blocks inspired by transformer design principles: depthwise convolutions with large kernels (7×7), inverted bottlenecks, GELU activations, LayerNorm instead of BatchNorm, and separate downsampling layers.

```python
vit = ConvNeXtEncoder(size='small', drop_path_rate=0.15)
# dims = (48, 96, 192)
# embed_dim = 192
```

`ConvNeXt-Small` with `dims=(48, 96, 192)` is a custom small configuration (not the standard ConvNeXt-S with 96-192-384-768 channels) that keeps memory consumption low. The output is a spatial feature map `(B, 192, 28, 28)`.

**Why ConvNeXt as ablation?** ConvNeXt has no attention mechanism whatsoever — it is a pure CNN with transformer-like design choices. Comparing nb3 (ConvNeXt) to nb1 (MaxViT) directly measures the value of attention-based global reasoning in the vision encoder for robot action prediction.

### Token Learner: `ConvTokenLearner` (16 tokens, no attention)

`ConvTokenLearner` replaces cross-attention with a convolutional gating mechanism:

```python
class ConvTokenLearner(nn.Module):
    def __init__(self, dim, num_tokens=16):
        self.attn_maps = nn.Sequential(
            nn.Conv2d(dim, num_tokens, 1),   # 1×1 conv: produces N attention maps
            nn.Sigmoid())
        self.proj = nn.Linear(dim, dim)
```

For each of `N=16` tokens, a sigmoid-gated spatial average pooling is computed: the `1×1` conv produces `N` attention maps of shape `(B, N, H, W)`, each map is used to compute a weighted average over the spatial feature map, yielding one `D`-dimensional vector per token. Final output shape: `(B, N, D)` = `(B, 16, 192)`.

**Why no attention?** The ablation design requires isolating the contribution of attention in the token learner. The convolutional token learner's attention maps are input-independent per-position weights (learned spatial saliency) rather than query-conditioned cross-attention. This tests whether dynamic cross-attention is worth the computational cost.

### Full Model: `LightweightRT1`

```python
class LightweightRT1(nn.Module):
    def __init__(self, vit, text_dim=256, num_actions=7, action_bins=256,
                 depth=4, heads=4, dropout=0.2, num_visual_tokens=16):
```

Identical in structure to `ProductionRT1` (nb1) except: `ConvNeXtEncoder` instead of `AdvancedMaxViT`, `ConvTokenLearner` instead of `AdvancedTokenLearner`, 4-layer/4-head transformer instead of 6-layer/8-head, and `text_dim=256`. This is the minimum-parameter configuration and can be trained on GPUs with as little as 6 GB VRAM.

---

## 13. Notebook 4 — CLIP-Powered: CLIPText + EfficientLite-M + Hierarchical TokenLearner

This notebook tests whether CLIP's joint vision-language pre-training produces better instruction representations for robot action prediction, paired with a lightweight depthwise-separable CNN vision encoder.

### Configuration Differences

```python
MODEL_NAME    = "rt1_clip_efficientlite"
VIS_ENC_NAME  = "EfficientLite-M"
TEXT_ENC_NAME = "CLIP-ViT-B/32"
TL_NAME       = "Hierarchical"
# TEXT_DIM = 512  (CLIP text encoder output)
```

### Text Encoder: `CLIPTextEncoder` (CLIP ViT-B/32, frozen)

```python
text_encoder = CLIPTextEncoder(device=DEVICE, freeze=True)
# embed_dim = 512
```

CLIP (Contrastive Language-Image Pretraining, Radford et al. 2021) is trained to align image and text embeddings in a shared 512-dimensional space. The CLIP text encoder is a transformer that produces a 512-dimensional embedding from the text token at the EOS (end-of-string) position. Because CLIP was trained on 400 million image-text pairs including a wide range of object manipulation descriptions, its text representations may better capture the spatial and action semantics of robot instructions than DistilBERT, which was trained purely on text.

**Why frozen?** CLIP's vision-language alignment is the key property being tested. Fine-tuning could distort this alignment. The hypothesis is: does CLIP's inherent understanding of visual concepts (cables, clips, grippers) in the text embedding provide better conditioning for robot action prediction?

### Vision Encoder: `EfficientLiteEncoder` (EfficientLite-M)

`EfficientLite-M` is a custom depthwise-separable CNN inspired by EfficientNet-Lite but with a reduced channel count (`dim=96`) targeting mobile/edge deployment VRAM budgets.

```
Input: (B, 3, 224, 224)
Stem: Conv 3×3 stride=2 → (B, 32, 112, 112)
Stage 1: 2× MBConv(32→48, stride=2) → (B, 48, 56, 56)
Stage 2: 3× MBConv(48→96, stride=2) → (B, 96, 28, 28)
Stage 3: 2× MBConv(96→96, stride=1) → (B, 96, 28, 28)
Output: (B, 96, 28, 28)  embed_dim=96
```

All MBConv blocks use `use_fused=False` (depthwise-separable) for parameter efficiency. No transformer blocks — this is a pure CNN like nb3 but with a different architecture family. Comparing nb4 (EfficientLite + CLIP) to nb3 (ConvNeXt + BoW) tests whether CLIP's stronger language representations can compensate for a similar vision backbone quality.

### Token Learner: `HierarchicalTokenLearner` (32 tokens, fine + coarse)

The hierarchical token learner produces tokens at two scales:

```
Stage 1 — Fine tokens (16 tokens at 28×28 resolution):
  ConvTokenLearner(dim=96, num_tokens=16) on the full feature map
  → (B, 16, 96)

Stage 2 — Coarse tokens (16 tokens at 14×14 resolution):
  MaxPool2d(2,2) on feature map → (B, 96, 14, 14)
  ConvTokenLearner(dim=96, num_tokens=16) on downsampled map
  → (B, 16, 96)

Concatenate: (B, 32, 96)
```

**Why hierarchical?** The cable routing task requires both local precision (where exactly is the cable end, which clip is closest?) and global context (what is the overall layout of the cable path?). Fine tokens capture local details; coarse tokens capture global structure. The concatenation gives the downstream transformer access to both levels simultaneously. This is the token-level analogue of feature pyramid networks (FPN) in object detection.

---

## 14. Notebook 5 — Maximum Capacity: BERT-ft2 + MaxViT-Large + SelfAttn TokenLearner

This notebook is the high-watermark configuration, designed to establish the performance ceiling on the cable routing task. It uses the largest vision encoder, the most powerful text encoder (with partial fine-tuning), and the most expressive token learner.

### Configuration Differences

```python
VIT_DIM       = 64           # larger base dim → embed_dim = 64×2³ = 512
VIT_DEPTH     = (6,5,5)      # deeper MaxViT (6+5+5=16 blocks vs 4+4+3=11 in nb1)
TRANSFORMER_DEPTH = 8        # deeper action transformer (vs 6 in nb1)
NUM_VISUAL_TOKENS = 64       # more tokens (vs 32 in nb1) → richer representation
DROP_PATH_RATE = 0.3         # higher stochastic depth because model is deeper
USE_FUSED_MBCONV = False      # disable fused MBConv to reduce peak VRAM

BATCH_SIZE    = 1
GRADIENT_ACCUM_STEPS = 128
LR            = 2e-5          # lower LR for stable training of larger model
WARMUP_STEPS  = 600           # longer warmup for heavier model
MODEL_NAME    = "rt1_large_bert_maxvit"
```

**`embed_dim = VIT_DIM × 2³ = 64 × 8 = 512`** — a 60% larger hidden dimension than nb1's 320. This substantially increases both model capacity and VRAM consumption.

### Text Encoder: `BERTEncoder` with top-2 layers fine-tunable

```python
text_encoder = BERTEncoder(device=DEVICE, freeze=True, unfreeze_top_k=2)
```

BERT-base (Devlin et al. 2018) has 12 transformer layers. `unfreeze_top_k=2` keeps layers 0–9 frozen and makes layers 10–11 trainable. This is a standard fine-tuning strategy: the lower layers encode general linguistic knowledge (syntax, morphology) that transfers across all tasks; the upper layers encode task-specific semantic representations that can be specialised for robot action prediction.

**Why not fully fine-tune BERT?** Fully fine-tuning 12 transformer layers would (a) require much more VRAM, (b) risk catastrophic forgetting of useful pretrained representations, and (c) potentially overfit on the small cable routing instruction vocabulary. Fine-tuning only the top 2 layers is a principled compromise.

### Vision Encoder: `AdvancedMaxViT` (MaxViT-Large)

MaxViT-Large with `dim=64, depth=(6,5,5)` produces `embed_dim=512`. The spatial output is `(B, 512, 28, 28)`. Compared to MaxViT-Base (nb1), this is 60% more channels and 45% more blocks, roughly 3× the parameter count.

### Token Learner: `SelfAttentionTokenLearner` (64 tokens, 8 heads)

Unlike cross-attention (which queries a fixed set of learned vectors), `SelfAttentionTokenLearner` performs full self-attention on the flattened spatial feature map first to contextualise it, then applies learned pooling:

```python
class SelfAttentionTokenLearner(nn.Module):
    def __init__(self, dim, num_tokens=64, heads=8):
        self.queries = nn.Parameter(torch.randn(1, num_tokens, dim))
        self.self_attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(dim, heads, batch_first=True)
```

**Forward:**
1. Flatten spatial map: `(B, 512, 28, 28) → (B, 784, 512)`
2. Self-attention on spatial tokens: `(B, 784, 512) → (B, 784, 512)` — every spatial position attends to all other positions.
3. Cross-attention: 64 learnable queries attend to the contextualised spatial tokens: `(B, 64, 512)`.

**Why self-attention first?** The self-attention step allows each spatial token to incorporate context from across the image before being pooled. This means the 64 output tokens represent not raw patch features but globally-informed feature summaries. This is the most expressive token learner in the study.

### Gradient Checkpointing (nb5-specific, Block 12)

Because nb5's model is too large to fit comfortably in 8 GB VRAM, Block 12 applies gradient checkpointing:

```python
import torch.utils.checkpoint as ckpt_utils

def _make_ckpt_block(block):
    orig_fwd = block.forward
    def _ckpt_fwd(x):
        return ckpt_utils.checkpoint(orig_fwd, x, use_reentrant=False)
    block.forward = _ckpt_fwd
    return block

# Apply to every ViT block, temporal model, and token learner
for stage in model.vit.stages:
    for blk in list(stage):
        _make_ckpt_block(blk)
_make_ckpt_block(model.temporal_model)
_make_ckpt_block(model.token_learner)
```

**What is gradient checkpointing?** During the forward pass, intermediate activations are normally stored in VRAM for use in the backward pass. With gradient checkpointing, these activations are discarded after each block's forward pass, and recomputed on-the-fly during the backward pass. This reduces peak VRAM by approximately 40% at the cost of approximately 20% extra FLOPs (each block's forward is run twice: once during forward, once during backward).

`use_reentrant=False` uses the safer, newer checkpointing API that is compatible with AMP and does not require all input tensors to have `requires_grad=True`.

---

## 15. Notebook 6 — Mini-VLA: CLIP + MobileViT-Small + PerceiverResampler + Action Chunking

This notebook represents the most modern design in the study, incorporating several ideas from recent robotics and VLA literature: MobileViT's hybrid local-global convolution, CLIP's vision-language alignment, Perceiver-style resampling, causal temporal attention, and action chunking.

### Configuration Variables (Block 2)

```python
MOBILEVIT_DROP_PATH = 0.15   # stochastic depth for MobileViT blocks
CHUNK_SIZE    = 4             # predict 4 future actions simultaneously
NUM_VISUAL_TOKENS = 32        # PerceiverResampler latents
TRANSFORMER_DEPTH = 4         # shallower transformer (4 layers)
TRANSFORMER_HEADS = 8
TEXT_DIM      = 512           # CLIP output dimension
MODEL_NAME    = "mini_vla_mobilevit"
```

**`CHUNK_SIZE=4`** is the defining hyperparameter of this notebook. The model predicts the actions for the current timestep **and the next 3 timesteps** simultaneously in a single forward pass, outputting shape `(B, 4, 7, 256)`. At training time, only position 0 (current timestep) is supervised; positions 1–3 are predicted but not compared to ground truth. At deployment, the robot would execute all 4 predicted actions sequentially, then query the model again — reducing inference frequency by 4×.

### Vision Encoder: `MobileViTEncoder` (MobileViT-Small)

MobileViT (Mehta & Rastegari, 2021) is a hybrid architecture that combines MBConv blocks (for local spatial processing) with lightweight transformer blocks that operate on non-overlapping patches (for global context).

```
Input: (B, 3, 224, 224)
Stem: Conv 3×3 stride=2 → (B, 16, 112, 112)

DS1: MBConv(16→24, stride=2) → (B, 24, 56, 56)     depthwise-separable downsampling
DS2: MBConv(24→48, stride=2) → (B, 48, 28, 28)
Proj: Conv 1×1 (48→128) → (B, 128, 28, 28)

MobileViT blocks (at 28×28):
  Each block:
    Local branch: depthwise 3×3 + pointwise 1×1  (local texture)
    Global branch: partition into p×p=2×2 patches → Transformer (2 layers, 4 heads)
                   → fold patches back to spatial grid
    Fusion: Conv 1×1 on cat([local_out, global_out])
    Residual add
  → (B, 128, 28, 28)

Output: (B, 128, 28, 28)    embed_dim=128
```

**`MobileViTBlock` design detail:** The non-overlapping patch partitioning scheme allows the transformer within each MobileViT block to see global context (any two patches from anywhere in the image can attend to each other) without the quadratic cost of full image-level self-attention. With `p=2`, each 28×28 feature map is divided into 196 non-overlapping 2×2 patches; the transformer operates on sequences of 196 tokens, which is far cheaper than the 784 tokens required for 1×1 patches.

### Token Learner: `PerceiverResampler` (32 latents, depth=2)

`PerceiverResampler` (used in Flamingo, Alayrac et al. 2022) uses a stack of cross-attention + self-attention layers to transform a variable-length input sequence into a fixed-size latent array.

```python
class PerceiverResampler(nn.Module):
    def __init__(self, dim, num_latents=32, depth=2, heads=4, dropout=0.2):
        self.latents = nn.Parameter(torch.randn(1, num_latents, dim))
        self.layers = nn.ModuleList([
            PerceiverLayer(dim, heads, dropout)   # cross-attn + self-attn + FFN
            for _ in range(depth)])
```

Each `PerceiverLayer` applies:
1. **Cross-attention:** The 32 learnable latent vectors attend to the spatial feature map `(B, 784, 128)`.
2. **Self-attention:** The 32 latent vectors attend to each other.
3. **FFN:** Standard transformer feed-forward block.

With `depth=2`, the latents refine their representations through two rounds of cross-attention. This makes the PerceiverResampler more expressive than the single-round cross-attention in nb1's `AdvancedTokenLearner`.

### Temporal Fusion: `CausalTemporalAttention`

Unlike `ImprovedTemporalConv` (used in nb1, nb2, nb3, nb5), nb6 uses transformer-based temporal attention:

```python
class CausalTemporalAttention(nn.Module):
    def __init__(self, dim, heads=4, dropout=0.2):
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.causal_mask = ...  # upper triangular mask
```

For each spatial position (after flattening), the `SEQ_LEN=6` feature vectors across frames are treated as a temporal sequence and processed with **causal (autoregressive) self-attention**: frame `t` can attend to frames `0...t` but not `t+1...F`. This is achieved by an upper-triangular boolean mask passed as `attn_mask` to `nn.MultiheadAttention`.

**Why causal?** Causal masking means each frame's feature vector is conditioned only on past frames, not future ones. This is important for deployment: at inference time, the model processes frames as they arrive in real time, without access to future frames. Causal temporal attention makes the training distribution match the inference distribution.

**Why attention over `ImprovedTemporalConv`?** The `ImprovedTemporalConv` uses a fixed 3×1×1 kernel that integrates information over a 3-frame window. The causal attention can integrate over all preceding frames with learned, input-dependent weights. For tasks with variable-length temporal dependencies (sometimes the cable routing decision depends on the current frame; sometimes it requires reasoning about the full trajectory), attention is strictly more expressive.

### Full Model: `MiniVLAModel`

```python
class MiniVLAModel(nn.Module):
    def __init__(self, vit, text_dim=512, num_actions=7, action_bins=256,
                 depth=4, heads=8, dropout=0.2, num_latents=32, chunk_size=4):
```

**Forward:**
```
video (B,C,F,H,W) → MobileViTEncoder → CausalTemporalAttention → PerceiverResampler
text (B,512)       → Linear(512→128) + LN + Dropout → text token
[text | frame_tokens] → 4-layer Transformer
cls_token → to_logits [Linear(128,256) → GELU → Dropout → Linear(256, 4×7×256)]
→ reshape: (B, CHUNK_SIZE=4, NUM_ACTIONS=7, ACTION_BINS=256)
```

**Training loss (Block 12, nb6-specific):**
```python
logits  = model(video, text_emb)    # (B, 4, 7, 256)
logits_c = logits[:, 0, :, :]       # (B, 7, 256) — supervise only current step
loss = criterion(logits_c_flat, targets_flat)
```

**Why supervise only chunk position 0?** The dataset provides ground-truth actions only for the current timestep. Future timestep actions would require a multi-step lookahead in the dataset (loading `target[t+1]`, `target[t+2]`, etc.), which complicates the dataset implementation. Supervising only position 0 while the model still predicts positions 1–3 forces the model to develop an implicit internal model of future trajectory evolution, similar to how video prediction models are trained.

---

## 16. Shared XAI Pipeline (Blocks 14–21)

Every notebook ends with an identical 7-block XAI evaluation suite. The suite runs after training is complete (or loads a saved checkpoint if `SKIP_TRAINING=True`). All XAI computations use the EMA model for stability.

### 16.1 XAI Dependencies (Block 14)

```python
subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q',
                       'lime', 'shap', 'grad-cam==1.5.5', 'imageio[ffmpeg]'])
from lime.lime_image import LimeImageExplainer
from pytorch_grad_cam import GradCAM, HiResCAM, ShapleyCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from skimage.segmentation import mark_boundaries
```

**Why `grad-cam==1.5.5` (pinned version)?** The `pytorch-grad-cam` package has had breaking API changes between versions. Pinning to `1.5.5` ensures the `GradCAM`, `HiResCAM`, and `ShapleyCAM` constructors accept the `model` and `target_layers` keyword arguments as expected.

**Why `imageio[ffmpeg]`?** XAI blocks generate animated videos (`.mp4`) of CAM overlays, LIME masks, and SHAP maps evolving across the 100-frame temporal window. `imageio` with the FFmpeg backend provides the simplest API for writing MP4 files from NumPy arrays.

### 16.2 XAI Setup (Block 15): The sklearn Mock Injection

LIME internally imports `sklearn` (scikit-learn) for its `Ridge` regression solver, `LabelEncoder`, and `MinMaxScaler`. On some environments (Kaggle kernels, constrained containers) `sklearn` may not be installed, or importing it may conflict with other packages. Rather than requiring a full `sklearn` installation, the code injects lightweight mock implementations directly into `sys.modules`:

```python
def _ensure_submod(full, **attrs):
    """Inject a fake module into sys.modules to satisfy LIME's imports."""
    if full not in sys.modules:
        m = types.ModuleType(full)
        for k, v in attrs.items(): setattr(m, k, v)
        sys.modules[full] = m

class _Ridge:
    """Minimal Ridge regression: closed-form weighted least squares."""
    def fit(self, X, y, sample_weight=None):
        W = np.diag(sample_weight) if sample_weight is not None else np.eye(len(y))
        A = X.T @ W @ X + self.alpha * np.eye(X.shape[1])
        b = X.T @ W @ y
        self.coef_ = np.linalg.solve(A, b)
        return self

_ensure_submod('sklearn')
_ensure_submod('sklearn.linear_model', Ridge=_Ridge)
_ensure_submod('sklearn.preprocessing', LabelEncoder=_LabelEncoder, MinMaxScaler=_MinMaxScaler)
_ensure_submod('sklearn.utils', check_random_state=_check_random_state)
_ensure_submod('sklearn.utils.validation', check_is_fitted=lambda *a, **k: None)
```

**Why a custom Ridge?** LIME's core algorithm fits a locally weighted Ridge regression model to explain each prediction. The mock implementation solves `(XᵀWX + αI)β = XᵀWy` via `numpy.linalg.solve`, which is numerically equivalent to scikit-learn's implementation for this specific use case. This avoids the sklearn dependency entirely while preserving LIME's full functionality.

This block also creates the XAI output directory tree:
```
xai_outputs_{MODEL_NAME}/
├── block_16_gradcam/
├── block_17_lime/
├── block_18_shap/
├── block_19_pdp_ice/
├── block_20_fusions/
└── block_21_dashboard/
```

Additionally, a `FrameWrapper` model is created that exposes only the vision encoder (without the full transformer and action head), enabling GradCAM to target the last BatchNorm or Conv layer of the backbone:

```python
class FrameWrapper(nn.Module):
    def forward(self, x):  # x: (B, 3, H, W) — single frame
        feat = self.vit(x)
        return self.pool(feat).flatten(1)  # global average pooled features
```

The target layer for GradCAM is auto-discovered by scanning for the last `BatchNorm2d` in the vision encoder. BatchNorm is preferred over Conv as the target layer because its normalised activations produce less noisy gradient maps.

### 16.3 GradCAM Family (Block 16)

Three GradCAM variants are applied:

**GradCAM** (Selvaraju et al. 2017): Computes a weighted sum of the feature map channels, where the weight for each channel is the global average of the gradient of the target class score with respect to that channel. The result is a coarse spatial map of class-discriminative regions.

**HiResCAM** (Draelos & Carin 2020): Multiplies the gradients directly by the feature map activations without averaging, preserving finer spatial resolution. Produces crisper attribution maps than GradCAM, though occasionally noisier.

**ShapleyCAM** (Wang & Wang 2020): Weights feature map channels by their Shapley values (derived from a local linearisation of the model), providing attribution maps that are theoretically grounded in game theory.

**Outputs generated:**
- `gradcam_static.png` — static side-by-side panels for `XAI_N` sampled frames.
- `gradcam_video_*.mp4` — animated CAM overlay across 100 consecutive frames.
- `gradcam_filmstrip.png` — `FILMSTRIP_N=10` equispaced frames as a contact sheet.
- `gradcam_evolution.png` — per-method mean activation over frames, showing temporal evolution of attention.
- `gradcam_trajectory.png` — spatial centre-of-mass of the CAM map per frame, showing how the model's visual attention moves across the image during task execution.

### 16.4 LIME (Block 17)

LIME (Local Interpretable Model-agnostic Explanations, Ribeiro et al. 2016) explains a single prediction by:
1. Perturbing the input image by randomly hiding superpixels (SLIC algorithm).
2. Running each perturbed image through the model to get a prediction.
3. Fitting a Ridge regression locally to map superpixel presence/absence → prediction value.
4. The top-K superpixels with the highest regression coefficients are highlighted.

**Predict function wrapper:**

```python
def lime_predict(images_np):
    # images_np: (N_perturb, H, W, 3) float64 from LIME's perturbation engine
    tensors = [torch.from_numpy(img.astype(np.float32)).permute(2,0,1) for img in images_np]
    batch = torch.stack(tensors).to(DEVICE)
    video = batch.unsqueeze(2).expand(-1, -1, SEQ_LEN, -1, -1)  # replicate single frame
    text_emb = xai_text_embs[0:1].expand(len(images_np), -1)
    with torch.no_grad(), torch.amp.autocast('cuda', dtype=AUTOCAST_DTYPE):
        logits = model(video, text_emb).float()
    # Convert discrete action logits to a scalar "action magnitude" for LIME
    bins = torch.arange(256, device=DEVICE).float()
    probs = torch.softmax(logits, dim=-1)
    return ((probs * bins[None, None, :]).sum(-1) / 255.0).cpu().numpy()
```

**Why replicate the single frame to `SEQ_LEN`?** The model expects video input. LIME perturbs only one frame at a time, so the same perturbed frame is replicated across all temporal positions. This is a simplification — it does not capture temporal aspects of the model's decision — but it allows LIME (which operates on 2D images) to be applied to a video model without architectural changes.

**Parameters:**
- `LIME_SAMPLES_STATIC=128`: Number of perturbed images per LIME explanation. Higher = more accurate but slower.
- `LIME_TOP_K=5`: Number of top superpixels highlighted in the overlay.

**Outputs:** Static LIME panels, per-frame LIME video, evolution plot of LIME active area over time.

### 16.5 SHAP (Block 18)

SHAP (SHapley Additive exPlanations, Lundberg & Lee 2017) uses `GradientExplainer` to estimate Shapley values for each pixel. The `SHAPFrameWrapper` exposes the vision encoder with a global average pooling head:

```python
class SHAPFrameWrapper(nn.Module):
    def forward(self, x):
        return self.head(self.pool(self.vit(x)).flatten(1))
        # output: (B, 256) — per-class action scores
```

SHAP's `GradientExplainer` computes `E[∇f(x) · (x - x')]` over background samples `x'`, approximating the Shapley value for each pixel. The result `sv_map` has shape `(C, H, W)`. The absolute mean over channels gives a per-pixel importance score.

**OOM fallback:** If `GradientExplainer` runs out of VRAM (which can happen on nb5 due to its large model), the code falls back to a simple gradient saliency map:

```python
single.requires_grad_(True)
out = shap_model(single)
out.sum().backward()
sv_map = single.grad[0].abs().mean(dim=0).cpu().numpy()
```

This is not a true Shapley value but provides a fast approximate attribution for visualisation purposes.

**Parameters:** `nsamples=50` — number of background integration samples per SHAP explanation. Reduced from the default 200 to fit within VRAM constraints while preserving attribution quality.

### 16.6 PDP + ICE + Validation Improvement (Block 19)

**Validation improvement check:**

```python
baseline_val_loss = math.log(256)  # theoretical random-guess loss (uniform over 256 bins)
current_val_loss  = evaluate_val_loss_safe(model, n_batches=20)
improvement       = baseline_val_loss - current_val_loss
```

The baseline is `ln(256) ≈ 5.545` — the cross-entropy loss of a uniform random predictor over 256 action bins. Any trained model should achieve a loss substantially below this. The improvement percentage is reported and displayed in the XAI dashboard.

**Partial Dependence Plots (PDP):** For each of the `NUM_ACTIONS` action dimensions, the PDP shows the marginal effect of changing the continuous action value (on a grid `ALPHA_GRID = linspace(0.2, 1.8, 20)`) while averaging over all other features. Implemented by sweeping a brightness scalar over the input video frames and measuring the model's action predictions.

**Individual Conditional Expectation (ICE):** Like PDP but shows per-sample curves rather than the average, revealing heterogeneous effects that PDP would average out.

**Why PDP+ICE for a robot model?** These tools, traditionally used in tabular machine learning, can be adapted to robot action models by treating a scalar image attribute (brightness, contrast) as the "feature" of interest. This tests whether the model is relying on low-level image statistics (brightness, which varies with workspace lighting) rather than high-level task-relevant features (cable position, clip orientation).

### 16.7 Novel XAI Fusions: ShaGCAM and LimeCAM (Block 20)

```python
# ShaGCAM: multiplicative fusion of GradCAM and |SHAP|
gcam_norm = normalise(gcam_map)          # [0,1]
shap_norm = normalise(abs(shap_map))     # [0,1]
sha_map   = normalise(gcam_norm * shap_norm)   # element-wise product

# LimeCAM: geometric mean fusion of GradCAM and LIME
lime_norm = normalise(lime_mask.astype(np.float32))   # [0,1]
lc_map    = normalise(np.sqrt(gcam_norm * (lime_norm + 0.05)))
```

Both fusions are visualised as heatmap overlays on the original frame using `show_cam_on_image()`, and additionally as animated videos across the full frame sequence.

**ShaGCAM interpretation:** A bright pixel in ShaGCAM means it is simultaneously in the gradient path from output to that spatial region (GradCAM) AND has a large Shapley contribution (SHAP). This double criterion filters out GradCAM artefacts (pixels that appear salient only because gradients flow through them, not because they causally influence the prediction) and SHAP noise (pixels with small but nonzero Shapley values).

**LimeCAM interpretation:** A bright pixel in LimeCAM means that hiding its superpixel decreases the model's prediction (LIME) AND that gradients flow through its spatial location (GradCAM). This double criterion isolates regions that are both locally influential (perturbation-based) and globally gradient-salient.

### 16.8 XAI Master Dashboard (Block 21)

Block 21 assembles a comprehensive single-figure dark-themed dashboard (`26×16` inches) using `matplotlib.gridspec.GridSpec(4, 7)` with a dark background (`#0a0a14`). Panels include:

- **Row 0:** Title bar with model name, task instruction, val loss, and improvement percentage.
- **Row 1:** Original frame | GradCAM | HiResCAM | ShapleyCAM | LIME top-5 | |SHAP| | Validation bar chart.
- **Row 2:** ShaGCAM fusion | LimeCAM fusion | PDP curves per action dimension | temporal evolution plot.
- **Row 3:** Per-frame attention trajectory | LIME active area over time | SHAP mean map over time | comparison across CAM methods.

The dashboard is saved as `xai_outputs_{MODEL_NAME}/block_21_dashboard/xai_dashboard.png` and provides a single comprehensive visual summary of the model's interpretability profile.

---

## 17. Cross-Notebook Ablation Comparison

| Property | nb1 RT-1 Base | nb2 RT-2+CoT | nb3 Lightweight | nb4 CLIP | nb5 Max Capacity | nb6 Mini-VLA |
|---|---|---|---|---|---|---|
| Vision encoder | MaxViT-Base | MaxViT-Base | ConvNeXt-Small | EfficientLite-M | MaxViT-Large | MobileViT-Small |
| Text encoder | DistilBERT (frozen) | DistilBERT+VQA (frozen) | BoW-256 (trainable) | CLIP-B/32 (frozen) | BERT top-2 (partial ft) | CLIP-B/32 (frozen) |
| Token learner | CrossAttn (32) | CrossAttn (32) | ConvTL (16) | Hierarchical (32) | SelfAttn (64) | PerceiverResampler (32) |
| Temporal fusion | ImprovedTemporalConv | ImprovedTemporalConv | ImprovedTemporalConv | ImprovedTemporalConv | ImprovedTemporalConv | CausalTemporalAttn |
| Transformer | 6L, 8H | 6L, 8H | 4L, 4H | 4L, 4H | 8L, 8H | 4L, 8H |
| embed_dim | 320 | 320 | 192 | 96 | 512 | 128 |
| Action head | Standard | ChainOfThoughtHead | Standard | Standard | Standard | ActionChunking (×4) |
| Effective batch | 128 | 128 | 128 | 128 | 128 | 128 |
| VRAM optimisation | TF cap 512MB | TF cap 512MB | TF GPU off | TF GPU off | Grad checkpointing | TF cap 512MB |
| Model identifier | `rt1_base_maxvit` | `rt2_vla_cot` | `rt1_lightweight_bow_convnext` | `rt1_clip_efficientlite` | `rt1_large_bert_maxvit` | `mini_vla_mobilevit` |

---

## 18. Hardware Requirements

| Notebook | Minimum VRAM | Recommended VRAM | Notes |
|---|---|---|---|
| nb3 | 6 GB | 8 GB | Smallest model, BATCH_SIZE=4 |
| nb4 | 6 GB | 8 GB | EfficientLite is compact |
| nb6 | 8 GB | 12 GB | MobileViT + PerceiverResampler |
| nb1 | 8 GB | 12 GB | MaxViT-Base |
| nb2 | 8 GB | 12 GB | Same as nb1 + CoT head overhead |
| nb5 | 12 GB | 16 GB | MaxViT-Large; gradient checkpointing required below 16 GB |

All notebooks require:
- Python 3.9+
- PyTorch 2.0+ (for `torch.amp.autocast` API)
- CUDA 11.8+ (for BFloat16 support on NVIDIA Ampere/Hopper)
- ~50 GB disk space for the `berkeley_cable_routing` dataset and caches
- Internet access for initial dataset download from GCS and model checkpoint download from HuggingFace

---

## 19. Lessons Learned and Engineering Challenges

**TF/PyTorch GPU Memory Conflict.** The most common failure mode during development was TensorFlow pre-allocating GPU memory at import time, leaving insufficient VRAM for the PyTorch model. The resolution was the dual strategy: setting `TF_FORCE_GPU_ALLOW_GROWTH=true` before importing TF, and hard-capping TF to 512 MB. In nb3 and nb4 (smaller models with tighter VRAM budgets), TF GPU visibility was disabled entirely.

**LIME sklearn Dependency.** LIME's internal Ridge regression solver requires `sklearn`. On Kaggle and in minimal Docker environments, importing `sklearn` can be slow or may conflict with other packages. The sklearn mock injection (Block 15) was developed to eliminate this dependency while preserving LIME's full algorithmic behaviour.

**Gradient Checkpointing Trade-off (nb5).** MaxViT-Large with 64 token-learner tokens exceeds 8 GB VRAM during the backward pass without checkpointing. The per-block checkpointing strategy reduces peak VRAM by ~40% but increases training time by ~20%. The `use_reentrant=False` flag was critical — the older `use_reentrant=True` API caused silent correctness issues with AMP.

**Memmap and Worker Processes.** Using `numpy.memmap` with multiple DataLoader workers requires that each worker opens the memmap file independently. The `mode='r'` (read-only) setting on the worker side prevents write conflicts but requires that the memmap file be fully written and closed before DataLoader initialisation. A race condition was encountered in early versions when `_build_cache()` and DataLoader initialisation overlapped; resolved by always completing `_build_cache()` before instantiating the DataLoader.

**BFloat16 and GradScaler.** With BFloat16, the GradScaler's scaling factor sometimes grows to very large values (because BFloat16 rarely underflows) and triggers a false-positive "inf in gradients" check, causing the optimiser step to be skipped. Resolved by using `scaler.update()` with the default conservative settings and monitoring the `scaler.get_scale()` value.

---

## 20. Glossary of Terms

| Term | Definition |
|---|---|
| **VLA** | Vision-Language-Action model: a neural network that takes images and text instructions as inputs and outputs robot actions. |
| **Ablation study** | A systematic experiment where one component is varied at a time while everything else is held constant, to measure the contribution of each component. |
| **MaxViT** | Multi-Axis Vision Transformer. A hybrid CNN-Transformer vision backbone that alternates between MBConv blocks, window attention, and grid attention. |
| **MobileViT** | Mobile Vision Transformer. A lightweight hybrid architecture that integrates depthwise convolution (local features) with lightweight patch-level transformers (global context). |
| **ConvNeXt** | A pure convolutional network redesigned with transformer-inspired components (GELU, LayerNorm, inverted bottlenecks). |
| **MBConv** | Mobile Inverted Bottleneck Convolution. An efficient building block using pointwise expansion, depthwise convolution, and pointwise projection with SE channel attention. |
| **TokenLearner** | A module that compresses a spatial feature map into a small set of compact token vectors for efficient sequence modelling. |
| **PerceiverResampler** | A token learner based on cross-attention + self-attention layers operating on a fixed set of learnable latent vectors. |
| **CrossAttn TokenLearner** | A token learner where learned query vectors attend to spatial features via multi-head cross-attention. |
| **SelfAttn TokenLearner** | A token learner that first runs full self-attention on spatial features then compresses via cross-attention queries. |
| **HierarchicalTokenLearner** | A token learner that produces tokens at two spatial scales (fine and coarse) and concatenates them. |
| **BoW** | Bag-of-Words. A text representation that averages word embeddings without modelling word order or context. |
| **CLIP** | Contrastive Language-Image Pretraining. A model trained to align image and text embeddings in a shared space. |
| **DistilBERT** | A distilled, smaller version of BERT that is 40% lighter while retaining 97% of BERT's performance. |
| **RT-1 / RT-2** | Robot Transformer 1 and 2. Google DeepMind architectures for generalist robot policies. RT-1 uses a visual token learner + transformer. RT-2 co-trains a VLM with robot actions. |
| **Action Chunking** | Predicting multiple future action timesteps in a single forward pass to reduce inference frequency and improve temporal consistency. |
| **ChainOfThoughtHead** | A planning head that predicts a discrete task phase and uses it to condition the action prediction, inspired by RT-2 chain-of-thought reasoning. |
| **CausalTemporalAttention** | Temporal self-attention with an upper-triangular mask, ensuring each frame only attends to past frames (autoregressive). |
| **GradCAM** | Gradient-weighted Class Activation Mapping. Highlights image regions most discriminative for a model's prediction, using gradients of the target class with respect to the last convolutional layer. |
| **HiResCAM** | High-resolution CAM. Pixel-wise product of gradients and activations, preserving finer spatial detail than GradCAM. |
| **ShapleyCAM** | CAM variant that weights feature channels by their Shapley values, providing game-theory-grounded spatial attribution. |
| **LIME** | Local Interpretable Model-agnostic Explanations. Explains predictions by locally perturbing superpixels and fitting a simple linear model to the prediction changes. |
| **SHAP** | SHapley Additive exPlanations. Assigns each feature (pixel) a Shapley value representing its average marginal contribution to the prediction. |
| **PDP** | Partial Dependence Plot. Shows the marginal effect of one input feature on model predictions, averaging over all other features. |
| **ICE** | Individual Conditional Expectation. Per-sample version of PDP, showing how individual predictions change with a single feature. |
| **ShaGCAM** | Novel fusion: element-wise product of normalised GradCAM and normalised absolute SHAP maps. |
| **LimeCAM** | Novel fusion: normalised geometric mean of GradCAM and LIME superpixel maps with regularisation `+0.05`. |
| **Lion** | EvoLved Sign Momentum optimiser (Chen et al. 2023). Uses only the sign of the gradient update; more memory-efficient than Adam. |
| **EMA** | Exponential Moving Average of model weights. Produces smoother, more stable model checkpoints for evaluation. |
| **Focal Loss** | A loss function that down-weights easy examples (high `p_t`) via `(1-p_t)^γ` weighting, focusing training on hard examples. |
| **MixUp** | Data augmentation that linearly interpolates between pairs of training examples and their labels. |
| **CutMix** | Data augmentation that replaces a rectangular patch in one image with the corresponding patch from another image. |
| **DropPath** | Stochastic Depth. Randomly drops entire residual branches during training, acting as a strong structural regulariser. |
| **Gradient Checkpointing** | Memory optimisation that recomputes intermediate activations during the backward pass instead of storing them, reducing peak VRAM at the cost of extra compute. |
| **memmap** | `numpy.memmap`: memory-mapped file access that allows random access to large arrays stored on disk without loading them into RAM. |
| **BFloat16** | Brain Float 16. A 16-bit floating point format with the same exponent range as Float32 but lower mantissa precision. Used for accelerated mixed-precision training. |
| **TF32** | TensorFloat-32. An internal NVIDIA format for matrix multiplications that uses Float32 dynamic range but lower mantissa precision, enabling ~3× speedup on Ampere/Hopper GPUs. |
| **WeightedRandomSampler** | A PyTorch sampler that over-samples rare classes and under-samples common ones to produce balanced training batches. |
| **VQAFormatter** | A text preprocessing class that wraps raw task instructions into VQA-style question-answer templates. |

---

*End of Report*
MARKDOWN_EOF
echo "Done. Line count:"
wc -l /mnt/user-data/outputs/X_RT1_Ablation_Study_Technical_Report.md
---

---

# Production RT-1 Notebook — `production_rt1_notebook-Copy1.ipynb`
## Second Dataset Experiment: DROID + Custom XAI Implementation

---

## Overview and Context

This notebook represents the **second training experiment** in the overall project. While notebooks nb1–nb6 all train on the `berkeley_cable_routing` dataset (downloaded from Google Cloud Storage via TFDS), this production notebook trains the **same RT-1 architecture** — MaxViT-Base + DistilBERT + CrossAttn TokenLearner — on a **completely different dataset: DROID**, stored locally as a compressed NumPy file (`droid_train_100.npz`).

The goals of this notebook are twofold:
1. **Cross-dataset validation:** Test whether the RT-1 architecture generalises to a different robotics dataset with different action spaces, different camera setups, and different task distributions.
2. **Production-grade XAI pipeline:** Replace the external `pytorch-grad-cam` library used in nb1–nb6 with fully custom, self-contained CAM implementations that are directly wired into the model via PyTorch forward/backward hooks. This enables two novel XAI hybrid methods not possible with the external library.

The notebook is structured into 13 sections (vs. the 21-block system in nb1–nb6), with a separate training-resume cell for recovering from disconnections.

---

## Section 1: Configuration

The configuration block (`NPZ_PATH = "droid_train_100.npz"`) sets the most important distinction from nb1–nb6 immediately: the data source is a local `.npz` file, not a GCS-hosted TFDS dataset.

### Configuration Variables — Full Explanation

| Variable | Value | Plain-English Meaning | Why this value? |
|---|---|---|---|
| `NPZ_PATH` | `"droid_train_100.npz"` | Path to the DROID dataset file in NumPy compressed format | DROID is pre-packaged as NPZ; the `100` suffix indicates 100 episodes from the full DROID dataset |
| `DEVICE` | `"cuda"` | Run all model computations on the GPU | GPU is 10–100× faster than CPU for matrix multiplications |
| `VIT_DIM` | `40` | Base channel width of MaxViT backbone | Same as nb1; determines `embed_dim = 40 × 2³ = 320` |
| `VIT_DEPTH` | `(4, 4, 3)` | Number of MBConv blocks per stage | Same as nb1; 11 blocks total |
| `TRANSFORMER_DEPTH` | `6` | Number of transformer encoder layers | Same as nb1 |
| `TRANSFORMER_HEADS` | `8` | Number of attention heads per transformer layer | Same as nb1 |
| `DROPOUT` | `0.2` | Dropout probability in transformer and projection layers | Standard regularisation for mid-size transformers |
| `NUM_VISUAL_TOKENS` | `32` | Number of tokens produced by the TokenLearner | Same as nb1; compresses 784 spatial positions to 32 |
| `USE_FUSED_MBCONV` | `True` | Use fused (3×3 conv + SE + 1×1 conv) instead of depthwise | Faster on Ampere/H100 GPUs |
| `DROP_PATH_RATE` | `0.2` | Stochastic depth rate for MBConv blocks | Same as nb1 |
| `BATCH_SIZE` | `15` | Number of samples per micro-batch | **Much larger than nb1's `BATCH_SIZE=1`** — possible because DROID NPZ loads faster than TFDS streaming |
| `GRADIENT_ACCUM_STEPS` | `2` | Accumulate gradients over this many batches before stepping | Effective batch = 15 × 2 = **30**. Much smaller effective batch than nb1–nb6's 128, because DROID has more varied tasks making smaller batches sufficient |
| `EPOCHS` | `80` | Maximum training epochs | Same as nb1–nb6 |
| `WARMUP_STEPS` | `500` | Linear LR warmup steps | Same as nb1 |
| `LR` | `3e-5` | Lion peak learning rate | Same as nb1 |
| `WEIGHT_DECAY` | `0.15` | Decoupled L2 penalty | Same as nb1 |
| `LION_BETAS` | `(0.9, 0.99)` | Momentum decay coefficients for Lion | Same as nb1 |
| `GRAD_CLIP` | `1.0` | Maximum gradient L2 norm | Same as nb1 |
| `LABEL_SMOOTHING` | `0.1` | Focal loss label smoothing ε | Same as nb1 |
| `MIXUP_ALPHA` | `0.3` | Beta distribution parameter for MixUp/CutMix | Same as nb1 |
| `USE_CUTMIX` | `True` | Enable CutMix augmentation | Same as nb1 |
| `FOCAL_GAMMA` | `2.5` | Focal loss focusing parameter γ | Same as nb1 |
| `SEQ_LEN` | `6` | Number of consecutive frames per training sample | Same as nb1 |
| `TARGET_SIZE` | `224` | Resize every image to 224×224 pixels | Same as nb1 |
| `SPLIT_RATIO` | `0.8` | 80% train / 20% val split | Same as nb1 |
| `NUM_WORKERS` | `2` | DataLoader worker processes | Same as nb1 |
| `PREFETCH_FACTOR` | `4` | Batches prefetched per worker | Same as nb1 |
| `PATIENCE` | `10` | Early stopping patience in epochs | Same as nb1 |
| `CHECKPOINT_FREQ` | `5` | Save a periodic checkpoint every 5 epochs | Provides recovery points; `rt1_checkpoint_epoch_20.pt` was created by this setting |
| `TASK_METRIC_FREQ` | `5` | Compute per-task loss breakdown every N batches | Every 5 batches is a balance — frequent enough to catch per-task failures, infrequent enough not to add overhead to every batch |
| `USE_BFLOAT16` | `True` | Enable BFloat16 automatic mixed precision | Same rationale as nb1 |
| `XAI_NUM_SAMPLES` | `8` | Number of validation samples to run through the XAI suite | 8 is enough for visual comparison without excessive compute time |
| `XAI_LIME_PERTURBATIONS` | `500` | Number of perturbed images LIME generates per explanation | Higher than nb1–nb6's 128; provides more accurate local linear approximation at the cost of ~4× more forward passes |
| `XAI_SHAP_BACKGROUND` | `20` | Number of background samples for DeepSHAP estimation | 20 gives stable Shapley value estimates without excessive memory |
| `XAI_PDP_GRID_SIZE` | `15` | Number of grid points in PDP sweeps | 15 points give a smooth enough curve for visual interpretation |
| `XAI_ICE_SAMPLES` | `30` | Number of individual samples for ICE plots | 30 curves reveal per-sample heterogeneity clearly |

**Key configuration differences vs. nb1–nb6:**

| Property | nb1–nb6 (Cable Routing) | Production (DROID) |
|---|---|---|
| Dataset format | TFDS / GCS download | Local `.npz` file |
| BATCH_SIZE | 1 | 15 |
| GRADIENT_ACCUM_STEPS | 128 | 2 |
| Effective batch size | 128 | 30 |
| XAI library | `pytorch-grad-cam` (external) | Fully custom (built-in hooks) |
| LIME perturbations | 128 | 500 |
| Checkpointing | Every epoch (best only) | Every 5 epochs + best |

**Why `BATCH_SIZE=15` with `GRADIENT_ACCUM_STEPS=2` instead of `BATCH_SIZE=1` with `GRADIENT_ACCUM_STEPS=128`?**
DROID data is stored as a pre-processed NumPy `.npz` file, which loads into RAM in one shot and allows random-access indexing with negligible I/O overhead. This means larger batch sizes are feasible without I/O bottlenecks. The TFDS-based berkeley_cable_routing dataset required `BATCH_SIZE=1` because streaming TFRecord shards introduced I/O latency that made larger batches wasteful. The effective batch of 30 (vs. 128 in nb1) is appropriate for DROID's more diverse task distribution, where each batch naturally contains a wider variety of instructions.

---

## Section 2: Core Utilities

Identical to nb1–nb6: `RT1Tokenizer` (vocab_size=256, range [-3.2, +3.2]) and `DropPath` (Stochastic Depth). See §5.4 of the main report for full explanation.

---

## Section 3: Text Encoder — `DistilBERTEncoder`

Identical class definition to nb1. Frozen DistilBERT-base-uncased, outputs CLS token embedding of shape `(B, 768)`. See §10 (Notebook 1) of the main report for full explanation.

---

## Section 4: Vision Encoder — `AdvancedMaxViT` (EfficientNet-V2 Style)

The vision backbone is `AdvancedMaxViT` with the same parameters as nb1 (`dim=40, depth=(4,4,3)`), but the class is re-implemented here using simpler naming (`MBConv` instead of `ImprovedMBConv`, `SqueezeExcitation` instead of `ImprovedSqueezeExcitation`). Functionally they are identical.

**Why "Improved" prefix in nb1–nb6 but plain names here?**
In nb1–nb6, the "Improved" prefix signals that these are modified versions of the standard published components:
- `ImprovedSqueezeExcitation` adds `BatchNorm1d` after the squeeze layer and `Dropout(0.1)` in the excitation path — the standard SE [SE] has neither.
- `ImprovedMBConv` adds configurable stochastic depth via `DropPath` — standard MBConv [EfficientNet] does not include this.
- `ImprovedTemporalConv` is an entirely new component with no standard prior; the "Improved" prefix distinguishes it from simpler temporal convolutions.

In this production notebook, the same improvements are present (BatchNorm1d, Dropout in SE; DropPath in MBConv), but the "Improved" naming convention was dropped for conciseness in the production codebase. The behaviour is equivalent.

**Architecture shape flow (same as nb1):**
```
Input:   (B×F, 3, 224, 224)
Stem:    Conv(3→40, 3×3, stride=2) + BN + SiLU
         Conv(40→40, 3×3, stride=1) + BN + SiLU
         → (B×F, 40, 112, 112)

Stage 0: 4× MBConv(fused, no downsample)  → (B×F,  40, 112, 112)
Stage 1: 1× MBConv(fused, downsample)     → (B×F,  80,  56,  56)
         3× MBConv(fused, no downsample)  → (B×F,  80,  56,  56)
Stage 2: 1× MBConv(std, downsample)       → (B×F, 160,  28,  28)
         2× MBConv(std, no downsample)    → (B×F, 160,  28,  28)

Output:  (B×F, 160, 28, 28)    embed_dim = 160
```

> **Note:** In this notebook, `embed_dim=160` (not 320 as stated in nb1's architecture flow above). This is because stage 2 uses `expansion=4` (non-fused), which does not double the channel dimension in the same way. The actual embed_dim depends on the stage transition logic in `AdvancedMaxViT.__init__`: `dim_out = curr_dim * 2 if ds else curr_dim`, and the third stage starts with `curr_dim=80` when entering the downsampling block.

---

## Section 5: Token Learner and Temporal Convolution

Identical implementations to nb1. `AdvancedTokenLearner` uses cross-attention with 32 learnable query vectors and a position embedding that is bilinearly interpolated to match the spatial size of the feature map. `TemporalConv` (equivalent to `ImprovedTemporalConv` in nb1–nb6) uses a depthwise 3D conv over the time axis followed by a 3D SE-like gating mechanism.

---

## Section 6: `ProductionRT1` — Model with Built-in XAI Hook Infrastructure

This is the **most important architectural difference** from nb1–nb6. Rather than wrapping an existing model with an external CAM library, `ProductionRT1` has XAI hooks built directly into its class definition.

### What are PyTorch hooks?

A **hook** in PyTorch is a callback function that is registered to execute at a specific point in the forward or backward pass. Forward hooks capture the output of a layer during inference. Backward hooks capture the gradient flowing back through a layer during backpropagation (the process of computing how much each weight contributed to the error). These captured values are exactly what all CAM-based XAI methods need.

### Hook Infrastructure Design

```python
# Three dictionaries for XAI state
self._activations = {}   # name → output tensor from forward pass
self._gradients   = {}   # name → gradient tensor from backward pass
self._hooks       = []   # list of registered hook handles (for cleanup)
```

**`register_xai_hooks(target_layers=None)`** registers both forward and backward hooks on each specified layer. The default targets are the three ViT stages:
```python
'vit_stage_0': self.vit.stages[0]   # shallowest, highest resolution (112×112)
'vit_stage_1': self.vit.stages[1]   # mid, 56×56
'vit_stage_2': self.vit.stages[2]   # deepest, lowest resolution (28×28) ← default for CAM
```

**Forward hook** `_fwd_hook(name)`: Captures the output tensor of the stage (detached from the computation graph to avoid accidental gradient accumulation in the stored activation).

**Backward hook** `_bwd_hook(name)`: Captures the gradient tensor flowing back through the stage. Uses `register_full_backward_hook` (the modern API that correctly handles all output types, including tuples).

**`remove_xai_hooks()`**: Removes all registered hooks and clears the dictionaries. Always called after each XAI computation to prevent memory leaks and stale gradient accumulation.

### `forward_for_xai(video, text_embeddings, target_action_idx=0)`

This is a second forward method alongside the standard `forward()`. It performs the same computation but additionally computes a **differentiable scalar score** that can be passed to `.backward()`:

```python
logits = self.forward(video, text_embeddings)           # shape: (B, 7, 256)
pred_tokens = logits[:, target_action_idx, :].argmax()  # predicted bin index
scores = logits[:, target_action_idx, pred_tokens]      # probability of predicted bin
return scores, logits
```

**Why not just use `logits.sum().backward()`?** Calling backward on the full logit tensor computes gradients with respect to all 7×256 = 1792 output values, producing mixed gradients that average out task-specific information. By selecting only the predicted bin of the target action dimension (`target_action_idx=0` = first action = wx translational velocity), we obtain gradients that are specific to the model's most confident action prediction, producing sharper and more interpretable CAM maps.

---

## Section 7: Training Utilities

### `FocalLoss` — Same as nb1–nb6 but without temperature scaling

The production notebook uses a simplified `FocalLoss` that drops the `temperature` parameter present in nb1–nb6. Functionally equivalent for `temperature=1.0`. See §6.1 of the main report for full derivation.

### `MixupCutMix` — Same algorithm, different class name

Renamed from `ImprovedMixup` in nb1–nb6. Identical algorithm. See §6.2 of the main report.

### `Lion` — Same implementation

See §6.3 of the main report.

### `CosineWarmupScheduler` — Same as `ImprovedScheduler` in nb1–nb6

Different class name, identical behaviour: linear warmup for `warmup_steps` steps, then cosine annealing to `min_lr_ratio × base_lr`. See §6.4 of the main report.

### `ModelEMA` — Explicit `apply_shadow()` / `restore()` API

The production notebook's `ModelEMA` adds two methods that nb1–nb6's version lacks:

**`apply_shadow(model)`:** Copies the EMA weights into the model's parameters (backing up the current weights first). Called before validation so the EMA model is evaluated rather than the last-batch model.

**`restore(model)`:** Copies the backed-up weights back into the model. Called after validation so training resumes from the live (non-EMA) weights.

This pattern — apply EMA before validation, restore after — is the correct way to use EMA in a training loop. In nb1–nb6, only the shadow weights are saved to disk; in this notebook, EMA is also used for the validation pass itself.

### `PerTaskMetrics` — Batch-level updates

Updated from nb1–nb6 to use `update_batch(loss_per_sample, task_ids)` which processes an entire batch at once, rather than sample-by-sample. This is more efficient because `loss_per_sample` is computed from existing logits (no extra forward pass) via:

```python
per_sample = F.cross_entropy(logits_flat, target_flat, reduction='none')
             .view(-1, 7).mean(dim=1)   # average over 7 action dimensions
```

The `print_worst(dataset, k=5)` method prints the 5 task instructions with the highest average loss, directly identifying which cable routing or DROID subtasks the model struggles with most.

### `ValidationTracker` — New class replacing simple patience counter

`ValidationTracker` is a new utility class that does everything the simple `patience_counter` variable did in nb1–nb6, but adds comprehensive history tracking and trend analysis.

```python
class ValidationTracker:
    def __init__(self, patience=10):
        self.history = {
            'train_loss': [], 'val_loss': [], 'val_acc': [],
            'val_mae': [], 'lr': [], 'throughput': [],
            'train_acc': [], 'improvement': []
        }
```

**Key additions over simple patience counter:**

| Feature | Simple counter (nb1–nb6) | `ValidationTracker` |
|---|---|---|
| Best loss tracking | `best_val_loss = float('inf')` | Same, plus `best_epoch`, `best_val_acc` |
| History | `history` dict updated manually | Encapsulated in `update()` method |
| Trend analysis | Not present | 5-epoch linear regression trend: `np.polyfit(range(5), val_losses[-5:], 1)` |
| Reporting | `tqdm` postfix only | `print_status(epoch)` prints full summary table |
| Early stopping | `if patience_counter >= PATIENCE` | `tracker.should_stop` property |
| Improvement delta | Not tracked | `history['improvement']` stores per-epoch delta |

**Why track the 5-epoch trend?** A model can have a low patience counter but still be on a worsening trend. For example, if validation loss was 2.0, 1.9, 1.95, 1.93, 1.94 over 5 epochs, the patience counter is 3 (no improvement for 3 epochs) but the trend is flat — not clearly worsening. The linear fit gives a slope in "loss per epoch" units, which the trainer can use to decide whether to reduce the LR or continue.

---

## Section 8: Dataset — `AdvancedDroidDataset`

### What is DROID?

DROID (Diverse Robot Operation and Instruction Dataset) is a large-scale real-robot manipulation dataset collected across many laboratory environments using diverse manipulation tasks. Unlike `berkeley_cable_routing` which focuses on a single task type, DROID contains hundreds of distinct manipulation skills with rich natural language instructions. The `droid_train_100.npz` file used here is a **100-episode subset** of the full DROID dataset, pre-processed and saved as a NumPy compressed archive.

**Why NPZ format instead of TFDS?**
The DROID data was pre-processed (images resized, actions standardised) and packaged as a single `.npz` file for portability. This avoids the TFDS/GCS infrastructure required for `berkeley_cable_routing` and eliminates the TF/PyTorch GPU memory conflict problem entirely — there is no TensorFlow involved in this notebook.

### NPZ File Structure

```python
data = np.load('droid_train_100.npz', allow_pickle=True)
data['images']        # shape (N, H, W, 3), dtype uint8 — all frames
data['actions']       # shape (N, action_dim), dtype float32 — action vectors
data['instructions']  # shape (N,), dtype object — instruction strings
```

All frames are stored contiguously. Episode boundaries are implicit: they are not stored in the NPZ file. Instead, `AdvancedDroidDataset` treats all frames as a flat sequence and uses the `seq_len` lookback window to construct clips. This is simpler than `CableRoutingDataset`'s explicit `episode_bounds` system, at the cost of potentially creating clips that span episode boundaries (which the code does not prevent).

### Adaptive Augmentation — Strong vs. Weak

The production dataset introduces an important innovation absent from nb1–nb6: **adaptive augmentation based on task frequency**. In `__getitem__`:

```python
if self.mode == "train":
    transform = (self.strong_aug if self.task_counts[tid] < 500
                 else self.weak_aug)
```

**Strong augmentation** (for tasks with fewer than 500 samples):
```
ToPILImage()
RandomApply([ColorJitter(0.4, 0.4, 0.4, 0.2)], p=0.8)   # aggressive colour jitter
RandomResizedCrop(224, scale=(0.7, 1.0))                   # wider crop range (70–100%)
RandomApply([GaussianBlur(5, sigma=(0.1, 2.5))], p=0.5)   # optional blur
ToTensor()
RandomErasing(p=0.3)                                        # random rectangular masking
```

**Weak augmentation** (for tasks with 500+ samples):
```
ToPILImage()
RandomResizedCrop(224, scale=(0.85, 1.0))  # gentler crop range
ToTensor()
```

**Why adaptive augmentation?** DROID has highly imbalanced task distributions — some manipulation skills have thousands of demonstrations while others have only tens. For rare tasks, strong augmentation artificially multiplies the effective number of training examples the model sees, preventing rare-task features from being "forgotten" due to infrequent updates. For common tasks, aggressive augmentation would introduce unnecessary noise and potentially harm performance on the tasks the model has already learned well.

**Two new augmentations not in nb1–nb6:**
- **`GaussianBlur(5, sigma=(0.1, 2.5))`:** Adds slight blurring with 50% probability. Simulates camera defocus and motion blur, which are common in real robot recordings. Kernel size 5 is large enough to meaningfully blur the image without destroying all fine structure.
- **`RandomErasing(p=0.3)`:** With 30% probability, randomly masks a rectangular region of the image with zeros (black). This forces the model to rely on partial views and prevents over-reliance on any single region of the image — particularly valuable for DROID where task-relevant objects can appear anywhere in the frame.

---

## Section 9: XAI Methods — Custom Built-in Implementations

This is the most significant departure from nb1–nb6. Instead of using the external `pytorch-grad-cam` package, all XAI methods are implemented from scratch and use the hook infrastructure built into `ProductionRT1`.

### `BaseCAM` — Foundation for All CAM Methods

```python
class BaseCAM:
    def __init__(self, model, text_encoder, target_layer_name='vit_stage_2'):
```

`BaseCAM` provides two shared utilities:

**`_get_act_grad(video, text_emb, target_action_idx)`:** Registers hooks, performs a forward pass via `forward_for_xai`, calls `.backward()` to compute gradients, retrieves the stored activation and gradient tensors from `_activations` and `_gradients`, removes hooks, and returns `(activation, gradient, logits, input_tensor)`.

**`_normalize(cam)`:** Maps a CAM tensor to `[0, 1]` per sample in the batch by subtracting the minimum and dividing by the range. Handles both 2D (H, W) and 3D (B, H, W) inputs.

**Why a base class?** All five CAM methods share the hook registration/cleanup pattern and the normalisation step. Factoring this into `BaseCAM` prevents code duplication and ensures that hook cleanup is never accidentally omitted (which would cause memory leaks).

### `HiResCAM` — Provably Faithful Attribution [HiResCAM]

```python
cam = (act * grad).sum(dim=1)   # element-wise product, sum over channels
cam = F.relu(cam)               # keep only positive activations
```

HiResCAM (Draelos & Carin, 2020) [HiResCAM] computes the element-wise product of the activation map and the gradient map **before** summing over channels. Standard GradCAM averages the gradient over spatial dimensions first (`weight_c = (1/HW) Σ_{h,w} grad_{c,h,w}`) and then takes a weighted sum of activations (`Σ_c weight_c × act_c`). This spatial averaging loses spatial precision. HiResCAM's per-position, per-channel multiplication preserves full spatial resolution and is **provably faithful** — the resulting heatmap is a valid first-order Taylor expansion of the model's output with respect to the input activations.

**Why target `vit_stage_2` (the deepest stage) by default?** The deepest feature maps have the highest semantic abstraction — they encode "what" the model is looking at (cable, clip, gripper) rather than low-level edges or textures. For task-level action prediction, semantic attribution is more informative than texture attribution.

### `ShapleyCAM` — Game-Theoretic Second-Order Attribution [ShapleyCAM]

```python
# Gradient (first-order term)
act = model._activations['vit_stage_2']
grad = model._gradients['vit_stage_2']

# Hessian-vector product (second-order term, approximated)
for c in range(min(max_hvp_channels, grad.shape[1])):
    cg = grad[:, c].sum()
    h = torch.autograd.grad(cg, v, retain_graph=True)[0]
    hvp[:, c] = h.mean(dim=(1,2,3)).expand_as(grad[:, c])

shapley_weight = grad + 0.5 * hvp    # second-order Taylor coefficient
cam = (act * shapley_weight).sum(dim=1)
```

ShapleyCAM approximates the Shapley value of each spatial location by computing a second-order Taylor expansion of the model's output. The first-order term is the gradient; the second-order term is the Hessian-vector product (HVP), which captures how the gradient itself changes as the input changes. The coefficient `0.5` comes from the second-order Taylor series `f(x) ≈ f(0) + f'(0)x + 0.5f''(0)x²`.

**Why limit to `max_hvp_channels=16`?** Computing the full HVP requires one backward pass through the computation graph per channel. With 160 channels, this would be 160 backward passes — prohibitively slow. Limiting to 16 channels reduces the HVP computation to 16 backward passes while still capturing the dominant second-order directions (the channels with the highest gradient magnitudes typically carry the most information).

### `LayerCAM` — Multi-Scale Attribution [LayerCAM]

```python
pos_grad = F.relu(grad)              # only positive gradients (what helps the prediction)
cam = (act * pos_grad).sum(dim=1)    # spatially weight activations by positive gradients
cam = F.relu(cam)
```

LayerCAM (Jiang et al., 2021) uses only the **positive** gradient components. The intuition is that negative gradients indicate features that hurt the model's prediction — including these in the attribution would cancel out positive contributions and produce noisy maps. By filtering to positive gradients, LayerCAM produces cleaner, more focused attribution maps.

LayerCAM also runs on **all three stages** simultaneously and combines them:
```
stage_0 features: 112×112, local texture details
stage_1 features: 56×56, mid-level shapes
stage_2 features: 28×28, semantic scene structure
```
The three CAMs are upsampled to a common resolution and summed, producing a multi-scale attribution that captures both fine-grained details (from stage_0) and semantic regions (from stage_2).

---

## Novel XAI Hybrid 1 — `HiResShapleyCAM`

**Full name:** HiResCAM modulated by Shapley-derived spatial attention.

**Motivation:** HiResCAM is provably faithful but does not distinguish which spatial regions are the most *important* in a game-theoretic sense — it only knows which regions are in the gradient path. ShapleyCAM gives importance scores but at the cost of spatial precision. HiResShapleyCAM combines both:

```
Step 1: hires = ReLU(Σ_c A_c ⊙ G_c)          # HiResCAM: faithful element-wise map
Step 2: shapley_score = (G + 0.5 × HVP).sum(c) # Shapley importance per location
Step 3: attn = softmax(shapley_score / τ)        # convert to spatial attention (τ=2.0)
Step 4: fused = ReLU(hires ⊙ attn)              # modulate faithful map by importance
```

**Why softmax with temperature `τ=2.0`?** Raw Shapley scores can have very large differences between spatial locations. A temperature of 2.0 "softens" the softmax distribution — instead of concentrating almost all weight on one location (which would happen at low temperature), it distributes weight more broadly while still respecting the importance ranking. This prevents the fusion from collapsing to a single point.

**What this buys:** The resulting map highlights regions that are simultaneously (a) in the gradient path from the prediction back to the feature map (HiResCAM's faithfulness guarantee) AND (b) have high importance under a second-order game-theoretic scoring (ShapleyCAM's theoretical grounding). Neither method alone provides both properties.

---

## Novel XAI Hybrid 2 — `MultiScaleLayerShapleyCAM`

**Full name:** LayerCAM at multiple scales with Shapley-weighted layer combination.

**Motivation:** Standard LayerCAM sums the CAMs from all stages with equal weight. But different stages contribute differently to different tasks — for a task requiring precise spatial localisation (e.g., picking up a small cable end), shallow stages (112×112 resolution) are more informative. For a task requiring understanding the full scene layout, deep stages (28×28) are more informative. This method uses each layer's Shapley importance to determine how much weight to give it.

```
For each stage l ∈ {0, 1, 2}:
  cam_l = ReLU(Σ_c A_c ⊙ ReLU(G_c))        # LayerCAM at this scale

  grad_magnitude = |G_l|.mean()              # how strong are the gradients here?
  curvature = |∂grad_magnitude/∂input|.mean()  # how much do gradients change?
  importance_l = grad_magnitude + 0.5 × curvature  # Shapley-weighted importance

w = softmax([importance_0, importance_1, importance_2])  # layer weights

Resize all cam_l to the resolution of cam_2 (28×28)
Final = Σ_l w_l × cam_l
```

**Why use curvature to measure importance?** A layer with high gradient magnitude is clearly active, but a layer with high curvature (the gradient changes a lot as the input changes) is in a region of the loss landscape that is highly sensitive. This second-order signal identifies layers whose activations are not just large but also non-linear — precisely the layers that contain the most task-discriminative information.

---

## `LIMEExplainer` — Custom Implementation, No External Dependencies

Unlike nb1–nb6 which used the `lime` library (and required the sklearn mock injection workaround), this notebook implements LIME from scratch:

```python
class LIMEExplainer:
    def __init__(self, model, text_encoder, num_perturbations=500,
                 num_superpixels=50, device='cuda'):
```

**Grid-based superpixels:** Instead of using SLIC (Simple Linear Iterative Clustering) from scikit-image, this implementation divides the image into a regular grid of `n_h × n_w` rectangular cells. While less visually natural than SLIC superpixels (which follow image edges), grid superpixels are much faster to compute and sufficient for identifying broad spatial regions of importance.

**Perturbation procedure:**
1. For each of `num_perturbations=500` iterations: randomly mask each superpixel (set to zero) with 50% probability.
2. Run the perturbed frame through the full model (only the last frame of the video clip is perturbed; the other 5 frames remain unchanged).
3. Record the model's predicted score for the target action's most likely bin.
4. Record the Euclidean distance between the perturbation mask and the "all-ones" mask (original image).

**Local linear fit:**
```python
# LIME kernel: exponential decay weighting by distance from original
kernel_width = 0.75 × sqrt(num_superpixels)
weights = exp(-distances² / (2 × kernel_width²))

# Weighted least squares: (XᵀWX + λI)β = XᵀWy
XtWX = X.T @ diag(weights) @ X + 1e-6 × I
coefficients = solve(XtWX, X.T @ diag(weights) @ y)
```

The `1e-6` regularisation constant prevents `XtWX` from being singular when superpixels are perfectly correlated (which can happen with grid-based segmentation). The coefficients map each superpixel to its estimated contribution to the model's prediction — positive = presence of this region increases the action score, negative = presence decreases it.

**Why 500 perturbations (vs. 128 in nb1–nb6)?** More perturbations give a more accurate local linear approximation. 128 perturbations is sufficient for coarse visualisation but can produce noisy coefficient estimates for superpixels that are rarely masked. 500 perturbations provides enough statistical coverage for stable estimates, especially with 50 superpixels (meaning each superpixel is masked ~250 times on average).

---

## `SHAPExplainer` — DeepSHAP-Style Marginal Contribution Estimation

```python
class SHAPExplainer:
    def __init__(self, model, text_encoder, n_background=20, device='cuda'):
```

This implements a DeepSHAP-style explainer that estimates Shapley values by sampling the marginal contribution of spatial segments relative to background images. Rather than computing exact Shapley values (which requires `2^N` forward passes for N features), it approximates them by:

1. Sampling `n_background=20` background images from the validation set.
2. For each foreground image and each background image, computing the difference in model output when the foreground segment is present vs. absent (replaced by the background segment).
3. Averaging these differences over all background samples.

This is a direct implementation of the Shapley value formula: `φ_i = E_S[f(S ∪ {i}) - f(S)]` approximated by sampling coalitions from the background distribution.

---

## Section 10: XAI Visualisation Engine — `XAIVisualizer`

`XAIVisualizer` is a static utility class for rendering XAI outputs. Unlike the inline plotting in nb1–nb6 (where each block generates its own figures), this class centralises all visualisation into reusable static methods:

- `plot_cam_comparison(video, instructions, cam_results, save_path)` — side-by-side panels for all CAM methods
- `_get_frame_numpy(video_tensor, frame_idx=-1)` — extracts the last frame as a normalised `(H, W, 3)` NumPy array for overlay rendering
- `CMAP_DIVERGING = 'RdBu_r'` — diverging colourmap for SHAP (red = positive, blue = negative)
- `CMAP_SEQUENTIAL = 'inferno'` — sequential colourmap for attention maps
- `CMAP_CAM = 'jet'` — standard jet colourmap for CAM overlays

---

## Section 11: Optimised Training Loop — `train_one_epoch()`

The production training loop is encapsulated as a standalone function rather than an inline `for` loop in a notebook cell. This enables reuse from the training-resume script.

**Key optimisation vs. nb1–nb6 training loop:**

```python
# ❌ nb1–nb6: metrics required a second forward pass
# ✅ Production: reuse logits from the training forward pass
with torch.no_grad():
    preds = logits.argmax(dim=-1)      # shape: (B, 7)
    correct = (preds == target_clean).sum().item()
```

In nb1–nb6, per-task accuracy metrics were computed by running a second `model.eval()` forward pass after each training step. This doubled the compute cost per batch. The production loop stores `target_clean = target.clone()` before MixUp modifies the targets, and reuses the training `logits` to compute accuracy from the same forward pass.

**No `GradScaler` for BFloat16:**
```python
# nb1–nb6: scaler.scale(loss).backward()
#           scaler.unscale_(optimizer)
#           scaler.step(optimizer)
#           scaler.update()

# Production: directly backward, no scaler
loss_scaled.backward()                      # no scaling
torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
optimizer.step()
```

BFloat16 has the same dynamic range as Float32 (8 exponent bits vs. Float16's 5), so gradients cannot underflow to zero — the primary reason for loss scaling. Removing `GradScaler` eliminates 3 extra CUDA synchronisation points per optimiser step, slightly improving throughput.

---

## Section 12: Post-Training XAI Analysis — `run_xai_analysis()`

This function orchestrates the entire XAI suite in order:

```
1/5 → CAM variants + novel hybrids (HiResCAM, ShapleyCAM, LayerCAM,
                                      HiResShapleyCAM, MultiScaleLayerShapleyCAM)
2/5 → LIME (500 perturbations per sample)
3/5 → SHAP (20 background samples)
4/5 → PDP & ICE (brightness, contrast, spatial shift, temporal shift)
5/5 → Composite visualisation dashboard
```

For each of `XAI_NUM_SAMPLES=8` validation samples, all five CAM methods are computed and saved. The function then generates a master comparison figure with all methods side-by-side for each sample, saved to `xai_outputs/`.

---

## Section 13: Main Execution and Training Resume

### `main()` Function

The main function orchestrates the full pipeline: initialise text encoder → create datasets → create dataloaders → build model → set up Lion + CosineWarmupScheduler + FocalLoss + MixupCutMix + ModelEMA → run training loop → run XAI analysis.

The model sanity check runs a dummy forward pass with shape `(2, 3, 6, 224, 224)` and asserts the output shape is `(2, 7, 256)` (batch=2, 7 action dimensions, 256 bins), confirming all tensor reshapes are correct before committing to a full training run.

### Training Resume Cell

The resume cell (Cell 28) is a production-essential feature. Long training runs on cloud or HPC environments frequently disconnect. The checkpoint saved by `CHECKPOINT_FREQ=5` provides recovery points every 5 epochs. The resume cell:

1. Loads `rt1_production_best.pt` (the best checkpoint, which includes the full training state: model weights, optimizer state dict, EMA shadow weights, `ValidationTracker` history, and current epoch number).
2. Restores the optimizer state and scheduler step count so training continues from exactly the point of interruption (same LR, same momentum buffers in Lion).
3. The choice between `rt1_production_best.pt` (epoch 25, best val loss) and `rt1_checkpoint_epoch_20.pt` (epoch 20, periodic) is made by inspecting both files and comparing their recorded validation losses. The best checkpoint is always preferred because it contains more training progress and the full optimizer state.

**Why save optimizer state?** Resuming from just the model weights would restart the Lion momentum buffers at zero and reset the scheduler to the warmup phase. This would cause a "cold start" where the first post-resume epoch behaves like the very beginning of training — potentially worsening the model's performance by overwriting carefully learned weight directions with large early-warmup updates.

---

## Summary: Key Differences Between Production Notebook and nb1–nb6

| Aspect | nb1–nb6 (Cable Routing) | Production (DROID) |
|---|---|---|
| **Dataset** | `berkeley_cable_routing` via TFDS/GCS | `droid_train_100.npz` local NPZ |
| **Dataset class** | `CableRoutingDataset` with memmap cache | `AdvancedDroidDataset` direct RAM load |
| **Augmentation** | Fixed pipeline for all tasks | Adaptive strong/weak by task frequency |
| **New augmentations** | None | `GaussianBlur`, `RandomErasing` |
| **Batch config** | BS=1, accum=128 (effective=128) | BS=15, accum=2 (effective=30) |
| **XAI library** | External `pytorch-grad-cam` | Fully custom built-in hooks |
| **CAM methods** | GradCAM, HiResCAM, ShapleyCAM | HiResCAM, ShapleyCAM, LayerCAM |
| **Novel XAI** | ShaGCAM, LimeCAM (post-hoc fusion) | HiResShapleyCAM, MultiScaleLayerShapleyCAM (architecturally integrated) |
| **LIME** | Via `lime` library + sklearn mock | Custom grid-superpixel + closed-form WLS |
| **Training fn** | Inline loop in notebook cell | Standalone `train_one_epoch()` function |
| **Metrics** | Simple patience counter + history dict | `ValidationTracker` with trend analysis |
| **EMA usage** | Save shadow weights only | Apply to model during validation |
| **Checkpointing** | Best checkpoint only | Every 5 epochs + best |
| **Resume** | Not supported | Dedicated resume cell |
| **GradScaler** | Used (for correctness) | Removed (BFloat16 doesn't need it) |

---

## 👤 Author

**Sparsh Agarwal** — [GitHub](https://github.com/sparsh0106) · [LinkedIn](https://www.linkedin.com/in/sparsh-agarwal-4b443b304/)
