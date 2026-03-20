# Methodology
## X-RT1 Ablation Study — Detailed Research Methodology
> **Authors:** [Fill in] | **Supervisor:** [Fill in] | **Date:** [Fill in]

---

## Table of Contents

1. [Research Design and Experimental Framework](#1-research-design)
2. [Dataset Acquisition and Preparation](#2-dataset)
3. [Data Preprocessing and Augmentation](#3-augmentation)
4. [Model Architecture Design](#4-architecture)
5. [Training Procedure](#5-training)
6. [Evaluation Protocol](#6-evaluation)
7. [Explainability Evaluation (XAI)](#7-xai)
8. [Hardware Configuration](#8-hardware)

---

## 1. Research Design and Experimental Framework

### 1.1 Rationale for Ablation Design

This study employs a **controlled ablation design** — a systematic experimental methodology in which one component is varied across conditions while all other components are held constant — to investigate the contribution of individual architectural elements to the performance, computational efficiency, and interpretability of Vision-Language-Action (VLA) robot transformer models. VLA models take camera images and natural language instructions as inputs and produce robot end-effector actions as outputs.

Published VLA research, including RT-1 [RT-1] and RT-2 [RT-2], presents single monolithic architectures evaluated against prior baselines using different datasets, training regimes, and evaluation protocols. This makes it impossible to attribute observed performance differences to specific architectural choices rather than confounding differences in training procedure or data. By constructing six model variants that share an identical dataset, loss function, optimiser, augmentation pipeline, and XAI evaluation suite — differing only in vision encoder, text encoder, and token learner — this study enables direct, confound-free attributions.

### 1.2 Axes of Variation

The six experimental conditions (notebooks nb1 through nb6) are systematically varied across three independent architectural dimensions:

**1. Vision encoder** — the neural network module responsible for extracting spatial feature representations from input RGB video frames. Variants range from a pure convolutional architecture (ConvNeXt-Small, nb3) to hybrid CNN-Transformer designs (MaxViT-Base/Large in nb1/nb2/nb5, MobileViT-Small in nb6) to a depthwise-separable CNN (EfficientLite-M, nb4). The vision encoder constitutes the dominant cost driver in terms of VRAM and compute.

**2. Text encoder** — the module responsible for encoding the natural language instruction into a fixed-length embedding vector that conditions the model's action prediction. Variants span the full complexity spectrum from a fully trainable Bag-of-Words model (nb3), frozen distilled transformers (DistilBERT in nb1/nb2), a partially fine-tuned BERT (nb5), to CLIP's jointly-trained vision-language text tower (nb4, nb6).

**3. Token learner** — the module that compresses the vision encoder's high-dimensional spatial feature map (784 positions at 28×28 resolution) into a small, fixed-length sequence of token vectors suitable for the downstream transformer. Variants include cross-attention with learned queries (nb1/nb2), convolutional attention-free pooling (nb3), hierarchical multi-scale pooling (nb4), full self-attention followed by cross-attention (nb5), and Perceiver-style iterative cross-attention (nb6).

All other components — the temporal fusion module, the downstream transformer encoder, the action tokenisation scheme, training utilities, the dataset class, and the XAI evaluation pipeline — are held fixed across all six conditions.

### 1.3 Full Experimental Matrix

| Property | nb1 | nb2 | nb3 | nb4 | nb5 | nb6 |
|---|---|---|---|---|---|---|
| Vision encoder | MaxViT-Base | MaxViT-Base | ConvNeXt-Small | EfficientLite-M | MaxViT-Large | MobileViT-Small |
| Text encoder | DistilBERT (frozen) | DistilBERT+VQA (frozen) | BoW-256 (trainable) | CLIP-B/32 (frozen) | BERT top-2 (partial ft) | CLIP-B/32 (frozen) |
| Token learner | CrossAttn 32 | CrossAttn 32 | ConvTL 16 | Hierarchical 32 | SelfAttn 64 | PerceiverResampler 32 |
| Temporal fusion | ImprovedTemporalConv | ImprovedTemporalConv | ImprovedTemporalConv | ImprovedTemporalConv | ImprovedTemporalConv | CausalTemporalAttn |
| Action head | Standard | ChainOfThought | Standard | Standard | Standard | ActionChunking ×4 |
| Transformer | 6L, 8H | 6L, 8H | 4L, 4H | 5L, 4H | 8L, 8H | 4L, 8H |
| embed_dim | 320 | 320 | 192 | 96 | 512 | 128 |
| Training status | Trained | Trained | Trained | Trained | **NOT trained** | Trained |

### 1.4 Second Dataset Experiment

A seventh experimental condition (the production notebook) replicates the nb1 architecture on the DROID dataset — a second, substantially different real-robot manipulation benchmark with a richer and more varied instruction vocabulary — to test whether findings from the single-task Berkeley Cable Routing domain generalise to multi-task manipulation.

---

## 2. Dataset Acquisition and Preparation

### 2.1 Berkeley Cable Routing

#### 2.1.1 Identity and Access

The primary experimental dataset is `berkeley_cable_routing`, published by the University of California, Berkeley as part of the Open X-Embodiment (OXE) collaborative dataset project [OXE] in 2023. It is hosted on Google Cloud Storage at `gs://gresearch/robotics/berkeley_cable_routing` and accessed via the TensorFlow Datasets (TFDS) `builder_from_directory` API.

#### 2.1.2 Task Description

Each episode corresponds to one complete robotic cable routing attempt: a robot arm picks up a loose cable, threads it through mounting clips on a surface, and releases it at a designated endpoint. All observations are captured from a single fixed-perspective RGB camera. The action space is 7-dimensional and continuous: three translational end-effector velocities (`wx`, `wy`, `wz`), three rotational velocities (`rx`, `ry`, `rz`), and a binary termination flag. Each step contains one synchronised (observation, action) pair.

A notable characteristic of this dataset is that every episode carries the same language instruction — "route cable" — reflecting the dataset's single-task nature. The text encoder's role is therefore primarily as a conditioning bias rather than a semantic discriminator between task types, which is an important caveat when interpreting cross-encoder comparison results.

#### 2.1.3 Download and Verification

Dataset acquisition is implemented in Block 3 of each notebook. `check_existing()` scans the local directory for `dataset_info.json` and bypasses download if found. `download_gcs_tf()` uses `tf.io.gfile` for GCS authentication, chunked transfer, and retry handling. Each file's local size is compared to its remote size via `tf.io.gfile.stat(p).length` before copying, making the download resumable after interruption.

**TF/PyTorch GPU conflict resolution:** TensorFlow and PyTorch share the same VRAM pool. Without intervention, TF pre-allocates all available VRAM at import time, leaving nothing for the PyTorch model. Two strategies are employed: (a) in nb1, nb2, nb5, nb6 — TF is hard-capped to 512 MB via `tf.config.LogicalDeviceConfiguration(memory_limit=512)`; (b) in nb3, nb4 — TF's GPU visibility is disabled entirely via `tf.config.set_visible_devices([], 'GPU')`. The 512 MB cap is the empirically determined minimum for TFDS to decode TFRecord shards without error.

#### 2.1.4 Runtime Structure Discovery

Field names vary across OXE sub-datasets. Block 4 auto-discovers them at runtime by scanning observation fields for a rank-3 tensor with final dimension 3 (image) and for a scalar string/bytes field (instruction). Action keys are sorted alphabetically to guarantee a deterministic concatenation order regardless of Python dict iteration order, yielding `ACTUAL_ACTION_DIM = 7`.

#### 2.1.5 Disk Caching

Direct TFDS iteration per epoch is prohibitively slow due to TFRecord decode overhead. All notebooks implement `CableRoutingDataset`, which performs a two-pass disk cache construction on first run:

- **Pass 1:** Count total frames to determine the `numpy.memmap` array shape before allocation.
- **Pass 2:** Resize each frame to `224×224` via `PIL.BILINEAR` and write sequentially to `images.dat` (memory-mapped `uint8` array, shape `(N, 224, 224, 3)`). Action vectors and instruction strings are saved to `actions.npy` and `instructions.npy`; episode boundary indices to `metadata.npy`.

`numpy.memmap` maps the file into virtual memory; the OS loads only the pages accessed per `__getitem__` call — in practice, 6 consecutive rows (one video clip). This provides random-access performance equivalent to an in-memory array while requiring negligible RAM regardless of dataset size.

The dataset split is performed at the **episode level** (not frame level): an 80/20 split expressed as a TFDS string slice ensures all frames from an episode appear exclusively in either train or validation, preventing temporal information leakage.

### 2.2 DROID Dataset (`droid_train_100.npz`)

The production experiment uses a 100-episode subset of the DROID dataset, pre-processed into a single NumPy compressed archive with three arrays: `images (N, H, W, 3)`, `actions (N, action_dim)`, and `instructions (N,)`. This format eliminates all TF infrastructure and the associated GPU memory conflict. The full archive is loaded into system RAM during initialisation, enabling O(1) frame access and the substantially larger effective batch size used in the production experiment (`BATCH_SIZE=15` vs. `BATCH_SIZE=1` in nb1–nb6).

---

## 3. Data Preprocessing and Augmentation

### 3.1 Strategy: Adaptive Augmentation by Task Frequency

An adaptive augmentation strategy is employed: task groups with fewer than 500 training samples receive the **strong augmentation** pipeline; groups with 500 or more samples receive the **weak augmentation** pipeline. Rare tasks benefit most from aggressive augmentation because they have fewer raw training examples; common tasks risk unnecessary distributional shift from overly aggressive transforms.

### 3.2 Strong Augmentation Pipeline

Applied to rare task categories (task_counts < 500). Defined in `_setup_augmentation()` of both `CableRoutingDataset` (nb1–nb6) and `AdvancedDroidDataset` (production):

```python
transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.2)], p=0.8),
    transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
    transforms.RandomApply([transforms.GaussianBlur(5, (0.1, 2.5))], p=0.5),
    transforms.ToTensor(),
    transforms.RandomErasing(p=0.3),
])
```

Each transform is motivated by specific real-world variability in robot deployments:

- **`RandomApply([ColorJitter(0.4, 0.4, 0.4, 0.2)], p=0.8)`** — Applied with 80% probability. Randomly perturbs brightness and contrast by up to ±40%, saturation by up to ±40%, and hue by up to ±20%. Simulates workspace lighting variation across time-of-day, shadow conditions, and reflections from the cable surface. The high application probability (0.8) reflects the importance of photometric invariance for a task whose semantics are purely geometric.

- **`RandomResizedCrop(224, scale=(0.7, 1.0))`** — Extracts a random crop covering 70–100% of image area and resizes to 224×224. Simulates variation in the effective camera field of view arising from small differences in mounting position and lens calibration across data collection sessions.

- **`RandomApply([GaussianBlur(kernel_size=5, sigma=(0.1, 2.5))], p=0.5)`** — Applied with 50% probability. Gaussian blur simulates camera defocus and motion blur, which are common artefacts in real robot video streams.

- **`RandomErasing(p=0.3)`** — Applied with 30% probability. A randomly sized and positioned rectangular region is set to zero (black). Forces the model to develop robustness to partial occlusion of the scene.

### 3.3 Weak Augmentation Pipeline

Applied to common task categories (task_counts ≥ 500):

```python
transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomResizedCrop(224, scale=(0.85, 1.0)),
    transforms.ToTensor(),
])
```

Only a gentle random crop (85–100% of area) is applied. No colour perturbation, blurring, or erasing. Provides minimal viewpoint variation without distributional shift.

### 3.4 Validation Transform

Deterministic and non-augmenting:

```python
transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])
```

No random operations. Validation metrics are fully reproducible across runs.

### 3.5 Balanced Sampling

The training set exhibits class imbalance at the task-instruction level. A `WeightedRandomSampler` assigns each sample a weight equal to the inverse frequency of its task category (`weight_i = 1 / count(task_id_i)`), normalised to sum to the total dataset size. This ensures each task category contributes equally to every epoch on average, regardless of raw frequency.

---

## 4. Model Architecture Design

### 4.1 Shared Computational Graph

All six model variants share the same high-level data flow:

```
Input: video (B, 3, 6, 224, 224), instruction string

Step 1 — Flatten frames:
  (B, 3, 6, 224, 224) → (B×6, 3, 224, 224)

Step 2 — Vision encoder:
  (B×6, 3, 224, 224) → (B×6, embed_dim, 28, 28)

Step 3 — Restore temporal dimension:
  (B×6, embed_dim, 28, 28) → (B, embed_dim, 6, 28, 28)

Step 4 — Temporal fusion:
  (B, embed_dim, 6, 28, 28) → (B, embed_dim, 6, 28, 28)

Step 5 — Token learning (flatten frames again):
  (B×6, embed_dim, 28, 28) → (B×6, N, embed_dim) → (B, 6N, embed_dim)

Step 6 — Text encoding + projection:
  instruction → (B, TEXT_DIM) → (B, 1, embed_dim)

Step 7 — Transformer:
  cat([text_token, visual_tokens]) shape (B, 6N+1, embed_dim) → attended

Step 8 — Action head from CLS (position 0):
  (B, embed_dim) → (B, 7, 256)  [or (B, 4, 7, 256) for nb6]
```

The text token is prepended at position 0 and its representation after the transformer acts as the CLS aggregation token — all 6N visual tokens are attended to through this text-conditioned query. This is the mechanism through which the language instruction directs the model's visual attention and action prediction.

The compression from 784 (28×28) to N spatial positions by the token learner is computationally critical: transformer self-attention cost scales as O((6N+1)²). At N=32 (most notebooks): 193 tokens → ~37K attention operations per head. At N=784 (no token learner): 4705 tokens → ~22M operations per head — a 595× increase.

### 4.2 Shared Primitive Components

Before describing the per-notebook architectures, the shared building blocks that appear across all variants are defined here. These are prefixed "Improved" in nb1–nb6 to signal they are enhanced versions of published components.

**`ImprovedSqueezeExcitation(dim, reduction_ratio=4)` [SE]**

Channel-wise recalibration gate. Forward pass:
1. Global average pool: `(B,C,H,W) → (B,C)` via `einops.Reduce`
2. `Linear(C→C/4)` → `BatchNorm1d(C/4)` → `SiLU` → `Dropout(0.1)`
3. `Linear(C/4→C)` → `Sigmoid`
4. Broadcast to `(B,C,1,1)` and multiply with input

Additions beyond standard SE [SE]: `BatchNorm1d` stabilises the squeeze MLP; `Dropout(0.1)` prevents the channel gate from overfitting. Reduction ratio = 4 gives a bottleneck of `C/4` channels — the standard [SE] trade-off between parameter efficiency and expressiveness.

**`ImprovedMBConv(dim_in, dim_out, *, downsample, expansion_rate, use_fused, drop_path_rate)` [EfficientNet]**

Mobile Inverted Bottleneck Convolution. Two variants:
- **Standard** (`use_fused=False`): `1×1 expand → BN → SiLU → 3×3 depthwise → BN → SiLU → SE → 1×1 project → BN`
- **Fused** (`use_fused=True`): `3×3 standard conv → BN → SiLU → SE → 1×1 project → BN`

Residual shortcut is applied only when `dim_in == dim_out` and `not downsample`. `DropPath` with probability `drop_path_rate × block_idx / total_blocks` is applied to the residual branch (probability increasing with block depth, providing stochastic depth regularisation [DropPath]).

`expansion_rate=4` is the EfficientNet standard: channel expansion before the depthwise conv increases representational capacity at low parameter cost. `se_ratio=0.25` means SE bottleneck = `1/0.25 = 4×` reduction.

**`ImprovedTemporalConv(dim)`**

Novel 3D temporal fusion module (no direct published analogue). Given input `(B, D, F, H, W)`:
1. Depthwise 3D conv `Conv3d(D, D, (3,1,1), groups=D, padding=(1,0,0))` → temporally smoothed `x_t` (blends adjacent frames, channel-independently)
2. Concatenate: `cat = [x, x_t]` → `(B, 2D, F, H, W)`
3. 3D SE gate: `AdaptiveAvgPool3d(1) → Conv3d(2D,D,1) → GELU → Conv3d(D,2D,1) → Sigmoid` → scale `cat`
4. `Conv3d(2D, D, 1)` projection → residual add with original `x`

The gating mechanism learns to amplify the most motion-rich frames (e.g., cable grasping) and suppress static frames. The temporal conv uses `groups=dim` (depthwise) to keep the time-axis blending computationally cheap — it does not mix channels, preserving the vision encoder's learned feature directions.

### 4.3 Vision Encoders

#### nb1, nb2 — `AdvancedMaxViT` (MaxViT-Base: `dim=40, depth=(4,4,3)`)

MaxViT [MaxViT] alternates between MBConv blocks and two attention mechanisms — local window attention and dilated grid attention — at each spatial scale. Captures both local texture (cable contact points, clip details) and long-range spatial context (overall routing path layout) without full O(N²) image-level attention.

Architecture flow for 224×224 input:
```
Stem: Conv(3→40,3×3,s=2) → BN → SiLU → Conv(40→40,3×3,s=1) → BN → SiLU
      → (B×F, 40, 112, 112)
Stage 0: 4× MBConv(40→40, fused, no DS) → (B×F, 40, 112, 112)
Stage 1: MBConv(40→80, fused, DS) + 3× MBConv(80→80, fused) → (B×F, 80, 56, 56)
Stage 2: MBConv(80→160, std, DS) + 2× MBConv(160→160, std) → (B×F, 160, 28, 28)
Output: embed_dim = 40 × 2³ = 320
```
`USE_FUSED_MBCONV=True` applies fused MBConv to stages 0 and 1 (smaller channel counts; fewer kernel launches). Stage 2 uses standard (depthwise-separable) MBConv.

#### nb5 — `AdvancedMaxViT` (MaxViT-Large: `dim=64, depth=(6,5,5)`)

Same architecture, scaled up: `embed_dim = 64 × 2³ = 512`. `USE_FUSED_MBCONV=False` (larger channels make fused convolutions more memory-intensive). `DROP_PATH_RATE=0.3` (increased regularisation for deeper model). The 16 total blocks and 512-dimensional features result in approximately 3× the parameters of MaxViT-Base and approximately 2.5 hours per training epoch.

#### nb3 — `ConvNeXtEncoder` (ConvNeXt-Small: `dims=(48,96,192)`)

ConvNeXt [ConvNeXt] is a pure convolutional architecture redesigned with transformer-inspired components: 7×7 depthwise convolutions (large receptive field per parameter), inverted bottleneck, `LayerNorm2d` in place of `BatchNorm2d`, `GELU` activation, and a separate patchify stem with stride 4. No attention mechanism is present — this is the control condition for testing whether global visual attention provides measurable benefit on this task.

```
Stem: Conv2d(3→48, 4×4, stride=4) + LayerNorm2d → (B×F, 48, 56, 56)
Stage 0: 2× ConvNeXtBlock(48) → (B×F, 48, 56, 56)
Stage 1: Downsample(48→96) + 2× ConvNeXtBlock(96) → (B×F, 96, 28, 28)
Stage 2: Downsample(96→192) + 3× ConvNeXtBlock(192) → (B×F, 192, 28, 28)
Output: embed_dim = 192
```
Each `ConvNeXtBlock` applies: `DWConv(7×7) → LN → Linear(C→4C) → GELU → Linear(4C→C) → gamma_scale → residual`.

#### nb4 — `EfficientLiteEncoder` (EfficientLite-M: `dims=(32,64,96)`)

A custom lightweight CNN using depthwise-separable `DSConvBlock`s (depthwise 3×3 then pointwise 1×1). Configuration `size='m'`, `blocks=(3,4,4)`:
```
Stem: Conv2d(3→32, 3×3, stride=2) + BN + SiLU → (B×F, 32, 112, 112)
Stage 0: 3× DSConvBlock(32→32, stride=1) → (B×F, 32, 112, 112)
Stage 1: DSConvBlock(32→64, stride=2) + 3× DSConvBlock(64→64) → (B×F, 64, 56, 56)
Stage 2: DSConvBlock(64→96, stride=2) + 3× DSConvBlock(96→96) → (B×F, 96, 28, 28)
Output: embed_dim = 96
```
The smallest `embed_dim` in the study. Paired with the 32-token HierarchicalTokenLearner, nb4 is the most VRAM-efficient non-convolutional-TL model.

#### nb6 — `MobileViTEncoder` (MobileViT-Small: `embed_dim=128`)

MobileViT [MobileViT] combines MBConv stages with `MobileViTBlock`s. Each `MobileViTBlock` partitions the feature map into non-overlapping 2×2 patches, applies a 2-layer 4-head transformer across patch sequences (allowing global pixel-level interaction at O(N) cost relative to patch count), then folds the output back to the spatial grid.

```
Stem: Conv2d(3→16, 3×3, stride=2) → (B×F, 16, 112, 112)
DS1:  MBConv(16→24, stride=2)    → (B×F, 24, 56, 56)
DS2:  MBConv(24→48, stride=2)    → (B×F, 48, 28, 28)
Proj: Conv2d(48→128, 1×1)        → (B×F, 128, 28, 28)
MV1, MV2: 2× MobileViTBlock(128) → (B×F, 128, 28, 28)
Output: embed_dim = 128
```

### 4.4 Text Encoders

#### nb1, nb2 — `DistilBERTEncoder` (768-dim, frozen)

DistilBERT-base-uncased [DistilBERT]: 6 transformer layers, 12 heads, hidden=768. Loaded from HuggingFace and fully frozen. CLS token from final hidden state → `(B, 768)`. Frozen because the cable routing instruction vocabulary is narrow and repetitive; fine-tuning risks overfitting to a handful of phrase variations.

In nb2, `VQAFormatter` wraps each instruction as `"Q: what action should the robot take to {task}? A:"` before tokenisation. This aligns the input format with the factoid QA pretraining patterns of DistilBERT at zero parameter cost, potentially eliciting richer semantic representations.

#### nb3 — `BoWTextEncoder` (256-dim, fully trainable)

`Embedding(8192, 128)` lookup using the DistilBERT subword vocabulary. Tokens averaged to a pooled vector → `MLP: Linear(128→256) → GELU → Dropout(0.1) → Linear(256→256) → LayerNorm` → `(B, 256)`. Fully trainable — text representations co-adapt with visual and action parameters. The lower `TEXT_DIM=256` matches the smaller `embed_dim=192` of ConvNeXt-Small, reducing the projection layer's parameter count.

#### nb4, nb6 — `CLIPTextEncoder` (512-dim, frozen)

CLIP ViT-B/32 text tower [CLIP], loaded from `openai/clip-vit-base-patch32`. `pooler_output` (EOS position embedding) → `(B, 512)`. Pretrained contrastively on 400M image-caption pairs; the shared vision-language embedding space may produce instruction embeddings that are more semantically aligned with visual robot features than text-only pretraining. `TEXT_DIM=512`.

#### nb5 — `BERTEncoder` with partial fine-tuning (768-dim)

BERT-base-uncased [BERT]: 12 transformer layers, 12 heads, hidden=768. `unfreeze_top_k=2`: layers 0–9 frozen, layers 10–11 trainable. CLS token → `(B, 768)`.

Lower layers (frozen): encode general syntax and morphology transferable across all language tasks. Upper layers (trainable): adapt task-specific semantic representations to cable routing instructions. Full fine-tuning is infeasible within nb5's VRAM budget (MaxViT-Large already uses the majority of available memory) and risks catastrophic forgetting on the small instruction vocabulary.

### 4.5 Token Learners

#### nb1, nb2 — `AdvancedTokenLearner` (Cross-Attention, N=32)

Maintains 32 learnable query vectors as `nn.Parameter(1, 32, D)`. A positional embedding `(1, D, 7, 7)` bilinearly interpolated to `(h, w)` is added to the input before flattening.

```
spatial = flatten(feature_map + pos_embed)   # (B, 784, D)
Q = queries.expand(B, -1, -1)               # (B, 32, D)
attn_out, _ = MultiheadAttention(Q, spatial, spatial)
tokens = LayerNorm(Q + attn_out)             # (B, 32, D)
tokens = LayerNorm(tokens + FFN(tokens))     # (B, 32, D)
```

After training, different query vectors specialise on different scene elements (cable body, clip positions, gripper). This dynamic, input-dependent pooling is the primary advantage over convolutional token learning.

#### nb3 — `ConvTokenLearner` (N=16, no attention)

```
attn_maps = Sigmoid(Conv2d(D→16,1×1)(BN(GELU(Conv2d(D→16,1×1)(x)))))  # (B,16,H,W)
attn_maps = attn_maps / attn_maps.sum(dim=(2,3), keepdim=True)          # normalise
tokens    = bmm(attn_maps.view(B,16,HW), x.view(B,HW,D))               # (B,16,D)
```

Attention distributions are functions only of input spatial values — not conditioned on learnable queries. Ablates the dynamic query-conditioned pooling of CrossAttn to test whether it provides measurable benefit.

#### nb4 — `HierarchicalTokenLearner` (N=32, two scales)

```
fine   = ConvTokenLearner(D, 24)(feature_map)              # (B, 24, D)  at 28×28
pooled = AvgPool2d(2,2)(feature_map)                       # (B, D, 14, 14)
coarse = ConvTokenLearner(D, 8)(pooled)                    # (B,  8, D)  at 14×14
tokens = LayerNorm(cat([fine, coarse], dim=1))             # (B, 32, D)
```

Fine tokens capture local spatial detail (cable endpoints, clip locations); coarse tokens capture global scene layout (cable routing path orientation).

#### nb5 — `SelfAttentionTokenLearner` (N=64, two-stage)

```
spatial  = flatten(feature_map) + pos_bias  # (B, 784, D)
sa, _    = self_attn(spatial, spatial, spatial)
spatial  = LayerNorm(spatial + sa)           # globally contextualised (B, 784, D)

Q        = queries.expand(B, -1, -1)        # (B, 64, D)
ca, _    = cross_attn(Q, spatial, spatial)
tokens   = LayerNorm(Q + ca)                # (B, 64, D)
```

Each of 64 output tokens is a globally-contextualised feature summary: every spatial position has attended to all others before pooling. Most expressive token learner in the study; contributes significantly to nb5's VRAM and compute cost.

#### nb6 — `PerceiverResampler` (N=32, depth=2)

Two layers of (cross-attention → self-attention → FFN) on 32 learnable latents [Perceiver]:

```
for l in range(2):
    latents = LayerNorm(latents + cross_attn(latents, spatial, spatial))
    latents = LayerNorm(latents + self_attn(latents, latents, latents))
    latents = LayerNorm(latents + FFN(latents))
```

Layer 1: latents gather spatial information. Layer 2: latents share information with each other, enabling collective scene representation. More expressive than single-round CrossAttn (nb1) due to the self-attention refinement step.

### 4.6 Temporal Fusion

#### nb1–nb5 — `ImprovedTemporalConv`

Described in §4.2 (Shared Primitive Components). Applied after vision encoder, before token learner.

#### nb6 — `CausalTemporalAttention`

Multi-head self-attention over the 6-frame temporal sequence with an upper-triangular causal mask. Each frame `t` attends only to frames `0..t`, matching the real-time deployment constraint (future frames are unavailable). More expressive than the 3-frame convolutional window of `ImprovedTemporalConv` — long-range temporal dependencies (e.g., the gripper state 5 frames ago) are directly accessible.

### 4.7 Action Representation

**RT1Tokenizer:** Continuous actions `∈ [-3.2, +3.2]` are quantised to 256 bins:
`bin = floor((clamp(v,-3.2,3.2) + 3.2) / 6.4 × 255)`
and decoded as `v = bin / 255 × 6.4 - 3.2`. Range ±3.2 covers ±3σ of end-effector velocities in the dataset. Discretisation converts regression to 256-class classification, avoiding the averaged-prediction problem of MSE and enabling cross-entropy training.

**Standard head (nb1–nb5):** `LN(D) → Linear(D,D/2) → GELU → Dropout → Linear(D/2, 7×256)` → reshape `(B,7,256)`.

**ChainOfThoughtHead (nb2):**
1. `plan_head(cls) → (B,5)` plan logits
2. Teacher-forced plan embedding at train time; argmax at inference
3. `fusion(cat([cls, plan_emb])) → (B,D)`
4. `action_head(fused) → (B,7,256)`
Total loss: `action_loss + 0.3 × plan_CE`.

**ActionChunking head (nb6):** Outputs `(B, 4, 7, 256)`. Predicts 4 future timesteps simultaneously [ActionChunk]. Only position 0 is supervised during training (single-step labels available). At deployment, all 4 predicted actions are executed before re-querying — 4× inference frequency reduction.

---

## 5. Training Procedure

### 5.1 Loss Function

**`ImprovedFocalLoss(γ=2.5, label_smoothing=0.1, temperature=1.0)`** [Focal]:

`L = mean((1 - p_t)^γ × CE(logits/T, targets; ε=0.1))`

where `p_t = softmax(logits/T)[correct_class]`.

- `γ=2.5`: stronger focusing than the original `γ=2.0`; appropriate for 256-class action bin setting where many bins are trivially incorrect.
- `label_smoothing=0.1`: prevents overconfident bin predictions; important given action discretisation noise.
- `temperature=1.0`: fixed during training; adjustable at inference for calibration.

For nb2: `total_loss = action_loss + 0.3 × CrossEntropy(plan_logits, plan_labels)`. `PLAN_LOSS_WEIGHT=0.3` provides a meaningful planning signal without overwhelming action prediction.

### 5.2 Optimisation

**`Lion(lr, betas=(0.9,0.99), weight_decay)` [Lion]:**

```
p    ← p × (1 - lr × wd)                         # decoupled weight decay
update = β₁ × m + (1 - β₁) × g                   # interpolated direction
p    ← p - lr × sign(update)                      # sign-based parameter update
m    ← β₂ × m + (1 - β₂) × g                     # momentum EMA update
```

Sign-based updates require half the memory of Adam (one momentum buffer vs. two) [Lion]. Larger effective step size vs. gradient-scaled Adam necessitates smaller learning rates:

| Notebook | LR | Rationale |
|---|---|---|
| nb3 | `1e-4` | Lightweight model tolerates larger updates |
| nb1, nb2, nb4, nb6 | `3e-5` | Standard for medium-scale transformer training |
| nb5 | `2e-5` | Larger model sensitive to large updates |

`weight_decay=0.15` (nb1/2/4/5/6) and `0.1` (nb3); decoupled implementation.

### 5.3 Learning Rate Schedule

Linear warmup then cosine decay via `ImprovedScheduler`:

- Warmup phase: `lr(t) = base_lr × t / warmup_steps` for `t < warmup_steps`
- Decay phase: `lr(t) = min_ratio × base + (base - min_ratio × base) × 0.5 × (1 + cos(π × progress))`

`min_lr_ratio=0.1`: LR decays to 10% of peak, not zero. Prevents complete learning stagnation in final epochs.

| Notebook | Warmup steps | Rationale |
|---|---|---|
| nb3 | 300 | Smaller model converges faster; shorter warmup sufficient |
| nb1,2,4,6 | 500 | Standard for medium-scale models |
| nb5 | 600 | Larger gradient norms require more stabilisation steps |

### 5.4 Regularisation

Four mechanisms operate simultaneously:

1. **DropPath** [DropPath]: Probability = `DROP_PATH_RATE × block_idx / total_blocks`. Randomly bypasses residual branches — implicit stochastic depth.
2. **Dropout** (`p=0.2`): Applied in transformer encoder and projection layers.
3. **MixUp + CutMix** [MixUp, CutMix]: `α=0.3`, `cutmix_prob=0.5`. Mixed loss: `λ·L(y_a) + (1-λ)·L(y_b)`.
4. **Decoupled weight decay** (`wd=0.15`): Applied via Lion before the gradient step.

### 5.5 Mixed Precision and GPU Optimisations

BFloat16 AMP via `torch.amp.autocast('cuda', dtype=torch.bfloat16)`. BFloat16 preferred over Float16: same 8-bit exponent range as Float32 prevents gradient underflow (no loss scaling required). `GradScaler` is retained for correctness but is a functional no-op in BF16 mode.

GPU throughput optimisations applied globally:
- `allow_tf32=True` for matmul and cuDNN: ~3× speedup on Ampere/H100 with negligible accuracy loss.
- `cudnn.benchmark=True`: Auto-selects fastest cuDNN kernels for fixed 224×224 input shapes.
- `non_blocking=True` on `.to(device)`: Overlaps CPU-to-GPU transfer with GPU kernel execution.

**Gradient checkpointing for nb5 [GradCheck]:** Applied to all ViT blocks, `temporal_model`, and `token_learner`. Recomputes intermediate activations during backward instead of storing them. Reduces peak VRAM ~40% at ~20% extra compute cost. Must use `use_reentrant=False` for correctness with BFloat16 AMP.

### 5.6 Training Loop and Gradient Accumulation

Gradient accumulation simulates larger effective batches within GPU memory constraints:

| Notebook | BATCH_SIZE | GRADIENT_ACCUM_STEPS | Effective batch |
|---|---|---|---|
| nb1,2,4,5,6 | 1 | 128 | 128 |
| nb3 | 4 | 32 | 128 |
| Production (DROID) | 15 | 2 | 30 |

All nb1–nb6 achieve the same effective batch of 128. Production uses 30 because DROID's richer task diversity makes smaller batches statistically sufficient.

`GRAD_CLIP=1.0` [GradClip]: Gradient norms clipped before each optimiser step. Prevents gradient explosion — critical for Lion's sign-based updates which can amplify large gradient directions.

**Core loop:**
```python
for epoch in range(80):
  for batch_idx, (video, inst, target, task_ids) in enumerate(loader_train):
    video    = video.to(DEVICE, non_blocking=True)
    text_emb = text_encoder.encode(inst)          # torch.no_grad()
    video, ya, yb, lam = mixup(video, target)     # in-batch augmentation
    with autocast(bfloat16):
      logits = model(video, text_emb)
      loss   = (lam*L(logits,ya) + (1-lam)*L(logits,yb)) / ACCUM
    scaler.scale(loss).backward()
    if (batch_idx+1) % ACCUM == 0:
      clip_grad_norm_(model, 1.0)
      scaler.step(optimizer); scaler.update()
      scheduler.step(); optimizer.zero_grad()
      ema.update(model)
```

### 5.7 Checkpointing and Early Stopping

- **Early stopping:** `PATIENCE=10` epochs without validation loss improvement → training terminates. 10 epochs ≈ 12% of the 80-epoch budget — long enough to distinguish genuine plateau from transient stagnation.
- **ModelEMA [EMA]:** `decay=0.9999`. Shadow weights updated as `ema ← 0.9999·ema + 0.0001·model`. Used for validation and inference. Produces consistently lower validation loss than last-batch checkpoint by averaging out stochastic batch-to-batch parameter fluctuations.
- **Best checkpoint:** Saved as `{MODEL_NAME}_best.pt` whenever validation loss improves. Stores EMA shadow weights, not live training weights.

---

## 6. Evaluation Protocol

Evaluation is performed at the end of each training epoch on the held-out validation set (last 20% of episodes).

**Primary metric:** Validation loss (`ImprovedFocalLoss`) computed without gradients, without augmentation (val transform only), and using EMA model weights.

**Baseline comparison:** The theoretical random-predictor baseline is `ln(256) ≈ 5.545` — the cross-entropy loss of a uniform distribution over 256 bins. Any trained model should achieve loss substantially below this value. Block 19 reports the absolute improvement `Δ = 5.545 - model_val_loss` and the percentage improvement `Δ / 5.545 × 100%`.

**Per-task metrics:** `PerTaskMetrics` tracks loss separately for each unique instruction string. `print_worst(k=5)` identifies the five task categories with the highest average validation loss, providing interpretable diagnostic information about which subtasks the model struggles with most.

---

## 7. Explainability Evaluation (XAI)

Following training, each model undergoes a standardised 7-block XAI evaluation suite (Blocks 14–21). All XAI computations use EMA model weights on a fixed set of `XAI_N` validation samples. TensorFlow GPU visibility is cleared before execution to free VRAM.

### 7.1 GradCAM Family (Block 16)

Target layer: last `BatchNorm2d` in the vision encoder backbone (auto-discovered by scanning named modules in reverse). Normalised activations at BatchNorm produce cleaner gradient maps than raw Conv2d outputs.

**GradCAM [GradCAM]:** `weight_c = mean_{h,w}(∂score/∂A_c)`. `CAM = ReLU(Σ_c weight_c × A_c)`. Coarse spatial heatmap; fast.

**HiResCAM [HiResCAM]:** `CAM = ReLU(Σ_c A_c ⊙ ∂score/∂A_c)`. Element-wise product preserves spatial resolution. Provably satisfies a faithfulness axiom violated by standard GradCAM.

**ShapleyCAM [ShapleyCAM]:** `weight_c ≈ ∂score/∂A_c + 0.5 × HVP_c`. Second-order Taylor expansion via Hessian-vector product (computed for ≤16 channels to control cost). Game-theoretically grounded channel importance.

Outputs: static heatmaps, animated 100-frame videos, filmstrip contact sheets, temporal evolution plots, spatial attention trajectory plots.

### 7.2 LIME (Block 17) [LIME]

Image partitioned into superpixels. For each of 128 perturbations: random superpixel subset masked to zero → model prediction recorded. Locally-weighted ridge regression fitted to (mask, prediction) pairs → top-5 superpixels by coefficient magnitude highlighted.

A custom sklearn mock (`_Ridge` implemented as closed-form WLS: `β = (XᵀWX + λI)⁻¹XᵀWy`) is injected into `sys.modules` before importing LIME, eliminating the scikit-learn dependency. The perturbed frame is replicated across all 6 temporal positions to adapt 2D LIME to the video model.

### 7.3 SHAP (Block 18) [SHAP]

`GradientExplainer` with 50 background images. `SHAPFrameWrapper` exposes vision encoder + global average pool + linear head. Shapley values estimated as `E[∇f(x) · (x - x_bg)]` over background samples.

OOM fallback: `output.sum().backward()` → `input.grad.abs().mean(0)` (gradient saliency) if full SHAP exhausts VRAM.

### 7.4 PDP + ICE + Validation Comparison (Block 19) [PDP]

Brightness sweep `α ∈ linspace(0.2, 1.8, 20)` applied to video tensor. PDPs (mean predicted action over samples) and ICE curves (per-sample predictions) test whether the model relies on photometric intensity statistics rather than semantic spatial features.

Validation improvement bar chart: `baseline=ln(256)` vs. trained model loss, visually communicating the magnitude of learning.

### 7.5 Novel XAI Fusions (Block 20) ⚠️ Preliminary

**ShaGCAM:** `normalise(gcam_norm × abs(shap_norm))`. Retains only pixels simultaneously gradient-discriminative (GradCAM) and Shapley-attributed (SHAP).

**LimeCAM:** `normalise(sqrt(gcam_norm × (lime_norm + 0.05)))`. Geometric mean → logical AND under perturbation-based (LIME) and gradient-based (GradCAM) attribution. `+0.05` prevents collapse at sparse LIME masks.

> ⚠️ These fusion methods are under active analysis. Visual results and interpretation will be revised.

---

## 8. Hardware Configuration

| Notebook | Model | Min VRAM | Recommended | Epochs/hr | Status |
|---|---|---|---|---|---|
| nb3 | ConvNeXt + BoW + ConvTL | 6 GB | 8 GB | ~10–15 | Trained |
| nb4 | EfficientLite + CLIP + HierTL | 6 GB | 8 GB | ~8–12 | Trained |
| nb1 | MaxViT-Base + DistilBERT + CrossAttn | 8 GB | 12 GB | ~3–5 | Trained |
| nb2 | MaxViT-Base + DistilBERT+VQA + CoT | 8 GB | 12 GB | ~3–5 | Trained |
| nb6 | MobileViT + CLIP + Perceiver | 8 GB | 12 GB | ~4–6 | Trained |
| nb5 | MaxViT-Large + BERT-ft2 + SelfAttn | 12 GB | 16 GB | **~0.4 (2.5 hr/epoch)** | **Not trained** |
| Production | MaxViT-Base + DistilBERT (DROID) | 8 GB | 12 GB | ~5–8 | Trained (partial) |

Notebook 5 has not been trained due to hardware limitations. At `embed_dim=512`, 64 token-learner tokens, and 8 transformer layers, each epoch requires approximately 2.5 hours on a 12 GB GPU with gradient checkpointing — approximately 200 GPU-hours for the full 80-epoch run. The architecture is fully implemented and verified via forward-pass sanity check; trained weights are unavailable.

**Software requirements:** Python ≥ 3.9, PyTorch ≥ 2.0, CUDA ≥ 11.8, ~50 GB disk, internet access for first-run dataset and model weight downloads.
