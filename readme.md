# RL Video Summarization

A reinforcement learning framework for unsupervised video summarization. The system learns to select a compact, representative, and diverse set of key frames from a video — without any ground-truth labels — by treating summary construction as a sequential decision-making problem.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Dataset & Video Embeddings](#dataset--video-embeddings)
3. [Transformer: Temporal Encoding](#transformer-temporal-encoding)
4. [MDP Formulation](#mdp-formulation)
5. [Policies](#policies)
6. [Reward Function](#reward-function)
7. [End-to-End Training](#end-to-end-training)
8. [Project Structure](#project-structure)
9. [Configuration](#configuration)
10. [Usage](#usage)

---

## 1. Project Overview

The pipeline operates in three stages:

```
Raw Video Frames
      │
      ▼
CLIP ViT-L/14 Embeddings  (768-dim per frame, pre-extracted)
      │
      ▼
Temporal Transformer Encoder  →  contextual features (512-dim per frame)
      │
      ▼
RL Environment (MDP)
      │
 ┌────┴────┐
 H-Policy  V-Policy   ←  REINFORCE updates
 └────┬────┘
      │
      ▼
  Final Summary  (set of key frame indices)
```

The RL agent iteratively refines a working summary by alternating between two complementary policies: one selects a frame from the current summary as an "anchor," and the other explores the video neighborhood to find a better replacement.

---

## 2. Dataset & Video Embeddings

### Source Data

Videos span three thematic categories stored under `videos/embeddings_clip_vitl14/`:

| Category | Description |
|---|---|
| **Activity** | Human activities — sports, cooking, crafts, dance (~100 videos) |
| **Animals** | Wildlife footage — elephants, lions, hippos, etc. (~9 videos) |
| **Incidents** | Natural disasters and accidents — avalanches, wildfires, floods, etc. (~50 videos) |

### CLIP ViT-L/14 Embeddings

Each video is pre-processed into a sequence of **768-dimensional embeddings** using [OpenAI CLIP](https://github.com/openai/CLIP) with the `ViT-L/14` backbone. These are stored as `.npz` files with the key `embeddings`, giving a matrix of shape `(T, 768)` where `T` is the number of frames.

CLIP's visual encoder was chosen because:
- It produces **semantically rich, transferable representations** trained on 400M image-text pairs.
- Frames with similar visual content cluster together in CLIP space, making cosine similarity a meaningful proxy for visual redundancy — a property directly exploited by the reward function.
- Embeddings are computed once and cached, making training fast.

The `VideoEmbeddingLoader` handles discovery and loading of these files, with automatic fallback to common HPC paths and support for the `VIDEOS_DIR` environment variable.

---

## 3. Transformer: Temporal Encoding

Raw CLIP embeddings are **frame-independent** — each frame is encoded in isolation by CLIP. To capture temporal context (what happens before and after a frame), the embeddings are passed through a **Temporal Transformer Encoder** before being fed to the RL policies.

### Architecture

```
Input: (B, T, 768)  ← raw CLIP embeddings
  │
  ├─ InputProjection: Linear(768→512) + LayerNorm + ReLU
  │
  ├─ PositionalEncoding: x = x + sinusoidal(position)
  │
  ├─ TransformerEncoderLayer × 3
  │     └─ Pre-norm multi-head self-attention (8 heads, 64 dim/head)
  │     └─ Feed-Forward Network (hidden: 2048)
  │     └─ Dropout (0.1)
  │
Output: (B, T, 512)  ← contextual features
```

Key design choices:

- **Pre-norm** (`norm_first=True`): Applies LayerNorm before the attention and FFN sub-layers, which stabilizes training.
- **Sinusoidal positional encoding**: Injects frame-order information without learned parameters, supporting arbitrary sequence lengths.
- **Padding mask support**: Allows variable-length videos to be batched efficiently; padded positions are masked out in attention.
- The encoder is implemented in `transformer/transformer_encoder.py` and managed by `training/feature_pipeline.py`, which exposes separate `training=True/False` modes — returning a gradient-attached Tensor for end-to-end updates or a detached NumPy array for inference.

---

## 4. MDP Formulation

Video summarization is cast as a **Markov Decision Process** over summary refinement steps.

### Components

| MDP Element | Definition |
|---|---|
| **State** $s_t$ | A set of selected frame indices $\mathcal{S} \subseteq \{0, \ldots, T-1\}$ plus a designated *anchor* index $a \in \mathcal{S}$ |
| **Action** | Chosen by either the H-policy (new anchor) or V-policy (replacement frame) |
| **Transition** | V-step: anchor is replaced by the chosen neighbor (if different), updating $\mathcal{S}$; H-step: only the anchor pointer changes |
| **Reward** $r_t$ | Diversity + representativeness of the current summary (see below); H-steps yield $r = 0$ |
| **Terminal Condition** | `max_steps` V-steps reached **or** `stability_patience` consecutive V-steps with no change |

### Episode Flow

1. **Reset**: The environment samples an initial summary of size $k = \text{clip}(\lfloor \alpha \cdot T \rfloor, k_{\min}, k_{\max})$ frames uniformly at random, and picks a random anchor from that set.
2. **H-turn**: The H-policy receives features of all frames currently in $\mathcal{S}$ and selects a new anchor.
3. **V-turn**: The V-policy receives features of the anchor's temporal neighborhood and selects a replacement (or keeps the anchor). If the summary changes, the patience counter resets; otherwise it increments.
4. Steps 2–3 alternate until termination.

### State Representation (`MDP/state.py`, `MDP/frame.py`)

`State` wraps the selected index set and the anchor pointer, providing `replace_anchor(new_idx)` and `get_sorted_indices()`. `Frame` is a thin wrapper around frame embeddings supporting arithmetic operations for potential future reward extensions.

---

## 5. Policies

Both policies are **feedforward neural networks** trained jointly with REINFORCE. They share the same architecture but serve complementary roles.

### Horizontal Policy (H-Policy) — `MDP/horizontal_policy.py`

**Role**: Given the frames currently in the summary, choose which one to focus on next (the anchor).

**Input**: Contextual features of the $k$ frames in $\mathcal{S}$ — shape `(k, 512)`.

**Output**: A scalar score per frame via:
```
Linear(512 → 256) → ReLU → Linear(256 → 128) → ReLU → Linear(128 → 1)
```
Scores are converted to a categorical distribution; a frame is sampled as the new anchor.

**Intuition**: The H-policy learns to identify which currently selected frame is the weakest candidate — the one most worth reconsidering — and designates it as the anchor for the V-policy to potentially replace.

### Vertical Policy (V-Policy) — `MDP/vertical_policy.py`

**Role**: Given the anchor frame, explore a local temporal window and decide whether to replace the anchor with a nearby frame.

**Input**: Contextual features of candidate frames = `{anchor} ∪ neighbors_within_window` — shape `(2·window + 1, 512)` at most.

**Output**: Same feedforward architecture as H-policy, producing scores over candidates including the "keep anchor" option.

**Intuition**: The V-policy performs local search — sliding the anchor to a temporally adjacent frame that better serves the summary (e.g., is more representative or reduces redundancy).

### Why Two Policies?

The H/V decomposition separates two distinct sub-problems:
- **Which frame to reconsider** (H): a selection problem over the current summary.
- **Where to move it** (V): a local search problem over the video timeline.

This factored structure reduces the action space at each step and makes the credit assignment problem more tractable.

---

## 6. Reward Function

The reward is computed at every V-step based on the full contextual feature matrix and the current summary indices (`MDP/reward.py`).

### Diversity Reward

Encourages selected frames to be visually dissimilar to each other:

$$r_{\text{div}} = 1 - \frac{1}{|\mathcal{S}|(|\mathcal{S}|-1)} \sum_{i \neq j} \cos(\mathbf{f}_i, \mathbf{f}_j)$$

High diversity rewards summaries where no two selected frames are redundant.

### Representativeness Reward

Ensures the summary covers the full video content:

$$r_{\text{rep}} = \frac{1}{T} \sum_{t=1}^{T} \max_{i \in \mathcal{S}} \cos(\mathbf{f}_t, \mathbf{f}_i)$$

This measures how well the video's most similar frame to the summary represents every frame in the video.

### Total Reward

$$r = w_{\text{div}} \cdot r_{\text{div}} + w_{\text{rep}} \cdot r_{\text{rep}}$$

Default: $w_{\text{div}} = w_{\text{rep}} = 0.5$. Both rewards use cosine similarity computed over the transformer's **contextual** features (not raw CLIP embeddings), so the reward signal is shaped by temporal context.

---

## 7. End-to-End Training

Training uses **episodic REINFORCE** (policy gradient) and updates all three components — the transformer encoder and both policies — jointly via a shared backward pass.

### Algorithm (per video, per epoch)

```
1. Encode video:  raw CLIP (T, 768)  →  transformer  →  ctx_features (T, 512)
2. Reset environment with ctx_features
3. Roll out episode:
     while not done:
         if H-turn:  anchor, log_p_H = H_policy(summary_features)
         if V-turn:  candidate, log_p_V = V_policy(anchor_neighborhood)
         obs, reward, done = env.step(action)
         collect (log_prob, reward)
4. Compute discounted returns G_t = Σ γ^k r_{t+k}
5. Subtract baseline b = mean(G_t)
6. Compute REINFORCE loss:
     L = -Σ_H log_p_H · (G_H - b)  -  Σ_V log_p_V · (G_V - b)
7. L.backward()  →  updates H-policy, V-policy, and transformer encoder
```

H-step and V-step returns are tracked by index (not assumed to strictly alternate) to handle early episode termination correctly.

### Optimizers

All three components use separate **Adam** optimizers (default lr = 1e-3), zeroed and stepped together after each episode:

```python
opt_h.step()       # H-policy parameters
opt_v.step()       # V-policy parameters
opt_transformer.step()  # Transformer encoder parameters
```

The transformer runs in `train()` mode during episodes, so gradients flow back through the encoding step — allowing the temporal features to adapt to the RL objective.

### Training Loop

```
for epoch in range(num_epochs):
    for category, videos in all_data.items():
        for video in videos:
            ctx = pipeline.encode_video(video, training=True)
            env = VideoSummarizationEnv(ctx, config)
            reward, summary = train_on_video(env, h_policy, v_policy, ...)
    save_checkpoint(epoch)   # saves H-policy, V-policy, transformer, stats
```

The best checkpoint (highest average reward) is saved separately as `best_model.pt`. Training statistics (per-epoch, per-category reward breakdown) are written to `checkpoints/training_stats.json`.

---

## 8. Project Structure

```
RL-Video-Summarization/
│
├── train.py                        # Main training entry point
│
├── transformer/
│   ├── transformer_encoder.py      # TemporalTransformerEncoder, PositionalEncoding, InputProjection
│   └── __init__.py
│
├── MDP/
│   ├── environment.py              # VideoSummarizationEnv (reset, step_H, step_V)
│   ├── state.py                    # State class (selected indices + anchor)
│   ├── frame.py                    # Frame abstraction
│   ├── reward.py                   # Reward (diversity + representativeness)
│   ├── horizontal_policy.py        # H-Policy network
│   └── vertical_policy.py          # V-Policy network
│
├── training/
│   ├── config.py                   # Hyperparameter dataclass
│   ├── data_loader.py              # VideoEmbeddingLoader (.npz → numpy)
│   ├── feature_pipeline.py         # FeaturePipeline (wraps transformer encode/decode)
│   └── trainer.py                  # train_on_video (REINFORCE update logic)
│
└── videos/
    └── embeddings_clip_vitl14/
        ├── Activity/               # (T, 768) .npz files
        ├── Animals/
        └── Incidents/
```

---

## 9. Configuration

All hyperparameters are defined in `training/config.py`:

| Parameter | Default | Description |
|---|---|---|
| `transformer_input_dim` | 768 | CLIP ViT-L/14 embedding dimension |
| `transformer_d_model` | 512 | Transformer hidden dimension |
| `transformer_nhead` | 8 | Attention heads |
| `transformer_num_layers` | 3 | Encoder depth |
| `transformer_dim_feedfwd` | 2048 | FFN hidden size (4× d_model) |
| `transformer_dropout` | 0.1 | Dropout rate |
| `alpha` | 0.02 | Initial summary size as fraction of video length |
| `min_k` / `max_k` | 3 / 20 | Summary size bounds |
| `window_size` | 5 | V-policy temporal search window (±5 frames) |
| `max_steps` | 100 | Max V-steps per episode |
| `stability_patience` | 5 | Early stop if no summary changes for 5 V-steps |
| `w_div` / `w_rep` | 0.5 / 0.5 | Reward weights |
| `lr` | 1e-3 | Learning rate (all optimizers) |
| `gamma` | 1.0 | REINFORCE discount factor |

---

## 10. Usage

### Training

```bash
# Basic training (10 epochs, auto device)
python train.py

# Custom epochs and video directory
python train.py --num_epochs 50 --videos_dir /path/to/embeddings

# Resume from checkpoint
python train.py --checkpoint checkpoints/checkpoint_epoch_005.pt

# Full options
python train.py --num_epochs 30 --seed 0 --device cuda --checkpoint_dir my_checkpoints
```

### Checkpoints

Each epoch saves `checkpoints/checkpoint_epoch_NNN.pt` containing the transformer, H-policy, V-policy state dicts, config, and per-epoch statistics. The highest-reward epoch is additionally saved as `checkpoints/best_model.pt`.