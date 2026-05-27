# Augmented Protocol: Unsupervised Video Summarization

## Overview

This folder implements an **unsupervised REINFORCE-based video summarization** pipeline optimized for the augmented protocol. Key differences from the original repo:

- **No ground truth required during training** – Rewards computed purely from diversity and representativeness
- **Trains on custom NPZ embeddings** – Direct loading from category folders
- **Tests on full datasets** – TVSum (50 videos) + SumMe (46 videos) with complete ground truth evaluation
- **3 embedding models** – DINOv2, CLIP RN50x64, CLIP ViT-L14
- **Modular architecture** – Clean separation of concerns

## Architecture

### Core Components

**models.py**
- `DSN` class: LSTM-based Deep Summarization Network
- Configurable input dimension (1024 or 768)
- Bidirectional attention-based encoding

**rewards.py**
- `compute_reward()`: Diversity + Representativeness unsupervised reward
- Handles both batch and scalar frame selections
- Normalized to [0, 1] range

**vsum_tools.py**
- `knapsack_dp()`: Dynamic programming for segment selection
- `generate_summary()`: Frame predictions → binary video summary
- `evaluate_summary()`: F-score computation (avg for TVSum, max for SumMe)
- **Critical fix**: Line 20 uses `>=` to catch all boundary violations

**utils.py**
- Logger, JSON I/O, model checkpointing

**main_augmented.py**
- CLI interface supporting:
  - `--mode train`: Load NPZ → Train REINFORCE
  - `--mode evaluate`: Load H5 → Compute F1-scores

## Hyperparameters

```python
MAX_EPOCHS = 60
LEARNING_RATE = 1e-5
WEIGHT_DECAY = 1e-5
LR_DECAY_STEP = 30
LR_DECAY_GAMMA = 0.1
HIDDEN_DIM = 256
NUM_EPISODES = 5  # per video per epoch
```

## Training Strategy

**Unsupervised REINFORCE**:
1. For each video, sample frame selections from policy
2. Compute reward: `(diversity_score + representativeness_score) / 2`
3. Compute expected reward with baseline subtraction
4. Update policy via REINFORCE gradient estimator
5. Update baseline (exponential moving average)

**No ground truth needed** – Only diversity and representativeness metrics.

## Datasets

### Training
- **Custom Dataset**: 129 videos in NPZ format
- **Embeddings**: 3 categories (Activity, Animals, Incidents)
- **Models**: DINOv2 ViT-L14, CLIP RN50x64, CLIP ViT-L14

### Testing
- **TVSum**: 50 videos, metric='avg'
- **SumMe**: 46 videos, metric='max'
- **Format**: H5 with human annotations from 5+ users

## Fixed Issues

1. **vsum_tools.py Line 20**: Changed `if i == len(ypred):` → `if i >= len(ypred):`
   - Prevents IndexError when positions array extends beyond predictions
   
2. **rewards.py Scalar Handling**: Added check for 0D tensor selection
   - When only 1 frame selected, properly unsqueeze before indexing

3. **Input Dimension Parameter**: Propagated through entire pipeline
   - DINOv2, CLIP RN50x64: 1024-dim
   - CLIP ViT-L14: 768-dim

## Usage

### Direct Training (main_augmented.py)

```bash
python main_augmented.py \
    --mode train \
    --input-dim 1024 \
    --train-npz-dir /path/to/custom/data \
    --max-epoch 60 \
    --save-dir ./outputs \
    --gpu 0
```

### Direct Evaluation

```bash
python main_augmented.py \
    --mode evaluate \
    --input-dim 1024 \
    --test-h5 /path/to/tvsum.h5 \
    --test-metric avg \
    --save-dir ./outputs \
    --gpu 0
```

### Via Standalone Notebook

See `videosum-augmented-standalone.ipynb` for complete end-to-end pipeline:
1. Load NPZ training data
2. Train all 3 embeddings (60 epochs each)
3. Evaluate on TVSum & SumMe
4. Generate 5 comprehensive plots
5. Export results to JSON & ZIP

## Output

**Plots Generated**:
1. **Training rewards**: Per-embedding + average curves with confidence bands
2. **Comparative F-scores**: Bar chart (TVSum blue, SumMe green)
3-5. **Per-embedding analysis**: Reward history + F-score distributions (3 separate figures)

**Results**:
- `augmented_results.json`: Metrics summary
- `augmented_results.zip`: All plots + JSON

## Expected Performance

Approximate ranges from 129-video custom dataset training:

| Embedding | TVSum | SumMe |
|-----------|-------|-------|
| DINOv2 | 45-50% | 42-48% |
| CLIP RN50x64 | 48-55% | 45-52% |
| CLIP ViT-L14 | 42-48% | 40-46% |

Note: Exact performance depends on custom training data characteristics.

## Key Differences from Original pytorch-vsumm-reinforce/

| Aspect | Original | Augmented |
|--------|----------|-----------|
| **Training** | Supervised (SumMe/TVSum) | Unsupervised (custom data) |
| **Ground truth during training** | Required | Not used |
| **Reward computation** | Custom per-dataset | Diversity + Representativeness |
| **Input format** | H5 only | NPZ + H5 |
| **Evaluation** | Per-split sampling | Full dataset (50+46 videos) |
| **Input dimensions** | Fixed 1024 | Configurable (1024/768) |

## Troubleshooting

**IndexError in vsum_tools**:
- Ensure vsum_tools.py line 20 uses `>=` not `==`

**Dimension mismatch error**:
- Verify `--input-dim` matches embedding model:
  - DINOv2, CLIP RN50x64 → 1024
  - CLIP ViT-L14 → 768

**OOM errors**:
- Reduce batch size or use smaller hidden dimension
- Process videos serially instead of batched

**Low F-scores**:
- Check if custom training data is representative
- Validate H5 schema (must include user_summary with proper dtype)

---

**Last updated**: 2024
**Author**: Custom augmented implementation for unsupervised video summarization
