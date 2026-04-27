# Implementation Details and Hyperparameters

## Step-by-Step Implementation Details

The system implements a Reinforcement Learning pipeline modeled to generate video summaries through a bi-level (Horizontal & Vertical) policy approach, updating via the REINFORCE algorithm.

1. **Initialization & Data Loading**:
   - The process starts in `train.py` where a custom `Trainer` sets up the RL models (`HorizontalPolicy`, `VerticalPolicy`) and the `FeaturePipeline` (Transformer encoder).
   - Pre-computed video frame features (like CLIP embeddings) are loaded by category. 
2. **State & Environment Initialization**:
   - For every video, the sequence of frame features is fed to the FeaturePipeline to obtain contextual features.
   - An environment `VideoSummarizationEnv` is initialized. By default, it randomly selects a starting summary (a subset of indices, length controlled by `alpha`) and randomly picks a starting `anchor_idx`.
3. **Sequential Policy Execution (Episode)**:
   - The environment runs an episode that alternates between two steps (H and V).
   - **Horizontal Step (H-step)**: When it's turn 'H', the `HorizontalPolicy` receives the features of the *current summary frames*. It outputs a probability distribution to pick an *anchor frame* from within that summary.
   - **Vertical Step (V-step)**: When it's turn 'V', the `VerticalPolicy` looks at the chosen anchor frame and examines frames in a local temporal window (controlled by `window_size`). It decides whether to swap the anchor with a neighboring frame or keep it.
4. **Reward Evaluation**: 
   - After the V-step (where physical changes to the summary happen), the total reward is computed. The reward evaluates how representative and diverse the current summary is compared to the entire video.
5. **Termination**:
   - An episode terminates either when a strict step budget (`max_steps`) is reached or when the summary stops changing for a certain number of steps (`stability_patience`).
6. **Policy Update (REINFORCE algorithm)**:
   - After an episode ends, `train_on_video` calculates the discounted returns for each step in the trajectory.
   - A standard baseline (mean of returns) is deducted to reduce variance.
   - The gradients are computed for the negative log-probabilities of actions multiplied by their respective baseline-adjusted returns, and the optimizer updates the models.

---

## Important Functions and Their Usage

Here are the core functions driving the pipeline:

* **`train_on_video` (`trainer.py`)** 
  Executes a full RL episode for a single video. It handles the `while not done:` loop, interrogates the 'H' and 'V' policies based on the `obs['turn']`, tracks rewards and log probabilities, and finally computes the REINFORCE loss to update the policies and the transformer encoder simultaneously.

* **`VideoSummarizationEnv.step_H` and `step_V` (`MDP/environment.py`)**
  Control state transitions. `step_H` merely updates the state's `anchor_idx` selected by the Horizontal Policy without directly changing the summary or fetching a reward. `step_V` updates the actual summary if the Vertical Policy selects a new neighbor, tracks patience (early stopping), increments the step counter, computes the reward for the new state, and flips the turn back to 'H'.

* **`Reward.compute_total_reward` (`MDP/reward.py`)**
  Computes the immediate reward for a specific summary layout. It combines two metrics based on Cosine Similarity:
  - `compute_diversity_reward`: Encourages keeping visually distinct frames in the summary to avoid redundancy (penalizing high similarity between selected frames).
  - `compute_representative_reward`: Encourages picking frames that best represent the overall video (maximizing the similarity of all video frames to the selected summary).

* **`HorizontalPolicy.select_anchor` (`MDP/horizontal_policy.py`)**
  Takes only the frames currently in the summary and outputs a probability distribution to select which one will act as the "anchor" to be optimized or replaced in the subsequent V-step.

* **`VerticalPolicy.select_neighbor` (`MDP/vertical_policy.py`)**
  Given an anchor index, this restricts the candidate pool to a local subset `[anchor - window_size, anchor + window_size]`. It evaluates those neighbors (plus the anchor itself) with its neural network to sample a replacement frame.

---

## Hyperparameters Controlling the Training

These parameters (defined in `training/config.py`) dictate model architecture, RL behavior, and loss shaping:

### Transformer Encoder Architecture
* `transformer_input_dim`: Expected input feature dimension (e.g., 768 for CLIP ViT-L/14).
* `transformer_d_model` (512), `transformer_nhead` (8), `transformer_num_layers` (3), `transformer_dim_feedfwd` (2048): Define the size and capacity of the self-attention transformer pipeline.
* `transformer_dropout` (0.1): Regularization.
* `transformer_max_len`: The maximum video length the positional encoding supports.

### Policy Architecture
* `d_model` (512): Must match the output dimension of the transformer.
* `hidden_size` (256): Number of hidden units inside the simple feedford networks (MLPs) defining the H and V policies.

### Episode & Summary Control (Crucial for RL)
* `alpha` (0.02): Determines the initial target fraction of frames to select for the summary (e.g., summary size is `k = alpha * total_video_frames`). 
* `min_k` (3) / `max_k` (20): Clamps the absolute summary length so it doesn't get too small on very short videos or too large on very long ones.
* `window_size` (5): The temporal neighborhood span the V-policy can search within to replace the anchor. (e.g. `anchor ± window_size`).
* `max_steps` (100): Prevents the episode from running infinitely by capping the maximum H/V interactions per episode.
* `stability_patience` (5): Triggers early termination if the V-policy decides not to swap frames for this many consecutive turns.

### Reward Weights
* `w_div` (0.5) & `w_rep` (0.5): Balances the trade-off in the reward function between how diverse the summary is versus how well it represents the original video context.

### Optimization
* `lr` (1e-3): Learning rate for the optimizers.
* `gamma` (1.0): The discount factor for calculating the discounted sum of rewards (returns) in the REINFORCE algorithm. No discounting here means all future steps are valued equally.
