# Project Details: RL-Video-Summarization

## Project Purpose
The project implements a **reinforcement learning (RL) framework for unsupervised video summarization**. The goal is to learn how to select a compact, representative, and diverse set of key frames from a video without relying on any ground-truth labels. It achieves this by treating the process of constructing and refining the summary as a sequential decision-making problem.

## The Pipeline
The processing pipeline operates in three main stages:
1. **Raw Feature Extraction**: Raw video frames are independently encoded into features using a pre-trained visual model (CLIP). 
2. **Temporal Contextualization**: The independently extracted features are passed through a Temporal Transformer Encoder to capture the temporal dependencies and context of the sequence (what happens before and after each frame).
3. **RL Refinement**: An RL agent iteratively refines a working summary of key frames. It alternates between two synergistic policies:
   - **Horizontal Policy (H-Policy)**: Chooses a frame from the current summary to act as an "anchor".
   - **Vertical Policy (V-Policy)**: Explores the chronological neighborhood of the chosen anchor to see if a nearby frame would make a better replacement.

## RL Formulation (MDP)
The video summarization is formalized as a Markov Decision Process (MDP):
- **State ($s_t$)**: A set of currently selected frame indices ($\mathcal{S}$) combined with a designated *anchor* index ($a \in \mathcal{S}$).
- **Action**: At alternating turns, the action is either choosing a new anchor from $\mathcal{S}$ (H-Policy) or choosing a replacement candidate from the local temporal window of the anchor (V-Policy).
- **Reward ($r_t$)**: The reward evaluates the quality of the current summary features based on two metrics:
  - **Diversity**: Encourages the selected frames to be visually dissimilar to avoid redundancy. Calculated using the inverse of the pairwise cosine similarity of the selected frames.
  - **Representativeness**: Ensures the sequence covers the whole video effectively. Calculated by measuring the maximum cosine similarity between every frame in the video and the frames in the summary.
  - The total reward is a weighted sum of these two scores ($w_{div} \cdot r_{div} + w_{rep} \cdot r_{rep}$). H-steps yield a reward of 0, and the reward is strictly computed after V-steps.
- **Terminal Condition**: The episode ends either when a maximum number of V-steps is reached or if the summary remains unchanged for a predetermined number of V-steps (stability patience).

## Policies
Both policies are feedforward neural networks (Linear $\rightarrow$ ReLU $\rightarrow$ Linear $\rightarrow$ ReLU $\rightarrow$ Linear) trained using the REINFORCE algorithm:
- **Horizontal Policy (H-Policy)**: takes the contextual features of the current summary frames and outputs a probability distribution. It essentially learns to pinpoint the "weakest" candidate within the summary to be reconsidered as the anchor.
- **Vertical Policy (V-Policy)**: takes the contextual features of the anchor and its temporal window (e.g., ±5 frames) and produces a probability distribution over the candidates. It conducts a local search to substitute the anchor with a more suitable adjacent frame, or decides to keep the anchor itself.

## Feature Extraction Pipeline
1. **CLIP ViT-L/14 Embeddings**: Raw video frames are pre-processed offline using OpenAI's CLIP (`ViT-L/14`). This creates semantically rich **768-dimensional** frame-independent embeddings. These are cached as `.npz` files organized into categories like Activity, Animals, and Incidents.
2. **Temporal Transformer Encoder**: Because the CLIP embeddings are frame-independent, they are projected via a Linear layer (768 $\rightarrow$ 512 dimensions), combined with sinusoidal positional encodings, and passed through a 3-layer Transformer Encoder (8 attention heads, pre-norm). This yields **512-dimensional contextual features** per frame, which are directly utilized by the RL environment and policies. 

## Implementation Details
- **End-to-End Training**: The framework trains the H-Policy, V-Policy, and the Temporal Transformer Encoder jointly. REINFORCE (policy gradient) is utilized to compute the loss with a baseline, which is backpropagated to update all three components via separate Adam optimizers.
- **Code Organization**: The project is cleanly separated into modular components: `MDP/` contains the RL environment, policies, states, and rewards; `transformer/` defines the temporal self-attention components; `training/` includes pipelines, data loaders, and main training loops; and `train.py` serves as the end-to-end entry point.
