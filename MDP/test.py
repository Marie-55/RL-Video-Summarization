# test.py
import numpy as np
import torch
import torch.optim as optim
from frame import Frame
from state import State
from reward import Reward
from horizontal_policy import HorizontalPolicy
from vertical_policy import VerticalPolicy

def test_framework():
    print("=" * 50)
    print("Testing PRLVS Framework")
    print("=" * 50)

    FEATURE_DIM  = 64   
    VIDEO_LENGTH = 50    
    SUMMARY_SIZE = 5    
    HIDDEN_SIZE  = 128
    WINDOW_SIZE  = 3   
    T            = 10   
    GAMMA        = 0.99  
    print("\n[1] Building fake video...")
    video = {
        i: Frame(data=np.random.randn(FEATURE_DIM), index=i)
        for i in range(VIDEO_LENGTH)
    }
    print(f"    Video length: {VIDEO_LENGTH} frames, feature dim: {FEATURE_DIM}")


    print("\n[2] Initializing summary...")
    summary_indices = np.linspace(0, VIDEO_LENGTH - 1, SUMMARY_SIZE, dtype=int)
    summary = {i: video[i] for i in summary_indices}
    state = State(frames=summary)
    print(f"    Summary frame indices: {list(state.frames.keys())}")

   
    print("\n[3] Initializing policies...")
    horizontal_policy = HorizontalPolicy(
        input_size=FEATURE_DIM,
        hidden_size=HIDDEN_SIZE
    )
    vertical_policy = VerticalPolicy(
        input_size=FEATURE_DIM,
        hidden_size=HIDDEN_SIZE
    )
    h_optimizer = optim.Adam(horizontal_policy.parameters(), lr=1e-3)
    v_optimizer = optim.Adam(vertical_policy.parameters(),   lr=1e-3)
    print("    Both policies initialized.")

   
    print("\n[4] Running episode...")
    log_probs_h = []
    log_probs_v = []
    rewards     = []

    for step in range(T):
       
        anchor, log_prob_h = horizontal_policy.select_anchor(state)
        log_probs_h.append(log_prob_h)
        reward_before = Reward(state, video).compute_total_reward()
        new_frame, log_prob_v = vertical_policy.select_neighbor(
            anchor, video, window_size=WINDOW_SIZE
        )
        if log_prob_v is not None:
            state.replace_frame(anchor.index, new_frame)
            log_probs_v.append(log_prob_v)
        else:
            log_probs_v.append(torch.tensor(0.0, requires_grad=True))

        reward_after = Reward(state, video).compute_total_reward()
        r_t = reward_after - reward_before
        rewards.append(r_t)

        print(f"    Step {step + 1:02d} | anchor: {anchor.index:3d} "
              f"→ new: {new_frame.index:3d} | "
              f"r_t: {r_t:+.4f} | reward: {reward_after:.4f}")

    print("\n[5] Computing discounted returns...")
    returns = []
    G = 0.0
    for r in reversed(rewards):
        G = r + GAMMA * G
        returns.insert(0, G)
    print(f"    Returns range: [{min(returns):.4f}, {max(returns):.4f}]")

    print("\n[6] Updating policies...")
    horizontal_policy.update_policy(log_probs_h, returns, h_optimizer)
    vertical_policy.update_policy(log_probs_v,   returns, v_optimizer)
    print("    Both policies updated successfully.")


    print("\n[7] Final summary frame indices:", list(state.frames.keys()))
    final_reward = Reward(state, video).compute_total_reward()
    print(f"    Final total reward: {final_reward:.4f}")

    print("\n" + "=" * 50)
    print("All tests passed!")
    print("=" * 50)


if __name__ == "__main__":
    test_framework()