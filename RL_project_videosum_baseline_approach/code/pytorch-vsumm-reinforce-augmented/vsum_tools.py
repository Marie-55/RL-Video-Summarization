import numpy as np
import math

def knapsack_dp(values, weights, n_items, limit):
    """Solve knapsack problem via dynamic programming."""
    dp = np.zeros((n_items + 1, limit + 1))
    for i in range(1, n_items + 1):
        for w in range(1, limit + 1):
            if weights[i-1] <= w:
                dp[i, w] = max(values[i-1] + dp[i-1, w - weights[i-1]], dp[i-1, w])
            else:
                dp[i, w] = dp[i-1, w]
    
    # Backtrack to get selected items
    picks = []
    w = limit
    for i in range(n_items, 0, -1):
        if dp[i, w] != dp[i-1, w]:
            picks.append(i - 1)
            w -= weights[i-1]
    return picks[::-1]

def generate_summary(ypred, cps, n_frames, nfps, positions, proportion=0.15, method='knapsack'):
    """Generate keyshot-based video summary.
    
    Args:
        ypred: predicted importance scores (seq_len,)
        cps: change points, 2D array with shape (n_segments, 2), each row [start, end]
        n_frames: original number of frames
        nfps: number of frames per segment
        positions: positions of subsampled frames in original video
        proportion: target summary length as proportion of video
        method: 'knapsack' or 'rank'
    
    Returns:
        machine_summary: binary vector (n_frames,) indicating selected frames
    """
    n_segs = cps.shape[0]
    frame_scores = np.zeros((n_frames,), dtype=np.float32)
    
    if positions.dtype != int:
        positions = positions.astype(np.int32)
    if positions[-1] != n_frames:
        positions = np.concatenate([positions, [n_frames]])
    
    # Map prediction scores to frame scores
    for i in range(len(positions) - 1):
        pos_left, pos_right = positions[i], positions[i+1]
        if i >= len(ypred):  # FIX: handle boundary correctly
            frame_scores[pos_left:pos_right] = 0
        else:
            frame_scores[pos_left:pos_right] = ypred[i]

    # Compute segment-level scores
    seg_score = []
    for seg_idx in range(n_segs):
        start, end = int(cps[seg_idx, 0]), int(cps[seg_idx, 1] + 1)
        scores = frame_scores[start:end]
        seg_score.append(float(scores.mean()))

    limits = int(math.floor(n_frames * proportion))

    # Select segments
    if method == 'knapsack':
        picks = knapsack_dp(seg_score, nfps, n_segs, limits)
    elif method == 'rank':
        order = np.argsort(seg_score)[::-1].tolist()
        picks = []
        total_len = 0
        for i in order:
            if total_len + nfps[i] < limits:
                picks.append(i)
                total_len += nfps[i]
    else:
        raise KeyError("Unknown method {}".format(method))

    # Create binary summary vector
    machine_summary = np.zeros((n_frames,), dtype=np.float32)
    for seg_idx in picks:
        start, end = int(cps[seg_idx, 0]), int(cps[seg_idx, 1] + 1)
        machine_summary[start:end] = 1

    return machine_summary

def evaluate_summary(machine_summary, user_summary, eval_metric='avg'):
    """Evaluate summary against ground truth.
    
    Args:
        machine_summary: binary vector (n_frames,)
        user_summary: ground truth binary matrix (n_users, n_frames)
        eval_metric: 'avg' (TVSum) or 'max' (SumMe)
    
    Returns:
        f_score: F-score
        precision: precision
        recall: recall
    """
    n_frames = len(machine_summary)
    
    # Compute F-score for each user
    f_scores = []
    for user_idx in range(user_summary.shape[0]):
        gt = user_summary[user_idx, :]
        
        # TP, FP, FN
        tp = np.sum(machine_summary * gt)
        fp = np.sum(machine_summary * (1 - gt))
        fn = np.sum((1 - machine_summary) * gt)
        
        # Precision and Recall
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        # F-score
        f = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        f_scores.append(f)
    
    # Aggregate F-scores
    if eval_metric == 'avg':
        f_score = np.mean(f_scores)
    elif eval_metric == 'max':
        f_score = np.max(f_scores)
    else:
        f_score = np.mean(f_scores)
    
    return f_score, np.mean([2*s/(1+s) if s > 0 else 0 for s in f_scores]), np.mean(f_scores)
