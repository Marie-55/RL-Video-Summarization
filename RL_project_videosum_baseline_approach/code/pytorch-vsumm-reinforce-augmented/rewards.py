import torch

def compute_reward(seq, actions, ignore_far_sim=True, temp_dist_thre=20, use_gpu=False):
    """Compute diversity and representativeness reward for REINFORCE.
    
    Args:
        seq: video feature sequence (1, seq_len, dim)
        actions: binary action vector (1, seq_len, 1)
        ignore_far_sim: whether to ignore distant frame pairs
        temp_dist_thre: temporal distance threshold for ignoring pairs
        use_gpu: whether to use GPU
    
    Returns:
        reward: scalar tensor (0 to 1)
    """
    _seq = seq.detach()
    _actions = actions.detach()
    pick_idxs = _actions.squeeze().nonzero().squeeze()
    num_picks = len(pick_idxs) if pick_idxs.ndimension() > 0 else 1
    
    # No frames selected
    if num_picks == 0:
        reward = torch.tensor(0.)
        if use_gpu: reward = reward.cuda()
        return reward

    _seq = _seq.squeeze()
    n = _seq.size(0)

    # Diversity reward: measure dissimilarity between selected frames
    if num_picks == 1:
        reward_div = torch.tensor(0.)
        if use_gpu: reward_div = reward_div.cuda()
    else:
        normed_seq = _seq / _seq.norm(p=2, dim=1, keepdim=True)
        dissim_mat = 1. - torch.matmul(normed_seq, normed_seq.t())
        dissim_submat = dissim_mat[pick_idxs,:][:,pick_idxs]
        if ignore_far_sim:
            pick_mat = pick_idxs.expand(num_picks, num_picks)
            temp_dist_mat = torch.abs(pick_mat - pick_mat.t())
            dissim_submat[temp_dist_mat > temp_dist_thre] = 1.
        reward_div = dissim_submat.sum() / (num_picks * (num_picks - 1.))

    # Representativeness reward: measure coverage of all frames
    dist_mat = torch.pow(_seq, 2).sum(dim=1, keepdim=True).expand(n, n)
    dist_mat = dist_mat + dist_mat.t()
    dist_mat.addmm_(1, -2, _seq, _seq.t())
    
    # CRITICAL FIX: Handle scalar pick_idxs (single frame selected)
    if pick_idxs.ndimension() == 0:
        pick_idxs = pick_idxs.unsqueeze(0)
    
    dist_mat = dist_mat[:,pick_idxs]
    dist_mat = dist_mat.min(1, keepdim=True)[0]
    reward_rep = torch.exp(-dist_mat.mean())

    # Combined reward
    reward = (reward_div + reward_rep) * 0.5

    return reward
