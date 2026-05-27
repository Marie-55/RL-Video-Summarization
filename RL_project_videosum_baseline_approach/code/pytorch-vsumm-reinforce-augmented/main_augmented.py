import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from torch.optim import lr_scheduler
from torch.distributions import Bernoulli
import h5py
import time
import datetime

from models import DSN
from rewards import compute_reward
import vsum_tools
from utils import Logger, read_json, write_json, save_checkpoint

parser = argparse.ArgumentParser("Augmented REINFORCE for unsupervised video summarization")

# Mode
parser.add_argument('--mode', type=str, default='train', choices=['train', 'evaluate'], 
                    help='train or evaluate')

# Training data (NPZ format)
parser.add_argument('--train-npz-dir', type=str, default='', 
                    help='directory containing NPZ embedding files (for training)')

# Test data (H5 format)
parser.add_argument('--test-h5', type=str, default='', 
                    help='path to H5 test dataset (TVSum or SumMe)')
parser.add_argument('--test-split-json', type=str, default='',
                    help='path to JSON file with train/test splits')
parser.add_argument('--split-id', type=int, default=0,
                    help='which split to use (default: 0)')
parser.add_argument('--test-metric', type=str, choices=['tvsum', 'summe'], default='summe',
                    help='evaluation metric (tvsum=avg, summe=max)')

# Model options
parser.add_argument('--input-dim', type=int, default=1024, 
                    help='input embedding dimension')
parser.add_argument('--hidden-dim', type=int, default=256,
                    help='LSTM hidden dimension')
parser.add_argument('--num-layers', type=int, default=1,
                    help='number of LSTM layers')

# Training options
parser.add_argument('--max-epoch', type=int, default=60,
                    help='maximum training epochs')
parser.add_argument('--lr', type=float, default=1e-5,
                    help='learning rate')
parser.add_argument('--weight-decay', type=float, default=1e-5,
                    help='weight decay')
parser.add_argument('--stepsize', type=int, default=30,
                    help='learning rate decay step size')
parser.add_argument('--gamma', type=float, default=0.1,
                    help='learning rate decay factor')
parser.add_argument('--num-episode', type=int, default=5,
                    help='number of REINFORCE episodes per video')
parser.add_argument('--beta', type=float, default=0.01,
                    help='summary length penalty weight')

# Misc
parser.add_argument('--save-dir', type=str, default='log',
                    help='output directory')
parser.add_argument('--resume', type=str, default='',
                    help='checkpoint to resume from')
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--gpu', type=str, default='0')
parser.add_argument('--use-cpu', action='store_true')

args = parser.parse_args()

torch.manual_seed(args.seed)
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
use_gpu = torch.cuda.is_available() and not args.use_cpu

os.makedirs(args.save_dir, exist_ok=True)

def load_npz_embeddings(npz_dir):
    """Load NPZ embedding files from directory."""
    videos = {}
    npz_dir = Path(npz_dir)
    for npz_file in sorted(npz_dir.glob('**/*.npz')):
        data = np.load(npz_file, allow_pickle=True)
        if 'features' in data:
            feat = data['features']
        elif 'embeddings' in data:
            feat = data['embeddings']
        else:
            feat = data[list(data.keys())[0]]
        videos[npz_file.stem] = feat
    return videos

def train(model, videos_dict, optimizer, scheduler, use_gpu):
    """Unsupervised training on NPZ embeddings."""
    sys.stdout = Logger(os.path.join(args.save_dir, 'log_train.txt'))
    print(f"=== Training on {len(videos_dict)} videos ===")
    
    model.train()
    baselines = {key: 0.0 for key in videos_dict.keys()}
    
    for epoch in range(args.max_epoch):
        epoch_rewards = []
        video_keys = list(videos_dict.keys())
        np.random.shuffle(video_keys)
        
        for video_key in video_keys:
            features = videos_dict[video_key]  # (seq_len, dim)
            seq = torch.from_numpy(features).unsqueeze(0).float()  # (1, seq_len, dim)
            if use_gpu:
                seq = seq.cuda()
            
            probs = model(seq)  # (1, seq_len, 1)
            
            # Summary length penalty
            cost = args.beta * (probs.mean() - 0.5) ** 2
            
            m = Bernoulli(probs)
            episode_rewards = []
            
            for _ in range(args.num_episode):
                actions = m.sample()
                log_probs = m.log_prob(actions)
                reward = compute_reward(seq, actions, use_gpu=use_gpu)
                expected_reward = log_probs.mean() * (reward - baselines[video_key])
                cost -= expected_reward
                episode_rewards.append(reward.item())
            
            optimizer.zero_grad()
            cost.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            
            baselines[video_key] = 0.9 * baselines[video_key] + 0.1 * np.mean(episode_rewards)
            epoch_rewards.append(np.mean(episode_rewards))
        
        epoch_reward = np.mean(epoch_rewards)
        print(f"epoch {epoch+1}/{args.max_epoch}\t reward {epoch_reward}")
        
        if scheduler is not None:
            scheduler.step()
    
    # Save model
    model_state = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
    save_checkpoint(model_state, os.path.join(args.save_dir, f'model_epoch{args.max_epoch}.pth.tar'))
    print(f"Model saved to {args.save_dir}")

def evaluate(model, dataset_h5, splits_json, metric, use_gpu):
    """Evaluate on H5 dataset (TVSum or SumMe)."""
    sys.stdout = Logger(os.path.join(args.save_dir, 'log_test.txt'))
    print(f"=== Evaluating on {metric.upper()} ===")
    
    model.eval()
    dataset = h5py.File(dataset_h5, 'r')
    splits = read_json(splits_json)
    split = splits[args.split_id]
    test_keys = split['test_keys']
    
    eval_metric = 'avg' if metric == 'tvsum' else 'max'
    f_scores = []
    
    with torch.no_grad():
        for video_key in test_keys:
            features = dataset[video_key]['features'][...]
            seq = torch.from_numpy(features).unsqueeze(0).float()
            if use_gpu:
                seq = seq.cuda()
            
            probs = model(seq)
            probs = probs.data.cpu().squeeze().numpy()
            
            cps = dataset[video_key]['change_points'][...]
            num_frames = dataset[video_key]['n_frames'][()]
            nfps = dataset[video_key]['n_frame_per_seg'][...].tolist()
            positions = dataset[video_key]['picks'][...]
            user_summary = dataset[video_key]['user_summary'][...]
            
            machine_summary = vsum_tools.generate_summary(probs, cps, num_frames, nfps, positions)
            f_score, _, _ = vsum_tools.evaluate_summary(machine_summary, user_summary, eval_metric)
            f_scores.append(f_score)
    
    mean_fscore = np.mean(f_scores)
    print(f"Average F-score {mean_fscore:.1%}")
    
    dataset.close()
    return mean_fscore

def main():
    print("==========")
    print("Args:", args)
    print("==========")
    
    if use_gpu:
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("Using CPU")
    
    # Initialize model
    model = DSN(in_dim=args.input_dim, hid_dim=args.hidden_dim, 
                num_layers=args.num_layers, cell='lstm')
    
    if use_gpu:
        model = nn.DataParallel(model).cuda()
    
    if args.mode == 'train':
        # Training mode
        videos_dict = load_npz_embeddings(args.train_npz_dir)
        print(f"Loaded {len(videos_dict)} videos")
        
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=args.stepsize, gamma=args.gamma) if args.stepsize > 0 else None
        
        if args.resume:
            print(f"Resuming from {args.resume}")
            checkpoint = torch.load(args.resume)
            if isinstance(model, nn.DataParallel):
                model.module.load_state_dict(checkpoint)
            else:
                model.load_state_dict(checkpoint)
        
        train(model, videos_dict, optimizer, scheduler, use_gpu)
    
    elif args.mode == 'evaluate':
        # Evaluation mode
        if args.resume:
            print(f"Loading from {args.resume}")
            checkpoint = torch.load(args.resume)
            if isinstance(model, nn.DataParallel):
                model.module.load_state_dict(checkpoint)
            else:
                model.load_state_dict(checkpoint)
        else:
            raise ValueError("Must provide --resume for evaluation")
        
        fscore = evaluate(model, args.test_h5, args.test_split_json, args.test_metric, use_gpu)

if __name__ == '__main__':
    main()
