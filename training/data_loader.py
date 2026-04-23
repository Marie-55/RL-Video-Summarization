"""
Data Loader for Video Embeddings.

Loads CLIP ViT-L/14 embeddings (768-dim) from .npz files in videos/embeddings_clip_vitl14/
"""

import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import random
import os


class VideoEmbeddingLoader:    
    def __init__(self, videos_dir: str = 'videos/embeddings_clip_vitl14'):
        """
        Args:
            videos_dir: Path to embeddings directory. Can be:
                        - Relative path (default: 'videos/embeddings_clip_vitl14')
                        - Absolute path
                        - Can be overridden by VIDEOS_DIR environment variable
        """
        # Check environment variable first (I added this for HPC environments)
        if 'VIDEOS_DIR' in os.environ:
            videos_dir = os.environ['VIDEOS_DIR']
            print(f"[VideoEmbeddingLoader] Using VIDEOS_DIR from environment: {videos_dir}")
        
        self.videos_dir = Path(videos_dir).expanduser().resolve()
        
        # Scan available categories
        self.categories = [d.name for d in self.videos_dir.iterdir() if d.is_dir()]
        self.categories.sort()
        
        print(f"[VideoEmbeddingLoader] Found {len(self.categories)} categories")
        print(f"[VideoEmbeddingLoader] Categories: {', '.join(self.categories)}")
        print(f"[VideoEmbeddingLoader] Videos directory: {self.videos_dir}")
    
    def get_categories(self) -> List[str]:
        """Get list of available categories."""
        return self.categories
    
    def list_videos_in_category(self, category: str) -> List[str]:
        """List all videos in a category."""
        category_dir = self.videos_dir / category
        if not category_dir.exists():
            raise ValueError(f"Category not found: {category}")
        
        videos = [f.name for f in category_dir.glob('*.npz')]
        videos.sort()
        return videos
    
    def load_single_video(self, video_path) -> np.ndarray:
        path = Path(video_path)
        
        if not path.is_absolute() and not path.exists():
            candidate = self.videos_dir / path
            if candidate.exists():
                path = candidate
        
        if not path.exists():
            raise FileNotFoundError(f"Video file not found: {path}")
        
        data = np.load(path)
        embeddings = data['embeddings']  # (T, 768)
        
        return embeddings.astype(np.float32)
    
    def load_category(
        self,
        category: str,
        num_videos: Optional[int] = None,
        shuffle: bool = True,
    ) -> List[np.ndarray]:
        """
        Load embeddings for all (or a specified number of) videos in a category.
        
        Args:
            category: Category name
            num_videos: If specified, load only this many random videos
            shuffle: Whether to shuffle video order
            
        Returns:
            List of (T_i, 768) embedding arrays
        """
        videos = self.list_videos_in_category(category)
        
        if shuffle:
            random.shuffle(videos)
        
        if num_videos is not None:
            videos = videos[:num_videos]
        
        embeddings_list = []
        for video_name in videos:
            video_path = self.videos_dir / category / video_name
            try:
                data = np.load(video_path)
                embeddings_list.append(data['embeddings'].astype(np.float32))
            except Exception as e:
                print(f"[Warning] Failed to load {video_name}: {e}")
        
        print(f"[VideoEmbeddingLoader] Loaded {len(embeddings_list)} videos from '{category}'")
        return embeddings_list
    
    def load_all_categories(
        self,
        num_videos_per_category: Optional[int] = None,
        shuffle: bool = True,
    ) -> Dict[str, List[np.ndarray]]:
        all_data = {}
        for category in self.categories:
            all_data[category] = self.load_category(
                category,
                num_videos=num_videos_per_category,
                shuffle=shuffle,
            )
        return all_data
    
    def get_statistics(self, category: str) -> Dict[str, any]:
        """
        Get statistics for a category.
        
        Args:
            category: Category name
            
        Returns:
            Dict with num_videos, min_length, max_length, avg_length
        """
        videos = self.list_videos_in_category(category)
        lengths = []
        
        for video_name in videos:
            # FIX: same path doubling fix as in load_category
            video_path = self.videos_dir / category / video_name
            try:
                data = np.load(video_path)
                lengths.append(data['embeddings'].shape[0])
            except Exception:
                pass
        
        if not lengths:
            return {
                'num_videos': 0,
                'min_length': 0,
                'max_length': 0,
                'avg_length': 0.0,
            }
        
        return {
            'num_videos': len(lengths),
            'min_length': min(lengths),
            'max_length': max(lengths),
            'avg_length': sum(lengths) / len(lengths),
        }
