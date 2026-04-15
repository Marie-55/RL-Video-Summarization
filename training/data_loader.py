"""
Data Loader for Video Embeddings.

Loads CLIP ViT-L/14 embeddings (768-dim) from .npz files in videos/embeddings_clip_vitl14/
"""

import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import random


class VideoEmbeddingLoader:
    """
    Loads video embeddings from .npz files.
    
    Usage:
        loader = VideoEmbeddingLoader(videos_dir='videos/embeddings_clip_vitl14')
        videos = loader.load_category('Activity', num_videos=10)
        features = loader.load_single_video(video_path)
    """
    
    def __init__(self, videos_dir: str = 'videos/embeddings_clip_vitl14'):
        """
        Args:
            videos_dir: Path to embeddings directory
        """
        self.videos_dir = Path(videos_dir)
        if not self.videos_dir.exists():
            raise FileNotFoundError(f"Videos directory not found: {self.videos_dir}")
        
        # Scan available categories
        self.categories = [d.name for d in self.videos_dir.iterdir() if d.is_dir()]
        self.categories.sort()
        
        print(f"[VideoEmbeddingLoader] Found {len(self.categories)} categories")
        print(f"[VideoEmbeddingLoader] Categories: {', '.join(self.categories[:5])}...")
    
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
    
    def load_single_video(self, video_path: str) -> np.ndarray:
        """
        Load embeddings for a single video.
        
        Args:
            video_path: Path to .npz file (relative to videos_dir or absolute)
            
        Returns:
            embeddings: (T, 768) numpy array of CLIP embeddings
        """
        path = Path(video_path)
        
        # Handle relative paths
        if not path.is_absolute():
            path = self.videos_dir / path
        
        if not path.exists():
            raise FileNotFoundError(f"Video file not found: {path}")
        
        # Load .npz file
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
        Load embeddings for all (or specified number of) videos in a category.
        
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
                embeddings = self.load_single_video(video_path)
                embeddings_list.append(embeddings)
            except Exception as e:
                print(f"[Warning] Failed to load {video_name}: {e}")
        
        print(f"[VideoEmbeddingLoader] Loaded {len(embeddings_list)} videos from '{category}'")
        return embeddings_list
    
    def load_all_categories(
        self,
        num_videos_per_category: Optional[int] = None,
        shuffle: bool = True,
    ) -> Dict[str, List[np.ndarray]]:
        """
        Load embeddings from all categories.
        
        Args:
            num_videos_per_category: If specified, load only this many videos per category
            shuffle: Whether to shuffle video order within each category
            
        Returns:
            Dictionary: category_name -> list of embedding arrays
        """
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
            video_path = self.videos_dir / category / video_name
            try:
                embeddings = self.load_single_video(video_path)
                lengths.append(embeddings.shape[0])
            except:
                pass
        
        if not lengths:
            return {
                'num_videos': 0,
                'min_length': 0,
                'max_length': 0,
                'avg_length': 0,
            }
        
        return {
            'num_videos': len(lengths),
            'min_length': min(lengths),
            'max_length': max(lengths),
            'avg_length': sum(lengths) / len(lengths),
        }


if __name__ == "__main__":
    # Test the loader
    loader = VideoEmbeddingLoader()
    
    print("\n" + "=" * 60)
    print("AVAILABLE CATEGORIES")
    print("=" * 60)
    categories = loader.get_categories()
    for i, cat in enumerate(categories[:10]):
        stats = loader.get_statistics(cat)
        print(f"{i+1}. {cat}")
        print(f"   Videos: {stats['num_videos']}, Length: {stats['min_length']}-{stats['max_length']} frames")
    
    print("\n" + "=" * 60)
    print("LOADING SAMPLE VIDEO")
    print("=" * 60)
    
    # Load a sample video
    sample_category = 'Activity'
    videos = loader.list_videos_in_category(sample_category)
    sample_video = videos[0]
    
    embeddings = loader.load_single_video(f"{sample_category}/{sample_video}")
    print(f"Video: {sample_video}")
    print(f"Embeddings shape: {embeddings.shape}")
    print(f"Embedding dtype: {embeddings.dtype}")
    print(f"Sample: {embeddings[0, :5]}")
