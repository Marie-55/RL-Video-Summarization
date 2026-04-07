## in this file , we will define the reward class that computes the reward for a given state , the reward will be computed using the frames in the summary 
## it will include the diversity reward and the representative reward along with the temporal reward 
## in this file we will define the state class , that is a set of frames that represents the state of the summary at a given time 
# reward.py
import numpy as np
from state import State

class Reward:
    def __init__(self, state, video):
        self.state = state
        self.video = video         

    def _cosine_similarity(self, a, b):
        denom = (np.linalg.norm(a) * np.linalg.norm(b))
        if denom == 0:
            return 0.0
        return np.dot(a, b) / denom

    def compute_diversity_reward(self):
        frames = list(self.state.frames.values())
        n = len(frames)
        if n <= 1:
            return 0.0

        total = 0.0
        count = 0
        for i in range(n):
            for j in range(i + 1, n):
                sim = self._cosine_similarity(frames[i].data, frames[j].data)
                total += (1 - sim)
                count += 1

        return total / count if count > 0 else 0.0

    def compute_representative_reward(self):
        summary_frames = self.state.frames
        if not summary_frames:
            return 0.0

        total = 0.0
        for video_frame in self.video.values():   
            max_sim = max(
                self._cosine_similarity(video_frame.data, s.data)
                for s in self.state.frames.values()
            )
            total += max_sim

        return total / len(self.video)  

    def compute_temporal_reward(self):
        indices = [f.index for f in self.state.frames.values()]
        if len(indices) <= 1:
            return 0.0

        video_length = len(self.video)
        std = np.std(indices)
        max_std = video_length / 2
        return std / max_std if max_std > 0 else 0.0

    def compute_total_reward(self, w_div=0.4, w_rep=0.4, w_temp=0.2):
        div  = self.compute_diversity_reward()
        rep  = self.compute_representative_reward()
        temp = self.compute_temporal_reward()

        return w_div * div + w_rep * rep + w_temp * temp
