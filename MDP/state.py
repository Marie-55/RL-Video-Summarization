## in this file we will define the state class , that is a set of frames that represents the state of the summary at a given time
from frame import Frame
from typing import List, Set

class State: 
    def __init__(self, selected_indices: List[int], anchor_idx: int):
        self.selected_indices: Set[int] = set(selected_indices)
        self.anchor_idx: int = anchor_idx
   
    def add_frame(self, idx: int) -> None:
        self.selected_indices.add(idx)
        self.anchor_idx = idx

    def remove_frame(self, index: int):
        self.selected_indices.remove(index)
        
    def replace_anchor(self, new_idx: int) -> bool:
        """Replaces current anchor with new_idx. Returns True if changed."""
        if self.anchor_idx == new_idx:
            return False
        self.selected_indices.discard(self.anchor_idx)
        self.selected_indices.add(new_idx)
        self.anchor_idx = new_idx
        return True
    
    def get_sorted_indices(self) -> List[int]:
        return sorted(list(self.selected_indices))
    
    def copy(self) -> 'State':
        return State(list(self.selected_indices), self.anchor_idx)

    def __repr__(self):
        return f"State(frame_indices={list(self.selected_indices)}, anchor={self.anchor_idx})"