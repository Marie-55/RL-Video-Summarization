## in this file we will define the state class , that is a set of frames that represents the state of the summary at a given time
from frame import Frame

class State:
    def __init__(self, frames: dict):
        self.frames = frames         

    def add_frame(self, frame: Frame):
        self.frames[frame.index] = frame

    def remove_frame(self, index: int):
        del self.frames[index]

    def replace_frame(self, old_index: int, new_frame: Frame):
        """Core operation of the vertical policy — swap anchor with neighbor."""
        del self.frames[old_index]
        self.frames[new_frame.index] = new_frame

    def __repr__(self):
        return f"State(frame_indices={list(self.frames.keys())})"