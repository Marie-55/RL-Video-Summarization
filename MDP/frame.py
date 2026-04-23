## this file defines the frame class, an abstraction of the frame that is a vectore 
## It contains the frame as it is along with operations that are overloaded for common uses across the methods 
# frame.py
import numpy as np

class Frame:
    def __init__(self, data, index):  
        self.data = data
        self.index = index   ## needed for the temporal reward if we want to integrate it in the future    

    def __add__(self, other):
        return Frame(self.data + other.data, self.index)

    def __sub__(self, other):
        return Frame(self.data - other.data, self.index)

    def __mul__(self, scalar):
        return Frame(self.data * scalar, self.index)

    def __truediv__(self, scalar):
        return Frame(self.data / scalar, self.index)

    def __repr__(self):
        return f"Frame(index={self.index}, data={self.data})"