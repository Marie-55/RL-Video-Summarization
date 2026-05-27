import json
import os
import sys

class Logger(object):
    """Redirect stdout to file."""
    def __init__(self, fpath):
        self.console = sys.stdout
        self.file = open(fpath, 'w')

    def __call__(self, msg):
        self.console.write(msg)
        self.file.write(msg)

    def flush(self):
        self.file.flush()

    def close(self):
        self.file.close()

def read_json(fpath):
    """Read JSON file."""
    with open(fpath, 'r') as f:
        obj = json.load(f)
    return obj

def write_json(obj, fpath):
    """Write JSON file."""
    with open(fpath, 'w') as f:
        json.dump(obj, f, indent=2)

def save_checkpoint(state_dict, fpath):
    """Save PyTorch model checkpoint."""
    import torch
    torch.save(state_dict, fpath)
