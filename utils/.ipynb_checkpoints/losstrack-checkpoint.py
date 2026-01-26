import pickle
from collections import defaultdict

class LossTracker:
    def __init__(self):
        self.reset()
    
    def update(self, loss_dict):
        for k, v in loss_dict.items(): self.losses[k].append(v)
    
    def get_avg_losses(self):
        return {k: sum(v)/len(v) if v else 0 for k,v in self.losses.items()}
    
    def reset(self):
        self.losses = defaultdict(list)
    
    def save(self, path):
        with open(path, 'wb') as f: pickle.dump(dict(self.losses), f)
    
    def load(self, path):
        with open(path, 'rb') as f: self.losses = defaultdict(list, pickle.load(f))