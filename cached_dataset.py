# cached_dataset.py

import os
import torch
from torch.utils.data import Dataset

class CachedMFCCDataset(Dataset):
    def __init__(self, root_dir):
        self.samples = []
        for label, genre in enumerate(['non_metal', 'metal']):
            genre_dir = os.path.join(root_dir, genre)
            if not os.path.isdir(genre_dir):
                continue
            for fname in os.listdir(genre_dir):
                if fname.endswith('.pt'):
                    self.samples.append((os.path.join(genre_dir, fname), label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        data = torch.load(path)
        return data['mfcc'], torch.tensor(label, dtype=torch.float32)
