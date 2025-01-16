import torch
from torch.utils.data import DataLoader, Dataset
import os
import glob
import numpy as np

class TROPOMISuperEmitterDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.positive = glob.glob(os.path.join(data_dir, 'positive', '*.npz'))
        self.negative = glob.glob(os.path.join(data_dir, 'negative', '*.npz'))
        self.all_samples = self.positive + self.negative

    def __len__(self):
        return len(self.all_samples)

    def __getitem__(self, idx):
        data = np.load(self.all_samples[idx])
        if idx < len(self.positive):
            label = [1]
        else:
            label = [0]
        m = data['methane']
        m[data['mask']] = 0.0
        return torch.tensor(m, dtype=torch.float), torch.tensor(label, dtype=torch.float)
