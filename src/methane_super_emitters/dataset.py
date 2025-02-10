import torch
from torch.utils.data import DataLoader, Dataset
import os
import glob
import random
import numpy as np
from torchvision.transforms import Compose, ToTensor, Normalize
from methane_super_emitters.dataset_stats import normalize

class TROPOMISuperEmitterDataset(Dataset):
    def __init__(self, data_dir, fields):
        self.data_dir = data_dir
        self.samples = []
        self.positive_filenames = glob.glob(os.path.join(data_dir, 'positive', '*.npz'))
        negative_filenames = glob.glob(os.path.join(data_dir, 'negative', '*.npz'))
        self.negative_filenames = negative_filenames[:len(self.positive_filenames)]
        for filename in self.positive_filenames:
            data = np.load(filename)
            self.samples.append((normalize(data, fields), 1.0))
        for filename in self.negative_filenames:
            data = np.load(filename)
            self.samples.append((normalize(data, fields), 0.0))

    def unload(self):
        """Make garbage collector collect the data.
        """
        self.samples = []
        import gc
        gc.collect()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img, label = self.samples[idx]
        return torch.tensor(img, dtype=torch.float), torch.tensor(label, dtype=torch.float)

