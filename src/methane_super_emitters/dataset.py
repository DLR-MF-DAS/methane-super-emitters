import torch
from torch.utils.data import DataLoader, Dataset
import os
import glob
import random
import numpy as np
from torchvision.transforms import Compose, ToTensor, Normalize
from methane_super_emitters.dataset_stats import normalize


class TROPOMISuperEmitterDataset(Dataset):
    def __init__(self, data_dir, fields, transform=None):
        self.data_dir = data_dir
        self.samples = []
        self.positive_filenames = glob.glob(os.path.join(data_dir, "positive", "*.npz"))
        negative_filenames = glob.glob(os.path.join(data_dir, "negative", "*.npz"))
        self.negative_filenames = negative_filenames[: len(self.positive_filenames)]
        for filename in self.positive_filenames:
            data = np.load(filename)
            image = normalize(data, fields)
            self.samples.append((image, 1.0))
        for filename in self.negative_filenames:
            data = np.load(filename)
            image = normalize(data, fields)
            self.samples.append((image, 0.0))

    def unload(self):
        """Make garbage collector collect the data."""
        self.samples = []
        import gc
        gc.collect()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img, label = self.samples[idx]
        return torch.tensor(img, dtype=torch.float), torch.tensor(
            label, dtype=torch.float
        )

class TransformWrapper(Dataset):
    def __init__(self, base_dataset, transform):
        self.base_dataset = base_dataset
        self.transform = transform

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        image, label = self.base_dataset[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

class TROPOMISuperEmitterLocatorDataset(TROPOMISuperEmitterDataset):
    def __init__(self, data_dir, fields):
        self.data_dir = data_dir
        self.samples = []
        self.positive_filenames = glob.glob(os.path.join(data_dir, "positive", "*.npz"))
        negative_filenames = glob.glob(os.path.join(data_dir, "negative", "*.npz"))
        self.negative_filenames = negative_filenames[: len(self.positive_filenames)]
        self.filenames = self.positive_filenames + self.negative_filenames
        for filename in self.filenames:
            data = np.load(filename)
            self.samples.append((normalize(data, fields), data["location"]))
