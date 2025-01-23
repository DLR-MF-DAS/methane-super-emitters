import torch
from torch.utils.data import DataLoader, Dataset
import os
import glob
import random
import numpy as np
from torchvision.transforms import Compose, ToTensor, Normalize

class TROPOMISuperEmitterDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.samples = []
        self.positive_filenames = glob.glob(os.path.join(data_dir, 'positive', '*.npz'))
        negative_filenames = glob.glob(os.path.join(data_dir, 'negative', '*.npz'))
        self.negative_filenames = negative_filenames[:len(self.positive_filenames)]
        for filename in self.positive_filenames:
            data = np.load(filename)
            m = data['methane']
            m[data['mask']] = np.nanmedian(m)
            u = data['u10']
            v = data['v10']
            sample = np.array([m, u, v])
            self.samples.append((sample, 1.0))
        for filename in self.negative_filenames:
            data = np.load(filename)
            m = data['methane']
            m[data['mask']] = np.nanmedian(m)
            u = data['u10']
            v = data['v10']
            sample = np.array([m, u, v])
            self.samples.append((sample, 0.0))
        self.means = np.array(self.mean())
        self.stds = np.array(self.std())
        for index, sample in enumerate(self.samples):
            img, label = sample
            img_ = np.array([(img[0] - self.means[0]) / self.stds[0],
                             (img[1] - self.means[1]) / self.stds[1],
                             (img[2] - self.means[2]) / self.stds[2]])
            self.samples[index] = (img_, label)

    def mean(self):
        mean_m = 0.0
        mean_u = 0.0
        mean_v = 0.0
        for img, label in self.samples:
            mean_m += img[0].mean()
            mean_u += img[1].mean()
            mean_v += img[2].mean()
        return float(mean_m / len(self)), float(mean_u / len(self)), float(mean_v / len(self))

    def std(self):
        std_m = 0.0
        std_u = 0.0
        std_v = 0.0
        for img, label in self.samples:
            std_m += img[0].std() ** 2
            std_u += img[1].std() ** 2
            std_v += img[2].std() ** 2
        return float(np.sqrt(std_m / len(self))), float(np.sqrt(std_u / len(self))), float(np.sqrt(std_v / len(self)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img, label = self.samples[idx]
        return torch.tensor(img, dtype=torch.float), torch.tensor(label, dtype=torch.float)

