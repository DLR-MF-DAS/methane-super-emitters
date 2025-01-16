import lightning as L
from torch.utils.data import random_split, DataLoader
from methane_super_emitters.dataset import TROPOMISuperEmitterDataset

class TROPOMISuperEmitterDataModule(L.LightningDataModule):
    def __init__(self, data_dir, batch_size=32):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.dataset = TROPOMISuperEmitterDataset(self.data_dir)
        self.train_set, self.val_set, self.test_set =\
            random_split(self.dataset, [0.7, 0.15, 0.15])

    def setup(self, stage):
        pass

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size, shuffle=True)

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.batch_size, shuffle=True)

    def predict_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.batch_size, shuffle=True)

    def teardown(self, stage):
        pass
