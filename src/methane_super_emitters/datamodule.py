import lightning as L
from torch.utils.data import random_split, DataLoader
from methane_super_emitters.dataset import (
    TROPOMISuperEmitterDataset,
    TROPOMISuperEmitterLocatorDataset,
)


class TROPOMISuperEmitterDataModule(L.LightningDataModule):
    def __init__(self, data_dir, fields, batch_size=32, locator=False):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        if locator:
            self.dataset = TROPOMISuperEmitterLocatorDataset(
                self.data_dir, fields=fields
            )
        else:
            self.dataset = TROPOMISuperEmitterDataset(self.data_dir, fields=fields)
        self.train_set, self.val_set, self.test_set = random_split(
            self.dataset, [0.7, 0.15, 0.15]
        )

    def setup(self, stage):
        pass

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.batch_size, shuffle=False)

    def predict_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.batch_size, shuffle=False)

    def teardown(self, stage):
        pass
