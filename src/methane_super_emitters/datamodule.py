import lightning as L
from torch.utils.data import random_split, DataLoader

class TROPOMIDataModule(L.LightningDataModule):
    def __init__(self, data_dir, batch_size=32):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size

    def setup(self, stage):
        pass

    def train_dataloader(self):
        pass

    def val_dataloader(self):
        pass

    def test_dataloader(self):
        pass

    def predict_dataloader(self):
        pass

    def teardown(self, stage):
        pass
