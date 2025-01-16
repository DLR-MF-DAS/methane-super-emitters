import pytest
import lightning as L
from methane_super_emitters.model import SuperEmitterDetector

def test_model():
    model = SuperEmitterDetector()
    datamodule = TROPOMISuperEmitterDataModule('./data/dataset')
    trainer = L.trainer(gpus=0, max_epochs=1)
    trainer.fit(model=model, datamodule=datamodule)
