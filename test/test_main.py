import pytest
import lightning as L
from methane_super_emitters.model import SuperEmitterDetector
from methane_super_emitters.datamodule import TROPOMISuperEmitterDataModule

def test_model():
    model = SuperEmitterDetector()
    datamodule = TROPOMISuperEmitterDataModule('./data/dataset_wf')
    trainer = L.Trainer(max_epochs=1)
    trainer.fit(model=model, datamodule=datamodule)
    trainer.test(model=model, datamodule=datamodule)
