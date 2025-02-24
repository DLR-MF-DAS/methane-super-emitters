import pytest
import lightning as L
from methane_super_emitters.model import SuperEmitterDetector
from methane_super_emitters.datamodule import TROPOMISuperEmitterDataModule
from methane_super_emitters.optimize import optimize_model

def test_model():
    model = SuperEmitterDetector(fields=['methane', 'qa', 'u10', 'v10'])
    datamodule = TROPOMISuperEmitterDataModule('./data/dataset', fields=['methane', 'qa', 'u10', 'v10'])
    trainer = L.Trainer(max_epochs=10)
    trainer.fit(model=model, datamodule=datamodule)
    trainer.test(model=model, datamodule=datamodule)

def test_optimize():
    optimize_model('data/dataset', 1, 10)
