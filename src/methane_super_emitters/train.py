import click
import lightning as L
from methane_super_emitters.model import SuperEmitterDetector
from methane_super_emitters.datamodule import TROPOMISuperEmitterDataModule

@click.command()
@click.option('-i', '--input-dir', help='Data directory')
def train_model(input_dir):
    model = SuperEmitterDetector()
    datamodule = TROPOMISuperEmitterDataModule(input_dir)
    trainer = L.Trainer(max_epochs=1000)
    trainer.fit(model=model, datamodule=datamodule)

if __name__ == '__main__':
    train_model()
