import click
import lightning as L
from methane_super_emitters.model import SuperEmitterDetector
from methane_super_emitters.datamodule import TROPOMISuperEmitterDataModule


@click.command()
@click.option("-i", "--input-dir", help="Data directory")
@click.option("-m", "--max-epochs", help="Maximum number of epochs", default=1)
def train_model(input_dir, max_epochs):
    fields = ["methane", "u10", "v10", "qa"]
    model = SuperEmitterDetector(fields=fields)
    datamodule = TROPOMISuperEmitterDataModule(input_dir, fields=fields)
    trainer = L.Trainer(max_epochs=max_epochs)
    trainer.fit(model=model, datamodule=datamodule)
    trainer.test(model=model, datamodule=datamodule)


if __name__ == "__main__":
    train_model()
