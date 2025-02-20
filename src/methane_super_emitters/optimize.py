import click
import lightning as L
import optuna
from methane_super_emitters.model import SuperEmitterDetector
from methane_super_emitters.datamodule import TROPOMISuperEmitterDataModule

@click.command()
@click.option("-i", "--input-dir", help="Data directory")
@click.option("-m", "--max-epochs", help="Maximum number of epochs", default=1)
def train_model(input_dir, max_epochs):
    def objective(trial):
        fields = ["methane", "u10", "v10", "qa"]
        dropout_rate = trial.suggest_float("dropout", 0.1, 0.9)
        weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-2)
        model = SuperEmitterDetector(fields=fields)
        datamodule = TROPOMISuperEmitterDataModule(input_dir, fields=fields)
        trainer = L.Trainer(max_epochs=max_epochs)
        trainer.fit(model=model, datamodule=datamodule)
        return trainer.callback_metrics['val_acc'].item()
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=200)
    df = study.trials_dataframe()
    df.to_csv('opt_results.csv')
    print("Best parameters:", study.best_params)

if __name__ == "__main__":
    train_model()
