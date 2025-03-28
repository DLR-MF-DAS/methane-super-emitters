import click
import lightning as L
import optuna
from methane_super_emitters.model import SuperEmitterDetector
from methane_super_emitters.datamodule import TROPOMISuperEmitterDataModule


def optimize_model(input_dir, max_epochs, n_trials):
    def objective(trial):
        fields = ["methane", "u10", "v10", "qa"]
        dropout_rate = trial.suggest_float("dropout", 0.1, 0.9)
        weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-2)
        learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2)
        n = 1
        result = 0.0
        for _ in range(n):
            model = SuperEmitterDetector(fields=fields, dropout=dropout_rate, weight_decay=weight_decay, lr=learning_rate)
            datamodule = TROPOMISuperEmitterDataModule(input_dir, fields=fields)
            trainer = L.Trainer(max_epochs=max_epochs)
            trainer.fit(model=model, datamodule=datamodule)
            result += trainer.callback_metrics["val_acc"].item()
        return result / n

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)
    df = study.trials_dataframe()
    df.to_csv("opt_results.csv")
    print("Best parameters:", study.best_params)


@click.command()
@click.option("-i", "--input-dir", help="Data directory")
@click.option("-m", "--max-epochs", help="Maximum number of epochs", default=100)
@click.option(
    "-n", "--n-trials", help="Number of trials or points to sample", default=200
)
def optimize_model_(input_dir, max_epochs, n_trials):
    return optimize_model(input_dir, max_epochs, n_trials)


if __name__ == "__main__":
    optimize_model_()
