import click
import torch
import lightning as L
import numpy as np
import glob
import os
import netCDF4
import uuid
from joblib import Parallel, delayed
import datetime
from methane_super_emitters.model import SuperEmitterDetector
from methane_super_emitters.dataset import TROPOMISuperEmitterDataset
from methane_super_emitters.utils import destripe, parse_date, patch_generator


def predict_from_tropomi_file(file_path, output_path, model, dataset, threshold):
    print(f"ANALYZING {file_path}")
    try:
        for patch in patch_generator(file_path):
            if predict(model, dataset, patch) > threshold:
                print(f"Found emitter in {file_path}")
                np.savez(os.path.join(output_path, str(uuid.uuid4()) + ".npz"), **patch)
    except OSError:
        pass


def predict(model, dataset, patch):
    x = np.array([dataset.normalize(patch)])
    y_hat = model(torch.tensor(x, dtype=torch.float))
    y_hat = y_hat.detach().cpu().numpy()
    return y_hat[0][0]


@click.command()
@click.option("-c", "--checkpoint", help="Checkpoint file")
@click.option("-d", "--dataset", help="Directory with the dataset")
@click.option("-i", "--input-dir", help="Directory with files")
@click.option("-o", "--output-dir", help="Output directory")
@click.option("-n", "--n-jobs", help="Number of parallel jobs", default=1)
@click.option(
    "-t",
    "--threshold",
    help="Threshold for the value of the sigmoid output to qualify as a hit",
    default=0.9,
)
def main(checkpoint, dataset, input_dir, output_dir, n_jobs, threshold):
    model = SuperEmitterDetector.load_from_checkpoint(checkpoint)
    dataset = TROPOMISuperEmitterDataset(dataset)
    dataset.unload()
    for month_path in glob.glob(os.path.join(input_dir, "*")):
        for day_path in glob.glob(os.path.join(month_path, "*")):
            Parallel(n_jobs=n_jobs)(
                delayed(predict_from_tropomi_file)(
                    file_path, output_dir, model, dataset, threshold
                )
                for file_path in glob.glob(os.path.join(day_path, "*.nc"))
            )


if __name__ == "__main__":
    main()
