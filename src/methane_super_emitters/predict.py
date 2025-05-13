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
import matplotlib.pyplot as plt
from pytorch_grad_cam import GradCAM
from methane_super_emitters.model import SuperEmitterDetector
from methane_super_emitters.dataset_stats import normalize
from methane_super_emitters.utils import destripe, parse_date, patch_generator


def predict_from_tropomi_file(file_path, output_path, model, threshold):
    print(f"ANALYZING {file_path}")
    try:
        for patch in patch_generator(file_path):
            x = normalize(patch, fields=['methane', 'u10', 'v10', 'qa'])
            x = torch.tensor(np.array([x]), dtype=torch.float)
            if predict(model, x) > threshold:
                stem = str(uuid.uuid4())
                print(f"Found emitter in {file_path}")
                target_layer = model.conv_layers[-1]
                cam = GradCAM(model=model, target_layers=[target_layer])
                input_tensor = x.requires_grad_(True)
                grayscale_cam = cam(input_tensor=input_tensor)[0]
                location = np.array(np.unravel_index(grayscale_cam.argmax(), shape=grayscale_cam.shape))
                lat = patch['lat'][location[0]][location[1]]
                lon = patch['lon'][location[0]][location[1]]
                with open('metadata.csv', 'a') as fd:
                    fd.write(f"{stem},{lat},{lon}\n")
                fig, axs = plt.subplots(1, 2)
                axs[0].imshow(x[0][0].detach().numpy())
                axs[1].imshow(grayscale_cam)
                axs[1].scatter([location[1]], [location[0]], color='red', s=50)
                plt.savefig(os.path.join(output_path, stem + ".png"))
                plt.close()
                np.savez(os.path.join(output_path, stem + ".npz"), **patch)
    except OSError:
        pass


def predict(model, x):
    y_hat = torch.sigmoid(model(x))
    y_hat = y_hat.detach().cpu().numpy()
    return y_hat[0][0]


@click.command()
@click.option("-c", "--checkpoint", help="Checkpoint file")
@click.option("-i", "--input-dir", help="Directory with files")
@click.option("-o", "--output-dir", help="Output directory")
@click.option("-n", "--n-jobs", help="Number of parallel jobs", default=1)
@click.option(
    "-t",
    "--threshold",
    help="Threshold for the value of the sigmoid output to qualify as a hit",
    default=0.9,
)
def main(checkpoint, input_dir, output_dir, n_jobs, threshold):
    model = SuperEmitterDetector.load_from_checkpoint(checkpoint, fields=['methane', 'u10', 'v10', 'qa'])
    model.eval()
    for month_path in glob.glob(os.path.join(input_dir, "*")):
        for day_path in glob.glob(os.path.join(month_path, "*")):
            Parallel(n_jobs=n_jobs)(
                delayed(predict_from_tropomi_file)(
                    file_path, output_dir, model, threshold
                )
                for file_path in glob.glob(os.path.join(day_path, "*.nc"))
            )


if __name__ == "__main__":
    main()
