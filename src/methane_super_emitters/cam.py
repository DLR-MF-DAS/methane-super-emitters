import torch
import numpy as np
import cv2
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from methane_super_emitters.dataset_stats import normalize
from methane_super_emitters.model import SuperEmitterDetector
import matplotlib.pyplot as plt
import glob
import os
import pathlib
import click

@click.command()
@click.option("-i", "--input", "input_path", required=True, type=click.Path(exists=True))
@click.option("-o", "--output", "output_path", required=True, type=click.Path())
@click.option("-c", "--checkpoint", "checkpoint", required=True, type=click.Path(exists=True))
def main(input_path, output_path, checkpoint):
    for input_file in glob.glob(os.path.join(input_path, '*.npz')):
        output_file = os.path.join(output_path, pathlib.Path(input_file).stem + ".png")
        model = SuperEmitterDetector.load_from_checkpoint(checkpoint, fields=['methane', 'u10', 'v10', 'qa'])
        model.eval()
        target_layer = model.conv_layers[-1]
        cam = GradCAM(model=model, target_layers=[target_layer])
        data = np.load(input_file)
        example_image = normalize(data, ['methane', 'u10', 'v10', 'qa'])
        example_image = torch.tensor(np.array([example_image]), dtype=torch.float)
        input_tensor = example_image.requires_grad_(True)
        grayscale_cam = cam(input_tensor=input_tensor)[0]
        fig, axs = plt.subplots(1, 2)
        axs[0].imshow(example_image[0][0].detach().numpy())
        axs[1].imshow(grayscale_cam)
        plt.savefig(output_file)

if __name__ == "__main__":
    main()
