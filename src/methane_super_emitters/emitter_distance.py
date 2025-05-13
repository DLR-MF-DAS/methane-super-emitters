import torch
import numpy as np
import cv2
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from methane_super_emitters.dataset_stats import normalize
from methane_super_emitters.dataset import TROPOMISuperEmitterDataset
from methane_super_emitters.model import SuperEmitterDetector
import matplotlib.pyplot as plt
import glob
import os
import pathlib
import click

@click.command()
@click.option("-i", "--input", "input_path", required=True, type=click.Path(exists=True))
@click.option("-c", "--checkpoint", "checkpoint", required=True, type=click.Path(exists=True))
def main(input_path, checkpoint):
    dataset = TROPOMISuperEmitterDataset(input_path, fields=['methane', 'u10', 'v10', 'qa'])
    distances = []
    for input_file in dataset.positive_filenames:
        model = SuperEmitterDetector.load_from_checkpoint(checkpoint, fields=['methane', 'u10', 'v10', 'qa'])
        model.eval()
        target_layer = model.conv_layers[-1]
        cam = GradCAM(model=model, target_layers=[target_layer])
        data = np.load(input_file)
        example_image = normalize(data, ['methane', 'u10', 'v10', 'qa'])
        example_image = torch.tensor(np.array([example_image]), dtype=torch.float)
        input_tensor = example_image.requires_grad_(True)
        grayscale_cam = cam(input_tensor=input_tensor)[0]
        location = np.argwhere(data['location'] == 1)
        if len(location) < 1:
            continue
        maximum = np.unravel_index(grayscale_cam.argmax(), shape=grayscale_cam.shape)
        idx = np.sqrt(((location - maximum) ** 2).sum(axis=1)).argmin()
        units = np.array([7.0, 5.5])
        distances.append(np.sqrt(((location[idx] * units - np.array(maximum) * units) ** 2).sum()))
    plt.hist(distances, bins=25)
    plt.savefig('emitter_distance_distribution.png')
        
if __name__ == "__main__":
    main()
