import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image
from methane_super_emitters.dataset_stats import normalize
import click

@click.command()
@click.option('-i', '--input-file', help='Input file')
@click.option('-o', '--output-file', help='Output file')
@click.option('-c', '--checkpoint', help='Model checkpoint file')
def main(input_file, output_file, checkpoint):
    model = SuperEmitterDetector.load_from_checkpoint(checkpoint)
    model.eval()
    target_layer = model.conv_layers[-1]
    cam = GradCAM(model=model, target_layers=[target_layer])
    data = np.load(input_file)
    example_image = normalize(data, ['methane', 'u10', 'v10', 'qa'])
    example_image = torch.tensor(np.array([example_image]), dtype=torch.float)
    input_tensor = example_image.requires_grad_(True)
    grayscale_cam = cam(input_tensor=input_tensor)[0]
    grayscale_cam = np.maximum(grayscale_cam, 0)
    grayscale_cam = grayscale_cam / grayscale_cam.max()
    input_image_np = example_image[0].mean(0).cpu().numpy()
    input_image_np = cv2.normalize(input_image_np, None, 0, 1, cv2.NORM_MINMAX)
    visualization = show_cam_on_image(input_image_np, grayscale_cam, use_rgb=True)
    plt.figure(figsize=(6, 6))
    plt.imshow(visualization)
    plt.axis("off")
    plt.title("Grad-CAM Heatmap")
    plt.savefig('test.png')

if __name__ == '__main__':
    main()
