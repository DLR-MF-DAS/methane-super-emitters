import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image
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

    # Load an example input (replace with your own preprocessing)
    example_image = torch.randn(1, 13, 32, 32)  # Example random input, replace with real data
    input_tensor = example_image.requires_grad_(True)  # Enable gradients

    # Get Grad-CAM heatmap
    grayscale_cam = cam(input_tensor=input_tensor)[0]  # [0] to remove batch dimension

    # Convert to numpy and normalize
    grayscale_cam = np.maximum(grayscale_cam, 0)  # ReLU operation
    grayscale_cam = grayscale_cam / grayscale_cam.max()  # Normalize between 0-1

    # Convert input to a displayable format (normalize & transpose)
    input_image_np = example_image[0].mean(0).cpu().numpy()  # Average over 13 channels
    input_image_np = cv2.normalize(input_image_np, None, 0, 1, cv2.NORM_MINMAX)  # Normalize

    # Overlay heatmap on image
    visualization = show_cam_on_image(input_image_np, grayscale_cam, use_rgb=True)

    # Show results
    plt.figure(figsize=(6, 6))
    plt.imshow(visualization)
    plt.axis("off")
    plt.title("Grad-CAM Heatmap")
    plt.savefig('test.png')

if __name__ == '__main__':
    main()
