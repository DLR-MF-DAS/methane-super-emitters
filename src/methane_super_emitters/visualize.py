import numpy as np
import matplotlib.pyplot as plt
import click
import glob
import os
import random


@click.command()
@click.option("-i", "--input-dir", help="Folder with samples")
def main(input_dir):
    sample = random.sample(glob.glob(os.path.join(input_dir, "*.npz")), 64)
    fig = plt.figure(figsize=(8, 8))
    row = 0
    col = 0
    for image in sample:
        data = np.load(image)
        m = data["methane"]
        mask = data["mask"]
        m[mask] = 0.0
        ax = fig.add_subplot(8, 8, row * 8 + col + 1)
        ax.imshow(m, vmin=-30, vmax=30)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal")
        col = col + 1
        if col >= 8:
            col = 0
            row += 1
            if row >= 8:
                row = 0
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.show()


if __name__ == "__main__":
    main()
