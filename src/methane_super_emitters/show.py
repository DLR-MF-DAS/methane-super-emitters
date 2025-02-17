import numpy as np
import matplotlib.pyplot as plt
import click
import os
import glob
from pathlib import Path


@click.command()
@click.option("-i", "--input-dir", help="Input directory")
@click.option("-o", "--output-dir", help="Output directory")
def main(input_dir, output_dir):
    for input_file in glob.glob(os.path.join(input_dir, "*.npz")):
        plt.figure(figsize=(8, 8))
        data = np.load(input_file)
        m = data["methane"]
        m[data["mask"]] = np.nanmedian(m)
        plt.imshow(m)
        plt.colorbar()
        x = np.arange(data["methane"].shape[1])
        y = np.arange(data["methane"].shape[0])
        xv, yv = np.meshgrid(x, y)
        plt.quiver(xv, yv, data["u10"], data["v10"])
        stem = Path(input_file).stem
        plt.savefig(os.path.join(output_dir, stem + ".png"))


if __name__ == "__main__":
    main()
