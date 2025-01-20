import numpy as np
import matplotlib.pyplot as plt
import click

@click.command()
@click.option('-i', '--input-file', help='Input NPZ file')
def main(input_file):
    data = np.load(input_file)
    plt.imshow(data['methane'], vmin=-30, vmax=30)
    plt.show()

if __name__ == '__main__':
    main()
