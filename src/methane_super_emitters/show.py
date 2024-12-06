import numpy as np
import matplotlib.pyplot as plt
import click

@click.command()
@click.option('-i', '--input-file', help='Input NPZ file')
@click.option('--vmin', help='Minimum value for methane concentration')
@click.option('--vmax', help='Maximum value for methane concentration')
@click.option('--histogram', help='Show histogram', is_flag=True)
def main(input_file, vmin, vmax, histogram):
    data = np.load(input_file)
    if histogram:
        plt.hist(data['xch4'], bins=50)
        plt.show()
    plt.imshow(data['xch4'], vmin=int(vmin), vmax=int(vmax))
    plt.show()

if __name__ == '__main__':
    main()
g
