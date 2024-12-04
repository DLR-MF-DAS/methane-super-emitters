import numpy as np
import matplotlib.pyplot as plt
import click

@click.command()
@click.option('-i', '--input-file', help='Input NPZ file')
def main(input_file):
    data = np.load(input_file)
    import pdb; pdb.set_trace()
    plt.imshow(data['methane'])
    plt.show()

if __name__ == '__main__':
    main()
