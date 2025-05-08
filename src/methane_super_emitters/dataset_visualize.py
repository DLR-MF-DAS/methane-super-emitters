from methane_super_emitters.dataset import TROPOMISuperEmitterDataset
import os
import matplotlib.pyplot as plt
import click
import uuid

@click.command()
@click.option('-i', '--data-dir', help='Dataset directory')
@click.option('-o', '--output-dir', help='Output directory')
def main(data_dir, output_dir):
    fields = ['methane', 'qa', 'u10', 'v10']
    dataset = TROPOMISuperEmitterDataset(data_dir, fields)
    for sample in dataset:
        filename = str(uuid.uuid4())
        fig, axs = plt.subplots(2, len(fields))
        for index, field in enumerate(fields):
            axs[0, index].imshow(sample[0][index])
            axs[1, index].hist(sample[0][index])
        if sample[1] == 0.0:
            plt.savefig(os.path.join(output_dir, 'negative', filename))
        else:
            plt.savefig(os.path.join(output_dir, 'positive', filename))
        plt.close()

if __name__ == '__main__':
    main()
