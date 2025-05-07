from methane_super_emitters.dataset import TROPOMISuperEmitterDataset
import click

@click.command()
@click.option('-i', '--data-dir', help='Dataset directory')
def main(data_dir):
    fields = ['methane', 'qa', 'u10', 'v10']
    dataset = TROPOMISuperEmitterDataset(data_dir, fields)

if __name__ == '__main__':
    main()
