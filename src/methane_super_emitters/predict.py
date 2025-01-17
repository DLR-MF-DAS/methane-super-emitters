import click
import torch
import lightning as L
import numpy as np
import glob
import os
from methane_super_emitters.model import SuperEmitterDetector

def predict(model, npz_file):
    data = np.load(npz_file)
    m = data['methane']
    m[data['mask']] = 0.0
    m = np.array([[m]])
    y_hat = model(torch.tensor(m, dtype=torch.float))
    y_hat = y_hat.detach().cpu().numpy()
    return round(y_hat[0][0])

@click.command()
@click.option('-c', '--checkpoint', help='Checkpoint file')
@click.option('-i', '--input-dir', help='Directory with files')
def main(checkpoint, input_dir):
    model = SuperEmitterDetector.load_from_checkpoint(checkpoint)
    model.eval()
    sum_ = 0.0
    counter = 0
    for npz_file in glob.glob(os.path.join(input_dir, '*.npz')):
        y_hat = predict(model, npz_file)
        print(npz_file, predict(model, npz_file))
        sum_ += y_hat
        counter += 1
    print(sum_ / counter)

if __name__ == '__main__':
    main()
