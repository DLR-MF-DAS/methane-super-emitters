import click
import numpy as np
import datetime
import netCDF4
import os
import glob
import math
import uuid
from joblib import Parallel, delayed

def destripe(fd):
    ch4 = fd['/PRODUCT/methane_mixing_ratio'][:]
    ch4corr = fd['/PRODUCT/methane_mixing_ratio_bias_corrected'][:]
    ch4corrdestrip = ch4corr.copy() * np.nan
    n = ch4corr.shape[1]
    # get the number of columns
    m = ch4corr.shape[2]
    back = np.zeros((1, n, m)) * np.nan
    for i in range(m):
        # define half window size
        ws = 7
        if i < ws:
            st = 0
            sp = i + ws
        elif m - i < ws:
            st = i - ws
            sp = m - 1
        else:
            st = i - ws
            sp = i + ws
        back[0, :, i] = np.nanmedian(ch4corr[0, :, st:sp], axis=1)
    this = ch4corr - back
    stripes = np.zeros((1, n, m)) * np.nan
    for j in range(n):
        ws = 60
        if j < ws:
            st = 0
            sp = j + ws
        elif n - j < ws:
            st = j - ws
            sp = n - 1
        else:
            st = j - ws
            sp = j + ws
        stripes[0, j, :] = np.nanmedian(this[0,st:sp,:], axis=0)
    ch4corrdestrip = this - stripes
    return ch4corrdestrip

def parse_date(date_str, time_str):
    return datetime.datetime.strptime(date_str + time_str, '%Y%m%d%H:%M:%S')

def process_tropomi_file(file_path, month_path, day_path, output_dir, input_file):
    print(f"ANALYZING: {file_path}")
    with open(input_file, 'r') as csv_fd:
        csv_data = csv_fd.readlines()[1:]
    try:
        with netCDF4.Dataset(file_path, 'r') as fd:
            destriped = destripe(fd)
            rows = destriped.shape[1]
            cols = destriped.shape[2]
            for row in range(0, rows, 16):
                for col in range(0, cols, 16):
                    if row + 32 < rows and col + 32 < cols:
                        methane_window = destriped[0][row:row + 32][:, col: col + 32]
                        original = fd['PRODUCT/methane_mixing_ratio_bias_corrected'][:][0][row:row + 32][:, col: col + 32]
                        lat_window = fd['PRODUCT/latitude'][:][0][row:row + 32][:, col: col + 32]
                        lon_window = fd['PRODUCT/longitude'][:][0][row:row + 32][:, col: col + 32]
                        qa_window = fd['PRODUCT/qa_value'][:][0][row:row + 32][:, col:col + 32]
                        times = fd['PRODUCT/time_utc'][:][0][row:row + 32]
                        parsed_time = [datetime.datetime.fromisoformat(time_str[:19]) for time_str in times]
                        emitter = False
                        for csv_line in csv_data:
                            date, time, lat, lon, _, _, _ = csv_line.split(',')
                            if ((lat_window.min() < float(lat) < lat_window.max()) and
                                (lon_window.min() < float(lon) < lon_window.max()) and
                                (min(parsed_time) <= parse_date(date, time)  <= max(parsed_time)) and
                                original.mask.sum() < 0.2 * 32 * 32):
                                print(f"FOUND: {csv_line}")
                                emitter = True
                        if not emitter and np.random.random() < 0.02 and original.mask.sum() < 0.2 * 32 * 32:
                            negative_path = os.path.join(output_dir, 'negative', f"{uuid.uuid4()}.npz")
                            np.savez(negative_path, methane=methane_window, lat=lat_window,
                                     lon=lon_window, qa=qa_window, time=parsed_time,
                                     mask=original.mask, non_destriped=original)
    except OSError:
        pass

@click.command()
@click.option('-i', '--input-file', help='Input CSV with super-emitter locations')
@click.option('-p', '--prefix', help='Folder with TROPOMI data', default='/dss/dsstbyfs03/pn56su/pn56su-dss-0022/Sentinel-5p/L2/CH4/2021/')
@click.option('-o', '--output_dir', help='Output folder')
@click.option('-n', '--njobs', help='Number of jobs', default=1)
def main(input_file, prefix, output_dir, njobs):
    for month_path in glob.glob(os.path.join(prefix, '*')):
        for day_path in glob.glob(os.path.join(month_path, '*')):
            Parallel(n_jobs=njobs)(delayed(process_tropomi_file)(file_path, month_path, day_path, output_dir, input_file) for file_path in glob.glob(os.path.join(day_path, '*.nc')))

if __name__ == '__main__':
    main()
