#importing packages
import numpy as np
from netCDF4 import Dataset
import click
import numpy.ma as ma
import datetime

@click.command()
@click.option('-i', '--input-file', help='Input netCDF file')
@click.option('-o', '--output-file', help='Output NPZ file')
def main(input_file, output_file):
    #load data 
    rootgrp = Dataset(input_file, "r", format="NETCDF4")

    lat = rootgrp.groups['instrument']['latitude_center'][:]
    lon = rootgrp.groups['instrument']['longitude_center'][:]
    time = rootgrp.groups['instrument']['time'][:]
    xch4 = rootgrp.groups['target_product']['xch4'][:].filled(0)
    xch4_corrected = rootgrp.groups['target_product']['xch4_corrected'][:].filled(0)

    scanline = rootgrp.groups['instrument']['scanline'][:].filled(-1)
    ground_pixel = rootgrp.groups['instrument']['ground_pixel'][:].filled(-1)

    xch4_matrix = np.full((scanline.max() + 1, ground_pixel.max() + 1), 0.0)
    xch4_corrected_matrix = np.full((scanline.max() + 1, ground_pixel.max() + 1), 0.0)
    lat_matrix = np.full((scanline.max() + 1, ground_pixel.max() + 1), -1000)
    lon_matrix = np.full((scanline.max() + 1, ground_pixel.max() + 1), -1000)
    time_matrix = np.full((scanline.max() + 1, ground_pixel.max() + 1), None)

    for i in range(scanline.shape[0]):
        if (scanline[i] >= 0) and (ground_pixel[i] >= 0):
            xch4_matrix[scanline[i]][ground_pixel[i]] = xch4[i]
            xch4_corrected_matrix[scanline[i]][ground_pixel[i]] = xch4_corrected[i]
            lat_matrix[scanline[i]][ground_pixel[i]] = lat[i]
            lon_matrix[scanline[i]][ground_pixel[i]] = lon[i]
            if time[i] is not None:
                time_matrix[scanline[i]][ground_pixel[i]] =\
                    datetime.datetime(time[i].data[0], time[i].data[1], time[i].data[2],
                                      time[i].data[3], time[i].data[4], time[i].data[5])
    np.savez(output_file, xch4=xch4_matrix, xch4_corrected=xch4_corrected_matrix, lat=lat_matrix, lon=lon_matrix, time=time_matrix)

if __name__ == '__main__':
    main()

