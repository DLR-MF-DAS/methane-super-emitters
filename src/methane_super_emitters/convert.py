#importing packages
import numpy as np
from scipy import interpolate
from netCDF4 import Dataset
from shapely.geometry import Point
import geopandas as gpd
from geopy.distance import geodesic
from scipy.interpolate import griddata
import rasterio
import matplotlib.pyplot as plt
import click
import numpy.ma as ma

@click.command()
@click.option('-i', '--input-file', help='Input netCDF file')
@click.option('-o', '--output-file', help='Output NPZ file')
def main(input_file, output_file):
    #load data 
    rootgrp = Dataset(input_file, "r", format="NETCDF4")

    lat = rootgrp.groups['instrument']['latitude_center'][:]
    lon = rootgrp.groups['instrument']['longitude_center'][:]
    time = rootgrp.groups['instrument']['time'][:]
    xch4 = rootgrp.groups['target_product']['xch4_corrected'][:].filled(0)

    scanline = rootgrp.groups['instrument']['scanline'][:].filled(-1)
    ground_pixel = rootgrp.groups['instrument']['ground_pixel'][:].filled(-1)

    methane_matrix = np.full((scanline.max() + 1, ground_pixel.max() + 1), 0.0)
    lat_matrix = np.full((scanline.max() + 1, ground_pixel.max() + 1), -1000)
    lon_matrix = np.full((scanline.max() + 1, ground_pixel.max() + 1), -1000)
    time_matrix = np.full((scanline.max() + 1, ground_pixel.max() + 1), None)

    for sl, gp, m, lt, ln, time in zip(scanline, ground_pixel, xch4, lat, lon, time):
        if (sl >= 0) and (gp >= 0):
            methane_matrix[sl][gp] = m
            lat_matrix[sl][gp] = lt
            lon_matrix[sl][gp] = ln
            time_matrix[sl][gp] = time

    np.savez(output_file, methane=methane_matrix, lat=lat_matrix, lon=lon_matrix, time=time_matrix)

if __name__ == '__main__':
    main()

