import glob
import os
import random
import shutil
import numpy as np
import netCDF4
import datetime

def sample_files(glob_pattern, output_dir, n):
    """A small utility function to sample files from a directory at random.

    Parameters
    ----------
    glob_pattern: str
        A string of the form /path/to/*.npz or so. To select files from.
    output_dir: str
        A directory name into which to copy files.
    n: int
        How many files to take at random.
    """
    for path in random.sample(list(glob.glob(glob_pattern)), k=n):
        shutil.copy(path, output_dir)

def destripe(fd):
    """A destriping algorithm for tropoomi data.

    Parameters
    ----------
    fd: a NetCDF4 file handler
    
    Returns
    -------
    A numpy array with destriped methane ppm amounts.
    """
    ch4 = fd['/PRODUCT/methane_mixing_ratio'][:]
    ch4corr = fd['/PRODUCT/methane_mixing_ratio_bias_corrected'][:]
    ch4corr[ch4corr>1E20]=np.nan
    ch4corrdestrip = ch4corr.copy() * np.nan
    # get the number of rows
    n = ch4corr.shape[1]
    # get the number of columns
    m = ch4corr.shape[2]
    back = np.zeros((n,m)) * np.nan
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
        back[:,i] = np.nanmedian(ch4corr[0, :, st:sp], axis=1)
    this = ch4corr[0,:,:] - back
    stripes = np.zeros((n, m)) * np.nan
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
        stripes[j, :] = np.nanmedian(this[st:sp, :], axis=0)
    ch4corrdestrip[0,:,:] = ch4corr[0,:,:] - stripes
    return ch4corrdestrip

def parse_date(date_str, time_str):
    """Parse date from a simple string format.

    Parameters
    ----------
    date_str: str
        Date of the form YYYYMMDD
    time_str: str
        Time of the form HH:MM:SS

    Returns
    -------
    datetime.datetime
    """
    return datetime.datetime.strptime(date_str + time_str, '%Y%m%d%H:%M:%S')

def patch_generator(file_path):
    """A generator for 32x32 pixel patches with 50% overlap.

    Parameters
    ----------
    file_path: str
        A path to a TROPOMI netCDF4 file.

    Yields
    ------
    dict
        A dictionary with patch arrays.
    """
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
                    u10_window = fd['PRODUCT/SUPPORT_DATA/INPUT_DATA/eastward_wind'][:][0][row:row + 32][:, col:col + 32]
                    v10_window = fd['PRODUCT/SUPPORT_DATA/INPUT_DATA/northward_wind'][:][0][row:row + 32][:, col:col + 32]
                    sza_window = fd['PRODUCT/SUPPORT_DATA/GEOLOCATIONS/solar_zenith_angle'][:][0][row:row + 32][:, col:col + 32]
                    vza_window = fd['PRODUCT/SUPPORT_DATA/GEOLOCATIONS/viewing_zenith_angle'][:][0][row:row + 32][:, col:col + 32]
                    saa_window = fd['/PRODUCT/SUPPORT_DATA/GEOLOCATIONS/solar_azimuth_angle'][:][0][row:row + 32][:, col:col + 32]
                    vaa_window =fd['/PRODUCT/SUPPORT_DATA/GEOLOCATIONS/viewing_azimuth_angle'][:][0][row:row + 32][:, col:col + 32]
                    f_delta = saa_window - vaa_window
                    raa_window = np.where(f_delta < 0, np.abs(f_delta + 180.0), np.abs(f_delta - 180.0))
                    raa_window = abs(180.0 - raa_window)
                    scattering_angle =\
                        (-np.cos(np.radians(sza_window)) * np.cos(np.radians(vza_window)) +
                         np.sin(np.radians(sza_window)) * np.sin(np.radians(vza_window)) * np.cos(np.radians(raa_window)))
                    sa_std_window = fd['PRODUCT/SUPPORT_DATA/INPUT_DATA/surface_altitude_precision'][:][0][row:row + 32][:, col:col + 32]
                    cloud_fraction_window = fd['PRODUCT/SUPPORT_DATA/INPUT_DATA/cloud_fraction_VIIRS_SWIR_IFOV'][:][0][row:row + 32][:, col:col + 32]
                    cirrus_reflectance_window = fd['PRODUCT/SUPPORT_DATA/INPUT_DATA/reflectance_cirrus_VIIRS_SWIR'][:][0][row:row + 32][:, col:col + 32]
                    methane_ratio_std_window = fd['PRODUCT/SUPPORT_DATA/INPUT_DATA/methane_ratio_weak_strong_standard_deviation'][:][0][row:row + 32][:, col:col + 32]
                    methane_precision_window = fd['PRODUCT/methane_mixing_ratio_precision'][:][0][row:row + 32][:, col:col + 32]
                    surface_albedo_window = fd['PRODUCT/SUPPORT_DATA/DETAILED_RESULTS/surface_albedo_SWIR'][:][0][row:row + 32][:, col:col + 32]
                    surface_albedo_precision_window = fd['PRODUCT/SUPPORT_DATA/DETAILED_RESULTS/surface_albedo_SWIR_precision'][:][0][row:row + 32][:, col:col + 32]
                    aerosol_optical_thickness_window = fd['PRODUCT/SUPPORT_DATA/DETAILED_RESULTS/aerosol_optical_thickness_SWIR'][:][0][row:row + 32][:, col:col + 32]
                    times = fd['PRODUCT/time_utc'][:][0][row:row + 32]
                    parsed_time = [datetime.datetime.fromisoformat(time_str[:19]) for time_str in times]
                    if original.mask.sum() < 0.2 * 32 * 32:
                        yield {
                            'methane': methane_window,
                            'lat': lat_window,
                            'lon': lon_window,
                            'qa': qa_window,
                            'time': parsed_time,
                            'mask': original.mask,
                            'non_destriped': original,
                            'u10': u10_window,
                            'v10': v10_window,
                            'sza': np.cos(sza_window),
                            'vza': np.cos(vza_window),
                            'scattering_angle': scattering_angle,
                            'sa_std': sa_std_window,
                            'cloud_fraction': cloud_fraction_window,
                            'cirrus_reflectance': cirrus_reflectance_window,
                            'methane_ratio_std': methane_ratio_std_window,
                            'methane_precision': methane_precision_window,
                            'surface_albedo': surface_albedo_window,
                            'surface_albedo_precision': surface_albedo_precision_window,
                            'aerosol_optical_thickness': aerosol_optical_thickness_window,
                        }
