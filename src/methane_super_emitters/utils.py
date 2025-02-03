import glob
import os
import random
import shutil

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
