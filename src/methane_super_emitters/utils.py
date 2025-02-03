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
