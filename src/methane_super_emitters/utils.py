import glob
import os
import random
import shutil

def sample_files(glob_pattern, output_dir, n):
    """A small utility function to sample files from a directory at randm.
    """
    for path in random.sample(list(glob.glob(glob_pattern)), k=n):
        shutil.copy(path, output_dir)
