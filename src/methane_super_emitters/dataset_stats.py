"""This module gathers descriptive statistics of the dataset for normalization porpoises.
"""

import click
import os
import glob
from collections import defaultdict
import numpy as np
import json
from methane_super_emitters.dataset import TROPOMISuperEmitterDataset

DATASET_STATS = {
    "methane": {
        "mean": 1862.9641199220912,
        "median": 1879.0816650390625,
        "std": 66.34578040359114,
        "min": 1115.669189453125,
        "max": 2527.0158081054688
    },
    "qa": {
        "mean": 0.8079299465221805,
        "median": 1.0,
        "std": 0.32904064000460764,
        "min": 0.0,
        "max": 1.0
    },
    "u10": {
        "mean": -0.3737712936343934,
        "median": -0.2512122094631195,
        "std": 2.5397220760580796,
        "min": -7.993372440338135,
        "max": 7.180738925933838
    },
    "v10": {
        "mean": -0.20653207072280394,
        "median": -0.3010627329349518,
        "std": 2.50340158368457,
        "min": -7.131891250610352,
        "max": 7.727571487426758
    },
    "sza": {
        "mean": 0.692825452684085,
        "median": 0.7061138153076172,
        "std": 0.16906553051495238,
        "min": 0.28583598136901855,
        "max": 0.9679996967315674
    },
    "vza": {
        "mean": 0.8777098636023852,
        "median": 0.9007660746574402,
        "std": 0.0949174862234199,
        "min": 0.5820207595825195,
        "max": 0.9988005757331848
    },
    "scattering_angle": {
        "mean": -0.6142603057303578,
        "median": -0.6498205645515349,
        "std": 0.2346965600393898,
        "min": -0.9900801177989922,
        "max": 0.06035856566720038
    },
    "sa_std": {
        "mean": 11.335806350486507,
        "median": 8.125806331634521,
        "std": 11.14189705095489,
        "min": 0.6719605922698975,
        "max": 126.7691650390625
    },
    "cloud_fraction": {
        "mean": 0.004773651326928909,
        "median": 0.0,
        "std": 0.03413009761311816,
        "min": 0.0,
        "max": 0.9139785170555115
    },
    "cirrus_reflectance": {
        "mean": 7.88115811657442e+35,
        "median": 0.0009864408057183027,
        "std": 2.689937819585094e+36,
        "min": 0.0003837809490505606,
        "max": 9.969209968386869e+36
    },
    "methane_ratio_std": {
        "mean": 7.70331073291505e+35,
        "median": 0.019470931962132454,
        "std": 2.6619884019915237e+36,
        "min": 0.004167073871940374,
        "max": 9.969209968386869e+36
    },
    "methane_precision": {
        "mean": 7.441922840598362e+35,
        "median": 1.7081022262573242,
        "std": 2.620150182907788e+36,
        "min": 0.9740676879882812,
        "max": 9.969209968386869e+36
    },
    "surface_albedo": {
        "mean": 6.72848642743153e+35,
        "median": 0.2202262580394745,
        "std": 2.5010086166203653e+36,
        "min": 0.025173652917146683,
        "max": 9.969209968386869e+36
    },
    "surface_albedo_precision": {
        "mean": 6.944741625014735e+35,
        "median": 0.00014865858247503638,
        "std": 2.537925211903355e+36,
        "min": 8.372333832085133e-05,
        "max": 9.969209968386869e+36
    },
    "aerosol_optical_thickness": {
        "mean": 7.416364958364563e+35,
        "median": 0.055175287649035454,
        "std": 2.6160094142138956e+36,
        "min": 0.009366033598780632,
        "max": 9.969209968386869e+36
    }
}

def normalize(data, fields):
    """Normalize the selected fields in the dataset.

    Parameters
    ----------
    data: dict
        A dictionary with the full dataset data from an .NPZ file
    fields: list
        A list with field names

    Returns
    -------
    NumPy array
    """
    result = []
    for field in fields:
        if field == 'methane':
            m = np.array(data['methane'])
            m[data['mask']] = np.nanmedian(m)
            m = (m - DATASET_STATS['methane']['mean']) / DATASET_STATS['methane']['std']
            results.append(m)
        else:
            results.append(np.array(data[field]))
    return np.array(result)

@click.command()
@click.option('-i', '--input-dir', help='Directory with the full dataset')
def main(input_dir):
    results = defaultdict(list)
    positive_filenames = glob.glob(os.path.join(input_dir, 'positive', '*.npz'))
    negative_filenames = glob.glob(os.path.join(input_dir, 'negative', '*.npz'))
    negative_filenames = negative_filenames[:len(positive_filenames)]
    for filename in positive_filenames + negative_filenames:
        data = np.load(filename)
        for key in data:
            if key not in ['time', 'location', 'lat', 'lon', 'lat_bounds', 'lon_bounds',
                           'mask', 'non_destriped']:
                results[key].append(data[key])
    for key in results:
        results[key] = np.array(results[key]).astype(np.float128)
        upper = np.percentile(results[key], 99)
        lower = np.percentile(results[key], 1)
        results[key][np.argwhere(results[key] > upper)] = np.nan
        results[key][np.argwhere(results[key] < lower)] = np.nan
    stats = {}
    for key in results:
        stats[key] = {
            'mean': float(np.nanmean(results[key])),
            'median': float(np.nanmedian(results[key])),
            'std': float(np.nanstd(results[key])),
            'min': float(np.nanmin(results[key])),
            'max': float(np.nanmax(results[key])),
        }
    print(json.dumps(stats, indent=4))

if __name__ == '__main__':
    main()
