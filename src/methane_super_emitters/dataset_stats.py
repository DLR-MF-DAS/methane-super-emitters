"""This module gathers descriptive statistics of the dataset for normalization porpoises."""

import click
import os
import glob
from collections import defaultdict
import numpy as np
import json

DATASET_STATS = {
    "methane": {
        "mean": 1862.5366583594632,
        "median": 1878.5830688476562,
        "std": 66.58710672017679,
        "min": 971.0758666992188,
        "max": 3324.72998046875
    },
    "qa": {
        "mean": 0.8106104458852114,
        "median": 1.0,
        "std": 0.3274058748852436,
        "min": 0.0,
        "max": 1.0
    },
    "u10": {
        "mean": -0.3505060237319564,
        "median": -0.237032949924469,
        "std": 3.0427688265223236,
        "min": -16.58311653137207,
        "max": 16.715341567993164
    },
    "v10": {
        "mean": -0.2016462336753655,
        "median": -0.3121262937784195,
        "std": 3.0363927475921657,
        "min": -15.385231018066406,
        "max": 22.70849609375
    },
    "sza": {
        "mean": 0.6852687383838644,
        "median": 0.70140740275383,
        "std": 0.17970663271587994,
        "min": 0.24475011229515076,
        "max": 0.9911531805992126
    },
    "vza": {
        "mean": 0.891019676507066,
        "median": 0.928871214389801,
        "std": 0.11010636053335394,
        "min": 0.49698495864868164,
        "max": 0.9998136758804321
    },
    "scattering_angle": {
        "mean": -0.6000556685447394,
        "median": -0.6434199315569306,
        "std": 0.26127084751804125,
        "min": -0.9999982782919568,
        "max": 0.4007348417791011
    },
    "sa_std": {
        "mean": 16.65754507928552,
        "median": 8.775827407836914,
        "std": 26.56312667561806,
        "min": 0.0,
        "max": 965.2076416015625
    },
    "cloud_fraction": {
        "mean": 0.010853144763690946,
        "median": 0.0,
        "std": 0.07413277621675883,
        "min": 0.0,
        "max": 1.0
    },
    "cirrus_reflectance": {
        "mean": 0.002408624468020358,
        "median": 0.0,
        "std": 0.00954486875404972,
        "min": 0.0,
        "max": 0.1243622899055481
    },
    "methane_ratio_std": {
        "mean": 0.0049173906056315655,
        "median": 0.0,
        "std": 0.0085810962161963,
        "min": 0.0,
        "max": 0.6137746572494507
    },
    "methane_precision": {
        "mean": 0.6099002507901794,
        "median": 0.0,
        "std": 1.0656924191550565,
        "min": 0.0,
        "max": 47.113624572753906
    },
    "surface_albedo": {
        "mean": 0.08097774154289568,
        "median": 0.0,
        "std": 0.16019963835712323,
        "min": -0.05953392758965492,
        "max": 0.7304432988166809
    },
    "surface_albedo_precision": {
        "mean": 6.182406963785989e-05,
        "median": 0.0,
        "std": 0.00012510325923893395,
        "min": 0.0,
        "max": 0.009150429628789425
    },
    "aerosol_optical_thickness": {
        "mean": 0.020228099796273182,
        "median": 0.0,
        "std": 0.04644573871021487,
        "min": 0.0,
        "max": 0.9478278756141663
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
        x = np.array(data[field])
        if field == "methane":
            x[data["mask"]] = np.nanmedian(x)
        x[np.argwhere(x > 1.0e30)] = 0.0
        a = DATASET_STATS[field]["min"]
        b = DATASET_STATS[field]["max"]
        x = (x - a) / (b - a)
        result.append(x)
    # for field in fields:
    #     if field in ["methane"]:
    #         m = np.array(data[field])
    #         m[data["mask"]] = np.nanmedian(m)
    #         mean = DATASET_STATS[field]["mean"]
    #         std = DATASET_STATS[field]["std"]
    #         m = (m - mean) / std
    #         result.append(m)
    #     elif field == "qa":
    #         qa = np.array(data["qa"])
    #         qa = (qa > 0.5).astype(np.float64)
    #         result.append(qa)
    #     elif field in ["u10", "v10"]:
    #         x = np.array(data[field])
    #         x[np.argwhere(x > 1.0e30)] = 0.0
    #         mean = DATASET_STATS[field]["mean"]
    #         std = DATASET_STATS[field]["std"]
    #         x = (x - mean) / std
    #         result.append(x)
    #     elif field in ["sza", "vza", "scattering_angle", "sa_std"]:
    #         x = np.array(data[field])
    #         mean = DATASET_STATS[field]["mean"]
    #         std = DATASET_STATS[field]["std"]
    #         x = (x - mean) / std
    #         result.append(x)
    #     else:
    #         result.append(np.array(data[field]))
    return np.array(result)


@click.command()
@click.option("-i", "--input-dir", help="Directory with the full dataset")
def main(input_dir):
    results = defaultdict(list)
    positive_filenames = glob.glob(os.path.join(input_dir, "positive", "*.npz"))
    negative_filenames = glob.glob(os.path.join(input_dir, "negative", "*.npz"))
    negative_filenames = negative_filenames[: len(positive_filenames)]
    for filename in positive_filenames + negative_filenames:
        data = np.load(filename)
        for key in data:
            if key not in [
                "time",
                "location",
                "lat",
                "lon",
                "lat_bounds",
                "lon_bounds",
                "mask",
                "non_destriped",
            ]:
                x = data[key]
                if key in [
                        "u10", "v10", "cloud_fraction", "cirrus_reflectance",
                        "methane_ratio_std", "methane_precision", "surface_albedo",
                        "surface_albedo_precision", "aerosol_optical_thickness"]:
                    x[np.argwhere(x > 1.0e30)] = 0.0
                results[key].append(x)
    for key in results:
        results[key] = np.array(results[key]).astype(np.float128)
    stats = {}
    for key in results:
        stats[key] = {
            "mean": float(np.nanmean(results[key])),
            "median": float(np.nanmedian(results[key])),
            "std": float(np.nanstd(results[key])),
            "min": float(np.nanmin(results[key])),
            "max": float(np.nanmax(results[key])),
        }
    print(json.dumps(stats, indent=4))


if __name__ == "__main__":
    main()
