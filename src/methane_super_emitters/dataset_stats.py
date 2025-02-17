"""This module gathers descriptive statistics of the dataset for normalization porpoises."""

import click
import os
import glob
from collections import defaultdict
import numpy as np
import json

DATASET_STATS = {
    "methane": {
        "mean": 1862.9641199220912,
        "median": 1879.0816650390625,
        "std": 66.34578040359114,
        "min": 1115.669189453125,
        "max": 2527.0158081054688,
    },
    "qa": {
        "mean": 0.8079299465221805,
        "median": 1.0,
        "std": 0.32904064000460764,
        "min": 0.0,
        "max": 1.0,
    },
    "u10": {
        "mean": -0.3512625828486657,
        "median": -0.23248033225536346,
        "std": 3.082583589284194,
        "min": -16.78030014038086,
        "max": 17.881803512573242,
    },
    "v10": {
        "mean": -0.23241149430312563,
        "median": -0.3439149558544159,
        "std": 3.0300053289038926,
        "min": -16.52522850036621,
        "max": 18.62321662902832,
    },
    "sza": {
        "mean": 0.6892348274598235,
        "median": 0.7051632106304169,
        "std": 0.17974190986091854,
        "min": 0.2416170984506607,
        "max": 0.9911504983901978,
    },
    "vza": {
        "mean": 0.8908392256240117,
        "median": 0.9287970662117004,
        "std": 0.11002508192787454,
        "min": 0.4970099925994873,
        "max": 0.9998136758804321,
    },
    "scattering_angle": {
        "mean": -0.6033922639033007,
        "median": -0.647180062178118,
        "std": 0.26156256858396837,
        "min": -0.9999999439365981,
        "max": 0.40050015763160485,
    },
    "sa_std": {
        "mean": 16.502662089815193,
        "median": 8.709863662719727,
        "std": 26.36883710896449,
        "min": 0.0,
        "max": 965.2076416015625,
    },
    "cloud_fraction": {
        "mean": 7.44786507344531e34,
        "median": 0.0,
        "std": 8.584557285723529e35,
        "min": 0.0,
        "max": 9.969209968386869e36,
    },
    "cirrus_reflectance": {
        "mean": 7.883227337810577e35,
        "median": 0.000916256511118263,
        "std": 2.690260605093876e36,
        "min": 0.00021310012380126864,
        "max": 9.969209968386869e36,
    },
    "methane_ratio_std": {
        "mean": 7.230339082303514e35,
        "median": 0.014838868286460638,
        "std": 2.5855944796044255e36,
        "min": 0.0,
        "max": 9.969209968386869e36,
    },
    "methane_precision": {
        "mean": 7.230339082303514e35,
        "median": 1.6617772579193115,
        "std": 2.5855944796044255e36,
        "min": 0.660350501537323,
        "max": 9.969209968386869e36,
    },
    "surface_albedo": {
        "mean": 7.230339082303514e35,
        "median": 0.1909976452589035,
        "std": 2.5855944796044255e36,
        "min": -0.06418812274932861,
        "max": 9.969209968386869e36,
    },
    "surface_albedo_precision": {
        "mean": 7.230339082303514e35,
        "median": 0.00013950675929663703,
        "std": 2.5855944796044255e36,
        "min": 4.564654591376893e-05,
        "max": 9.969209968386869e36,
    },
    "aerosol_optical_thickness": {
        "mean": 7.230339082303514e35,
        "median": 0.04946461133658886,
        "std": 2.5855944796044255e36,
        "min": 0.00025335102691315114,
        "max": 9.969209968386869e36,
    },
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
        if field in ["methane"]:
            m = np.array(data[field])
            m[data["mask"]] = np.nanmedian(m)
            mean = DATASET_STATS[field]["mean"]
            std = DATASET_STATS[field]["std"]
            m = (m - mean) / std
            result.append(m)
        elif field == "qa":
            qa = np.array(data["qa"])
            qa = (qa > 0.5).astype(np.float64)
            result.append(qa)
        elif field in ["u10", "v10"]:
            x = np.array(data[field])
            x[np.argwhere(x > 1.0e30)] = 0.0
            mean = DATASET_STATS[field]["mean"]
            std = DATASET_STATS[field]["std"]
            x = (x - mean) / std
            result.append(x)
        elif field in ["sza", "vza", "scattering_angle", "sa_std"]:
            x = np.array(data[field])
            mean = DATASET_STATS[field]["mean"]
            std = DATASET_STATS[field]["std"]
            x = (x - mean) / std
            result.append(x)
        else:
            result.append(np.array(data[field]))
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
                if key in ["u10", "v10"]:
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
