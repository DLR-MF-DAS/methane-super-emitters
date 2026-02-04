import click
import numpy as np
import datetime
import netCDF4
import os
import glob
import math
import uuid
from joblib import Parallel, delayed
from methane_super_emitters.utils import destripe, parse_date, patch_generator


def process_tropomi_file(
    file_path, month_path, day_path, output_dir, input_file
):
    print(f"ANALYZING: {file_path}")
    with open(input_file, "r") as csv_fd:
        csv_data = csv_fd.readlines()[1:]
    try:
        for patch in patch_generator(file_path):
            emitter = False
            location = np.zeros((32, 32))
            for csv_line in csv_data:
                location, country, lat, lon = csv_line.split(",")
                if (
                    (patch["lat"].min() < float(lat) < patch["lat"].max())
                    and (patch["lon"].min() < float(lon) < patch["lon"].max())
                    and patch["mask"].sum() < 0.2 * 32 * 32
                ):
                    print(f"FOUND: {csv_line}")
                    emitter = True
                    positive_path = os.path.join(
                        output_dir,
                        "positive",
                        f"{location}_{country}_{lat}_{lon}_{str(patch['time'])}.npz",
                    )
                    np.savez_compressed(positive_path, location=location, **patch)
    except OSError:
        pass


@click.command()
@click.option("-i", "--input-file", help="Input CSV with super-emitter locations")
@click.option(
    "-p",
    "--prefix",
    help="Folder with TROPOMI data",
    default="/dss/dsstbyfs03/pn56su/pn56su-dss-0022/Sentinel-5p/L2/CH4/2021/",
)
@click.option("-o", "--output_dir", help="Output folder")
@click.option("-n", "--njobs", help="Number of jobs", default=1)
def main(input_file, prefix, output_dir, njobs):
    for month_path in glob.glob(os.path.join(prefix, "*")):
        for day_path in glob.glob(os.path.join(month_path, "*")):
            Parallel(n_jobs=njobs)(
                delayed(process_tropomi_file)(
                    file_path, month_path, day_path, output_dir, input_file
                )
                for file_path in glob.glob(os.path.join(day_path, "*.nc"))
            )


if __name__ == "__main__":
    main()
