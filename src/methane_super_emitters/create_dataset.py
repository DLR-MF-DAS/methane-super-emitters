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
    file_path, month_path, day_path, output_dir, input_file, negative=False
):
    print(f"ANALYZING: {file_path}")
    with open(input_file, "r") as csv_fd:
        csv_data = csv_fd.readlines()[1:]
    try:
        for patch in patch_generator(file_path):
            emitter = False
            location = np.zeros((32, 32))
            for csv_line in csv_data:
                date, time, lat, lon, _, _, _ = csv_line.split(",")[:7]
                if (
                    (patch["lat"].min() < float(lat) < patch["lat"].max())
                    and (patch["lon"].min() < float(lon) < patch["lon"].max())
                    and (
                        min(patch["time"])
                        <= parse_date(date, time)
                        <= max(patch["time"])
                    )
                    and patch["mask"].sum() < 0.2 * 32 * 32
                ):
                    print(f"FOUND: {csv_line}")
                    emitter = True
                    # Locate the pixel with the emitter
                    for row in range(32):
                        for col in range(32):
                            min_lat = patch["lat_bounds"][row][col].min()
                            max_lat = patch["lat_bounds"][row][col].max()
                            min_lon = patch["lon_bounds"][row][col].min()
                            max_lon = patch["lon_bounds"][row][col].max()
                            if (
                                min_lat < float(lat) < max_lat
                                and min_lon < float(lon) < max_lon
                            ):
                                location[row][col] = 1
                    if not negative:
                        positive_path = os.path.join(
                            output_dir,
                            "positive",
                            f"{date}_{time}_{lat}_{lon}_{np.random.randint(0, 10000):04d}.npz",
                        )
                        np.savez_compressed(positive_path, location=location, **patch)
            if (
                negative
                and not emitter
                and np.random.random() < 0.02
                and patch["mask"].sum() < 0.2 * 32 * 32
            ):
                negative_path = os.path.join(
                    output_dir, "negative", f"{uuid.uuid4()}.npz"
                )
                np.savez_compressed(negative_path, location=location, **patch)
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
@click.option(
    "--negative",
    is_flag=True,
    help="Whether to collect negative samples",
    default=False,
)
def main(input_file, prefix, output_dir, njobs, negative):
    for month_path in glob.glob(os.path.join(prefix, "*")):
        for day_path in glob.glob(os.path.join(month_path, "*")):
            Parallel(n_jobs=njobs)(
                delayed(process_tropomi_file)(
                    file_path, month_path, day_path, output_dir, input_file, negative
                )
                for file_path in glob.glob(os.path.join(day_path, "*.nc"))
            )


if __name__ == "__main__":
    main()
