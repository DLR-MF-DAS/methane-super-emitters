import cdsapi
import netCDF4
import click
import numpy as np
from scipy.ndimage import zoom
import uuid
import os
import tempfile
import glob
from pathlib import Path


def download_windfield(input_file, output_file, tmp_dir):
    c = cdsapi.Client()
    data = np.load(input_file, allow_pickle=True)
    latitude = data["lat"]
    longitude = data["lon"]
    area = [
        float(latitude.min()),
        float(longitude.min()),
        float(latitude.max()),
        float(longitude.max()),
    ]
    time = data["time"].min()
    tmp_file = os.path.join(tmp_dir, str(uuid.uuid4()) + ".nc")
    c.retrieve(
        "reanalysis-era5-single-levels",
        {
            "product_type": "reanalysis",
            "variable": ["10m_u_component_of_wind", "10m_v_component_of_wind"],
            "year": [f"{time.year}"],
            "month": [f"{time.month:02d}"],
            "day": [f"{time.day:02d}"],
            "time": [f"{time.hour:02d}:00"],
            "area": area,
            "format": "netcdf",
        },
        tmp_file,
    )
    with netCDF4.Dataset(tmp_file, "r") as fd:
        u = fd["u10"][:][0]
        v = fd["v10"][:][0]
        zoom_factors = (32 / u.shape[0], 32 / u.shape[1])
        u_ups = zoom(u, zoom_factors, order=3)
        v_ups = zoom(v, zoom_factors, order=3)
        np.savez(
            output_file,
            methane=data["methane"],
            lat=data["lat"],
            lon=data["lon"],
            qa=data["qa"],
            time=data["time"],
            mask=data["mask"],
            non_destriped=data["non_destriped"],
            u=u_ups,
            v=v_ups,
        )


@click.command()
@click.option("-i", "--input-dir", help="Input directory")
@click.option("-o", "--output-dir", help="Output directory")
def main(input_dir, output_dir):
    tmp_dir_ = tempfile.TemporaryDirectory()
    tmp_dir = tmp_dir_.name
    for input_file in glob.glob(os.path.join(input_dir, "*.npz")):
        stem = Path(input_file).stem
        output_file = os.path.join(output_dir, stem + "_wf.npz")
        if os.path.isfile(output_file):
            continue
        download_windfield(input_file, output_file, tmp_dir)
    tmp_dir_.cleanup()


if __name__ == "__main__":
    main()
