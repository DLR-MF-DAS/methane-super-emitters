# Methane Super Emitter Detection from TROPOMI Data

The goal of this project is to detect methane super emitters in TROPOMI data.

# Installation

To install in a new conda environment:

```
git clone https://github.com/DLR-MF-DAS/methane-super-emitters.git
cd methane_super_emitters
pip install -e .
```

# Data Formats

## Dataset

The dataset follows a very simple format. The dataset directory contains two subfolders. One is called `positive` and the other is called `negative`. Data samples in the positive folder correspond to samples with methane emitters in them. Data samples in the negative folder correspond to samples without emitters.

```
dataset/
  positive/
    ...
  negative/
    ...
```

The samples themselves are `.npz` files. These are simply compressed dictionaries of numpy arrays that you can load with `data = np.load(filename)` and then access the data with `data['fieldname']`. Below is the list of valid field names (only methane, qa, u10 and v10 are used when training and predicting):

```
methane: Methane concentrations ppm (32x32 array)
lat: Latitude of the pixel center (32x32 array)
lon: Longitude of the pixel center (32x32 array)
lat_bounds: The latitudes of pixel corners (32x32x4 array)
lon_bounds: The longitudes of pixel corners (32x32x4 array)
qa: The quality assurance values (0 to 1 where high is good and low is bad) (32x32 array)
time: Timestamps for each pixel (32x32 array)
mask
non_destriped
u10: Windfield u component at 10m elevation (32x32 array)
v10: Windfield v component at 10m elevation (32x32 array)
sza
vza
scattering_angle
sa_std
cloud_fraction
cirrus_reflectance
methane_ratio_std
methane_precision
surface_albedo
surface_albedo_precision
aerosol_optical_thickness
location: An array that is zero everywhere but where the estimated methane emission site is located (32x32 array)
```

## CSV file

The CSV file contains known emitter locations. The format of the CSV file should be similar to the following (there can be extra columns if need be):

```
date,time_UTC,lat,lon,source_rate_t/h,uncertainty_t/h,estimated_source_type
20210101,06:00:45,36.75,109.76,32,16,Coal
20210101,06:00:55,37.53,110.75,39,22,Coal
20210101,07:37:49,20.89,85.22,4,2,Coal
20210101,07:38:15,23.3,90.79,51,15,Landfill/Urban
20210101,07:38:27,23.56,86.44,25,12,Coal
...
```

# Data Set Collection

In order to convert TROPOMI data to a format that can be used by our model we have prepared a script. The command line options for said script can be seen with:

```
python -m methane_super_emitters.create_dataset --help
```

Which should display something like the following:

```
Usage: python -m methane_super_emitters.create_dataset [OPTIONS]

Options:
  -i, --input-file TEXT  Input CSV with super-emitter locations
  -p, --prefix TEXT      Folder with TROPOMI data
  -o, --output_dir TEXT  Output folder
  -n, --njobs INTEGER    Number of jobs
  --negative             Whether to collect negative samples
  --help                 Show this message and exit.
```

A typical way to run it on terrabyte in order to collect samples with emitters (positive samples) would be:

```
python -m methane_super_emitters.create_dataset -i ../Schuit_etal2023_TROPOMI_all_plume_detections_2021.csv -o ../dataset_v3/ -n 40 -p /dss/dsstbyfs03/pn56su/pn56su-dss-0022/Sentinel-5p/L2/CH4/2021/
```

Some notes: the TROPOMI folder has to be the one that contains a full year of TROPOMI data. The .csv file has to follow a certain format (see source). The dataset folder has to contain two subfolders called positive and negative.

For negative samples use:

```
python -m methane_super_emitters.create_dataset -i ../Schuit_etal2023_TROPOMI_all_plume_detections_2021.csv -o ../dataset_v3/ -n 40 -p /dss/dsstbyfs03/pn56su/pn56su-dss-0022/Sentinel-5p/L2/CH4/2021/ --negative
```

# Model Training

The interface for training is a very simple script called `methane_super_emitters.train`.

```
python -m methane_super_emitters.train --help
```

Which should output:

```
Usage: python -m methane_super_emitters.train [OPTIONS]

Options:
  -i, --input-dir TEXT      Data directory
  -m, --max-epochs INTEGER  Maximum number of epochs
  --help                    Show this message and exit.
```

An example call would be:

```
python -m methane_super_emitters.train -i ../dataset_v3/ -m 500
```

Which will train for 500 epochs. The trained model checkpoint with the weights can be found under `lightning_logs/version_xy/checkpoints`

# Prediction

There is also a script for doing prediction using the data in the TROPOMI archive.

```
python -m methane_super_emitters.predict --help
```

Which will bring up the following help page.

```
Usage: python -m methane_super_emitters.predict [OPTIONS]

Options:
  -c, --checkpoint TEXT  Checkpoint file
  -i, --input-dir TEXT   Directory with files
  -o, --output-dir TEXT  Output directory
  -n, --n-jobs INTEGER   Number of parallel jobs
  -t, --threshold FLOAT  Threshold for the value of the sigmoid output to
                         qualify as a hit
  --help                 Show this message and exit.
```

The input directory is a directory with the yearly TROPOMI archive (same as during dataset creation). The output directory is used to store patches that are predicted to be super-emitters with a probability above the given threshold.

Here is a sample call:

```
python3 -m methane_super_emitters.predict -c lightning_logs/version_10/checkpoints/epoch\=99-step\=23400.ckpt -d ../dataset_v3 -i /dss/dsstbyfs03/pn56su/pn56su-dss-0022/Sentinel-5p/L2/CH4/2018/ -o ../predictions_2018 -n 40
```

# API Documentation

https://dlr-mf-das.github.io/methane-super-emitters/methane_super_emitters.html
