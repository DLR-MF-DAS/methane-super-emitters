# Methane Super Emitter Detection from TROPOMI Data

The goal of this project is to detect methane super emitters in TROPOMI data.

# Installation

To install in a new conda environment:

```
git clone https://github.com/DLR-MF-DAS/methane-super-emitters.git
cd methane_super_emitters
pip install -e .
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

```

# API Documentation

https://dlr-mf-das.github.io/methane-super-emitters/methane_super_emitters.html
