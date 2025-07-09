# Methane Super Emitter Detection from TROPOMI Data

The goal of this project is to detect methane super emitters in TROPOMI data.

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

# API Documentation

https://dlr-mf-das.github.io/methane-super-emitters/methane_super_emitters.html
