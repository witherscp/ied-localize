<h2 align="center">Source Localization of IEDs using GM and WM</h2>

<p align="center">
<a href="https://github.com/witherscp/ied-localize/blob/main/LICENSE.txt"><img alt="License: MIT" src="https://black.readthedocs.io/en/stable/_static/license.svg"></a>
<a href="https://github.com/psf/black"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>
</p>

## Description
Source code for the localization of interictal epileptiform discharge (IED) sequences. Patient data will be made available in the future.

---
## Installation
```pip install git+https://github.com/witherscp/ied-localize.git```

This installation adds `do_localize_source.py` to the command line for localization of the subjects. Users may also utilize any functions or methods implemented throughout the package via `import ied_localize`.

## Usage



```
do_localize_source.py [-h] [--only_gm] [--only_wm] [--fixed_gm] [-p PARCS] [-n NETWORKS] [--dist DIST] [-l MAX_LENGTH] subj

to localize the putative source of interictal spike sequences using GM, WM, and spike timings

positional arguments:
  subj                  subject code

optional arguments:
  -h, --help            show this help message and exit
  --only_gm             use gray matter localization only
  --only_wm             use white matter localization only
  --variable_gm         allow for variable GM velocity within a sequence; defaults to fixed
  -p PARCS, --parcs PARCS
                        Schaefer parcellation; defaults to 600
  -n NETWORKS, --networks NETWORKS
                        Yeo network {7 or 17}; defaults to 17
  --dist DIST           GM max search distance; defaults to 45 mm
  -l MAX_LENGTH, --max_length MAX_LENGTH
                        maximum allowable length of a sequence
```
---
## Citation
_pending_