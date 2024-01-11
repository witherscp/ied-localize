<h2 align="center">Source Localization of IEDs using GM and WM</h2>

<p align="center">
<a href="https://github.com/witherscp/ied-localize/blob/main/LICENSE.txt"><img alt="License: MIT" src="https://black.readthedocs.io/en/stable/_static/license.svg"></a>
<a href="https://github.com/psf/black"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>
</p>

## Description
Source code for the localization of interictal epileptiform discharge (IED) sequences. Processed data are available upon request.

---
## Installation
```pip install git+https://github.com/witherscp/ied-localize.git```

This installation adds `do_localize_source.py` to the command line for localization of the subjects. Users may also utilize any functions or methods implemented throughout the package via `import ied_localize`.

## Usage



```
do_localize_source.py [-h] [--cluster] [--only_gm] [--only_wm] [--variable_gm] [-p PARCS] [-n NETWORKS] [--dist DIST] [-l MAX_LENGTH] subj

to localize the putative source of interictal spike sequences using GM, WM, and spike timings

positional arguments:
  subj                  subject code

optional arguments:
  -h, --help            show this help message and exit
  --cluster             cluster to localize; defaults to 0 which means all clusters will be localized
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
C Price Withers, Joshua M Diamond, Braden Yang, Kathryn Snyder, Shervin Abdollahi, Joelle Sarlls, Julio I Chapeton, William H Theodore, Kareem A Zaghloul, Sara K Inati, Identifying sources of human interictal discharges with travelling wave and white matter propagation, _Brain_, Volume 146, Issue 12, December 2023, Pages 5168–5181, https://doi.org/10.1093/brain/awad259
