<div align="center">

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5"></a>

# Ellipse R-CNN

</div>

A PyTorch (Lightning) implementation of Ellipse R-CNN. Originally developed for [another project](https://github.com/wdoppenberg/crater-detection), it has proven succesful in predicting instanced ellipses.
The methodology is based on [this paper](https://arxiv.org/abs/2001.11584), albeit different in the sense that this model uses the regressed bounding box predictions instead of region proposals as the base for predicted normalised ellipse parameters.


## Setup

Make sure you have [uv](https://docs.astral.sh/uv/getting-started/installation/) installed.

From the project root, run:

```shell
uv sync
```

This sets up a new virtual environment and installs all packages into it.

Get info on either the training or sample (prediction) script using:

```shell
uv run train.py --help
# or
uv run sample.py --help
```

Currently the training script only supports training with FDDB. See the required steps for
getting & structuring the data below. More datasets will be supported later.

## FDDB Data

### Download
Visit https://vis-www.cs.umass.edu/fddb/ and download:
* Original, unannotated set of images
* Face annotations


### File structure

Download & unzip until you achieve the following file structure (from root):


```
data
└── FDDB
   ├── images
   │  ├── 2002
   │  └── 2003
   └── labels
      ├── FDDB-fold-01-ellipseList.txt
      ├── FDDB-fold-01.txt
      ├── FDDB-fold-02-ellipseList.txt
      ├── FDDB-fold-02.txt
      ├── ...
```
