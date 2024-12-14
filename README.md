# InternetUseClassification

## Overview
This repository contains the code, data, and resources for my Internet Use Classification DATA 1030 final project. The repo is laid out as follows:
```bash
.
├── data/         # Contains all raw
├── figures/      # Stores generated figures and visualizations
├── results/      # Includes predictions, saved models, and other output results (empty)
├── report/       # Midterm and final presentations and the final report
├── src/          # Source code (Jupyter Notebooks for EDA and ML training)
├── .gitignore    # Specifies files/folders to ignore in Git
├── LICENSE       # License file for the repository
└── README.md     # This readme file
```

Data can be downloaded from this [link](https://www.kaggle.com/competitions/child-mind-institute-problematic-internet-use/data) (requires a Kaggle account).

## Python and Package Versions
This project used a modified version of the data1030 environment from class which uses Python 3.12.5 and includes various scientific packages such as ```sklearn``` and ```xgboost```. Specifically, ```seaborn```, ```fastparquet```, and ```tensorflow``` were also added to the list of dependencies. An included ```.yml``` file can be used to set up the environment.

To set up the environment, use the following command from the main directory:

```bash
conda env create -f internet_classification.yml
```