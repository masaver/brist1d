# DataScienTest Project - Brist1D

This is the project work for the datascientest bootcamp DS-Engineer started in September 2024.

## Challenge

[BrisT1D Blood Glucose Prediction Competition](https://www.kaggle.com/competitions/brist1d)

## Goal

Goal: Forecast blood glucose levels one hour ahead using the previous six hours of participant data.

## Team

* [@masaver](https://github.com/masaver)
* [@rabbl](https://github.com/rabbl)
* [@sander-NULL](https://github.com/sander-NULL)
* [@svetigreen](https://github.com/svetigreen)

## References

www.kaggle.com/competitions/brist1d/overview/$citation

## Installation

Make sure you have [pyenv](https://github.com/pyenv/pyenv) and `unzip` installed. 
You can install it with your favorite package manager.

On Mac with brew:
```
$ brew install pyenv unzip
```

The project contains a `.python-version` file that specifies the Python version to use. 
You can install it with pyenv:

```
$ pyenv install
```

There is a makefile that can be used to install the project.
All make commands are in the Makefile.

List commands with:

```
$ make help
```

Install all requirements:

```
$ make install
```

List all installed packages:

```
$ make list
```

Reset the environment:

```
$ make reset
```

## Data

The data is available in the `data` folder.

### Run preprocessing steps

To run the preprocessing script, use the following command:

```bash
$ make preprocess
```

or 

```bash
python src/features/01-extract-patient-data.py
```

Project Organization
------------

    ├── LICENSE
    ├── README.md          <- The top-level README for developers using this project.
    ├── data               <- Data folder
    │   ├── data.zip       <- The original data from the competition comressed in a zip file
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's name, and a short `-` delimited description, e.g.
    │                         `1.0-alban-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, links, and all other explanatory materials.
    │
    ├── reports            <- The reports that you'll make during this project as PDF
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── .python-version    <- The Python version to used in the project by pyenvz`
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   ├── visualization  <- Scripts to create exploratory and results oriented visualizations
    │   │   └── visualize.py

--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
