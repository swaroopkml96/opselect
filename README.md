# SurgicalToolDetection
Presence detection and classification of instruments in laparoscopic cholecystectomy videos

## Folder Structure 

    ├── configs            # config files like toml/yaml/json to save parameters
    ├── data
    │   ├── original       # Unmodified or unprocessed data
    │   ├── processed      # Preprocessed datasets
    │   ├── csvs           # Meta data in csv format
    ├── docs               # API docs with their dependencies and methodology
    ├── models             # Pretrained model files like pth/checkpoint
    ├── notebooks          
    │   ├── ipynb          # Jupyter notebooks
    │   ├── py             # python scripts of notebooks
    ├── reports            # experiment result's plots, metrics, logs and csvs
    ├── src                # Source code for use in this project
    │   ├── dataset        # Scripts to data loaders and its pipeline
    │   ├── model          # Scripts to handle model to train and evaluate
    │   ├── preprocessing  # Scripts to download or generate and preprocess data
    │   ├── util           # Scripts to other genric helper classes like metrics and report generator
    │   ├── evaluate.py    # Scripts to evaluate model with ground truth
    │   ├── infer.py       # Scripts to infer model without ground truth
    │   └── train.py       # Scripts to train models
    ├── test               # Unit testing scripts
    ├── Makefile           # Makefile with commands like `make data` or `make train`
    ├── requirements.txt   # The requirements file for reproducing the environment
    ├── README.md          # The top-level README for developers using this project

## Branch Structure

## Dependencies
- Hardware and os dependencies

## Assumptions

## Config parameters

## Getting Started
### Setup

### Run

## Limitations

## References