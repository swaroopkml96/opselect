import os
import argparse
from random import shuffle

import numpy as np
import pandas as pd

from src.util.data_helpers import get_files, get_labels
from src.util.config_helpers import load_config


def get_df(images_path):
    file_paths = get_files(images_path)
    labels = get_labels(file_paths)
    df = pd.DataFrame({
        "img_path": file_paths,
        "label": labels
    })
    return df

def generate_csv():
    input_path = os.path.join("data", "processed", "cifar-10", "rgb")
    output_path = os.path.join("data", "csvs")
    cfg = load_config()

    # Test
    images_path = os.path.join(input_path, "test")
    df = get_df(images_path)
    df.to_csv(os.path.join(output_path, "test.csv"), index=None)

    # Train and validation
    images_path = os.path.join(input_path, "train")
    df = get_df(images_path)

    np.random.seed(cfg["general"]["random_seed"])
    shuffled_df = df.sample(frac=1)
    valid_df = shuffled_df[:int(cfg["data"]["val_ratio"]*len(shuffled_df))]
    train_df = shuffled_df[int(cfg["data"]["val_ratio"]*len(shuffled_df)):]
    
    valid_df.to_csv(os.path.join(output_path, f"valid.csv"), index=None)
    train_df.to_csv(os.path.join(output_path, f"train.csv"), index=None)
    
    
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Generate train, val and test csvs from the folder containing images'
    )
    args = parser.parse_args()
    generate_csv()