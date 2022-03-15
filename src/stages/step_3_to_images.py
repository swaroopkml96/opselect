import os
import pickle
import argparse

import numpy as np
def load_batch(batch_path):
    with open(batch_path, 'rb') as fo:
        batch = pickle.load(fo, encoding='bytes')
    return batch

def transform_dataset(force=False):
    batch = load_batch("data/processed/cifar-10/extracted/cifar-10-batches-py/data_batch_1")
    print(batch.keys())

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Load the extracted CIFAR-10 dataset and save it as images.'
    )
    parser.add_argument(
        '--force', default=False, action="store_true",
        help="Transform the dataset again, even if it exists on disk."
    )
    
    args = parser.parse_args()
    transform_dataset(args.force)