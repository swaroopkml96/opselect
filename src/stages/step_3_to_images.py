import os
import pickle
import argparse
from tqdm import tqdm as tqdm

import numpy as np
import matplotlib.pyplot as plt


def load_batch(batch_path):
    with open(batch_path, 'rb') as fo:
        batch = pickle.load(fo, encoding='bytes')
    return batch

def save_batch_images(batch_path, save_path, pbar):
    batch = load_batch(batch_path)
    for i in range(len(batch[b'data'])):
        img = batch[b'data'][i]
        fname = batch[b'filenames'][i]
        img_red = np.reshape(img[:32*32], [32, 32])
        img_green = np.reshape(img[32*32:2*32*32], [32, 32])
        img_blue = np.reshape(img[2*32*32:], [32, 32])
        
        img_rgb = np.dstack([img_red, img_green, img_blue])
        plt.imsave(os.path.join(save_path, fname.decode('ascii')), img_rgb)
        pbar.update(1)


def transform_dataset(force=False):
    rgb_path = os.path.join("data", "processed", "cifar-10", "rgb")
    if not force and os.path.exists(rgb_path):
        print("RGB images already exist on disk. Not converting.\n"
              "Pass the flag --force to convert while replacing existing files.")
    else:
        # Train
        save_path = os.path.join(rgb_path, "train")
        pbar = tqdm(total=50_000, desc="Converting [train]")
        os.makedirs(save_path, exist_ok=True)
        for batch_id in range(1, 6):
            batch_path = os.path.join(
                "data", "processed", "cifar-10", "extracted", "cifar-10-batches-py", 
                f"data_batch_{batch_id}"
            )
            save_batch_images(batch_path, save_path, pbar)

        # Test
        save_path = os.path.join(rgb_path, "test")
        pbar = tqdm(total=10_000, desc="Converting [test]")
        os.makedirs(save_path, exist_ok=True)
        
        batch_path = os.path.join(
            "data", "processed", "cifar-10", "extracted", "cifar-10-batches-py", 
            "test_batch"
        )
        save_batch_images(batch_path, save_path, pbar)
        
        
        

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