import os
import wget
import argparse

def download_url(url, destination):
    os.makedirs(destination, exist_ok=True)
    wget.download(url, destination)

def download_dataset(force=False):
    if not force and os.path.exists(
        "data/original/cifar-10/cifar-10-python.tar.gz"
    ):
        print("Dataset already exists on disk. Not downloading.\n"
              "Pass the flag --force to remove existing file and re-download")
    else:
        if force:
            os.remove("data/original/cifar-10/cifar-10-python.tar.gz")
        download_url(
            "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz",
            "data/original/cifar-10/"
        )

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Download the CIFAR-10 dataset')
    parser.add_argument(
        '--force', default=False, action="store_true",
        help="Download the dataset again, even if it exists on disk."
    )
    
    args = parser.parse_args()
    download_dataset(args.force)