import os
import shutil
import tarfile
import argparse

def extract_file(source, destination):
    os.makedirs(destination, exist_ok=True)
    tf = tarfile.open(source)
    tf.extractall(destination)

def extract_dataset(force=False):
    if not force and os.path.exists(
        "data/processed/cifar-10/extracted/cifar-10-batches-py"
    ):
        print("Extracted copy already exists on disk. Not Extracting.\n"
              "Pass the flag --force to remove existing files and re-extract")
    else:
        if force:
            shutil.rmtree("data/processed/cifar-10/extracted/cifar-10-batches-py")
        extract_file(
            "data/original/cifar-10/cifar-10-python.tar.gz",
            "data/processed/cifar-10/extracted"
        )

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract the CIFAR-10 dataset')
    parser.add_argument(
        '--force', default=False, action="store_true",
        help="Extract the dataset again, even if it exists on disk."
    )
    
    args = parser.parse_args()
    extract_dataset(args.force)