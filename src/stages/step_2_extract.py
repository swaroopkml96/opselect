import os
import shutil
import tarfile
import argparse

def extract_file(source, destination):
    os.makedirs(destination, exist_ok=True)
    tf = tarfile.open(source)
    tf.extractall(destination)

def extract_dataset(force=False):
    data_file_path = os.path.join(
        "data", "original", "cifar-10", "cifar-10-python.tar.gz"
    )
    extract_location = os.path.join(
        "data", "processed", "cifar-10", "extracted"
    )
    extracted_folder_path = os.path.join(
        extract_location, "cifar-10-batches-py"
    )

    if not force and os.path.exists(extracted_folder_path):
        print("Extracted copy already exists on disk. Not Extracting.\n"
              "Pass the flag --force to remove existing files and re-extract")
    else:
        if force:
            shutil.rmtree(extracted_folder_path)
        extract_file(data_file_path, extract_location)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract the CIFAR-10 dataset')
    parser.add_argument(
        '--force', default=False, action="store_true",
        help="Extract the dataset again, even if it exists on disk."
    )
    
    args = parser.parse_args()
    extract_dataset(args.force)