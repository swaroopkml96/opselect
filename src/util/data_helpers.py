import os

def get_files(images_path):
    return [os.path.join(images_path, p) for p in os.listdir(images_path)]

def get_labels(file_paths):
    return [os.path.basename(p).split('_')[0] for p in file_paths]