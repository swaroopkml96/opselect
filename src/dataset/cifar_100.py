import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader

class ClassifierDataset(Dataset):
    def __init__(self, csv_path, tfms, classes):
        self.tfms = tfms
        self.classes = classes
        self.num_classes = len(self.classes)

        df = pd.read_csv(csv_path)
        self.img_paths = df["img_path"].tolist()

        labels = df["label"].tolist()
        self.label_idx = [self.classes.index(l) for l in labels]

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img = Image.open(self.img_paths[idx]).convert("RGB")
        img = self.tfms(img)

        label = self.label_idx[idx]
        label_ohe = torch.zeros(self.num_classes)
        label_ohe[label] = 1

        return img, label_ohe


def get_dataloader(
    csv_path, tfms, batch_size, classes, shuffle=True, num_workers=4
):
    ds = ClassifierDataset(csv_path, tfms, classes)
    dl = DataLoader(ds, batch_size, shuffle, num_workers=num_workers)
    return dl


def _find_mean_std(ds, num_samples):
    imgs = torch.zeros(num_samples, 3, 32, 32)
    for i in range(num_samples):
        img = ds[i][0]
        imgs[i] = img
    mean = imgs.mean(axis=(0,2,3))
    std = imgs.std(axis=(0,2,3))
    return mean, std
    

if __name__ == '__main__':
    import torchvision.transforms as transforms
    
    # Dataset
    tfms = transforms.Compose([
        transforms.ToTensor()
    ])
    classes = [
        'horse', 'bird', 'truck', 'ship', 'dog', 'cat', 'frog', 
        'automobile', 'deer', 'airplane'
    ]

    ds = ClassifierDataset("data/csvs/valid.csv", tfms, classes)

    # Normalize
    mean, std = _find_mean_std(ds, 1000)
    print(f"Mean:\n{mean}")
    print(f"Std.:\n{std}")

    ### OUTPUT ###
    # Mean:
    # tensor([0.4955, 0.4838, 0.4482])
    # Std.:
    # tensor([0.2433, 0.2407, 0.2575])



    