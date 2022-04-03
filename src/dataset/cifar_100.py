import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader

class ClassifierDataset(Dataset):
    def __init__(self, csv_path, tfms):
        self.tfms = tfms

        df = pd.read_csv(csv_path)
        self.img_paths = df["img_path"].tolist()

        labels = df["label"].tolist()
        self.classes = sorted(list(set(labels)))
        self.label_idx = [self.classes.index(l) for l in labels]

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img = Image.open(self.img_paths[idx]).convert("RGB")
        img = self.tfms(img)

        label = self.label_idx[idx]
        label = torch.tensor(label).long()

        return img, label


def get_dataloader(csv_path, tfms, batch_size, shuffle=True, num_workers=4):
    ds = ClassifierDataset(csv_path, tfms)
    dl = DataLoader(ds, batch_size, shuffle, num_workers=num_workers)
    return dl


if __name__ == '__main__':
    import torch
    import torchvision.transforms as transforms
    
    # Dataset
    tfms = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(30),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.4444, 0.4360, 0.4028),
            std=(0.2718, 0.2683, 0.2790)
        )
    ])
    ds = ClassifierDataset("data/csvs/valid.csv", tfms)

    # Normalize
    imgs = torch.zeros(1000, 3, 32, 32)
    for i in range(1000):
        img = ds[i][0]
        imgs[i] = img
    print(f"Mean:\n{imgs.mean(axis=(0,2,3))}")
    print(f"Std.:\n{imgs.std(axis=(0,2,3))}")
    
    # Dataloader
    dl = get_dataloader("data/csvs/valid.csv", tfms, 64, False, 20)
    print(dl)



    