import argparse

import torchvision.transforms as transforms

# TODO: Should import cifar_100 or other dataset and resnet_34 or other model based on config
from src.dataset.cifar_100 import get_dataloader
from src.model.resnet_34 import Model


def train():
    # Dataset
    # TODO: Move tfms and get_dl into a different function, parametrize with config
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
    dl = get_dataloader("data/csvs/train.csv", tfms, 64, True, 20)

    # Model
    # TODO: Move tmodel loading into a different function, parametrize with config
    model = Model()
    
    # TODO: Trainer


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train the model'
    )
    args = parser.parse_args()
    train()