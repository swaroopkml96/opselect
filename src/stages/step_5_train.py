import argparse

import torch
import torchvision.transforms as transforms

from src.dataset.cifar_100 import get_dataloader
from src.model.resnet_34 import Model
from src.train.trainer import Trainer


def train():
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
    dl = get_dataloader("data/csvs/train.csv", tfms, 64, False, 20)

    # Model
    model = Model().to('cuda:0')

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=0.01, betas=[0.9, 0.999])
    
    # Trainer
    trainer = Trainer(dl, dl, model, optimizer)

    # Fit
    trainer.fit()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train the model'
    )
    args = parser.parse_args()
    train()