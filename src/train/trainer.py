from tqdm import tqdm as tqdm

import numpy as np
import torch
import torch.nn as nn

class Trainer:
    def __init__(self, train_dl, valid_dl, model, optimizer, device):
        self.train_dl = train_dl
        self.valid_dl = valid_dl
        self.model = model
        self.optimizer = optimizer
        self.device = device

    def _run_epoch(self, epoch, split):
        model = self.model
        optimizer = model.configure_optimizers()

        is_train = split == 'train'
        model.train(is_train)
        loader = self.train_dl if is_train else self.valid_dl

        losses = []
        pbar = tqdm(enumerate(loader), total=len(loader)
                    ) if is_train else enumerate(loader)
        for it, (x, y) in pbar:

            # place data on the correct device
            x = x.to(self.device)
            y = y.to(self.device)

            # forward the model
            with torch.set_grad_enabled(is_train):
                out = model(x)
                loss = nn.BCEWithLogitsLoss()(out, y)
                # collapse all losses if they are scattered on multiple gpus
                loss = loss.mean()
                losses.append(loss.item())

            if is_train:
                # backprop and update the parameters
                model.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), 1.0)
                optimizer.step()

                # report progress
                pbar.set_description(
                    f"epoch {epoch+1} iter {it}: train loss {loss.item():.5f}")

        if not is_train:
            valid_loss = float(np.mean(losses))
            print(f"epoch {epoch+1}: validation loss {valid_loss:.5f}")
            return valid_loss

    def fit(self):

        best_loss = float('inf')
        for epoch in range(10):
            self._run_epoch(epoch, 'train')
            if self.valid_dl is not None:
                valid_loss = self._run_epoch(epoch, 'valid')

            # supports early stopping based on the validation loss, or just save
            # always if no validation set is provided
            good_model = self.valid_dl is None or valid_loss < best_loss
            # if self.args.save_path is not None and good_model:
            #     best_loss = valid_loss