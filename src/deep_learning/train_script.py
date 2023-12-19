import pickle
import os

import torch
from torch.optim import AdamW, lr_scheduler
from tqdm import tqdm

from .mlp import MLP

torch.manual_seed(29)


def train(model, optimizer, scaler, train_loader, device):
    train_loss = 0
    model.train()

    for _, (X, y) in zip(
        tqdm(range(len(train_loader)), desc="Training batch"),
        train_loader,
    ):
        X = X.to(device)
        y = y.to(device)

        pred = model(X)

        loss = model.loss(y, pred)
        train_loss += loss.item()
        optimizer.zero_grad()

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

    return train_loss


def validate(model, data_loader, desc, device):
    model.eval()
    valid_loss = 0

    for _, batch in zip(
        tqdm(range(len(data_loader)), desc=desc),
        data_loader,
    ):
        X, y = batch
        X = X.to(device)
        y = y.to(device)

        pred = model(X)

        loss = model.loss(y, pred)
        valid_loss += loss.item()

    return valid_loss


def train_epoch(
    model,
    num_epochs,
    learning_rate,
    patience,
    root_save_dir,
    model_name,
    train_loader,
    valid_loader,
):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = MLP(
        3,
        num_output_list=[256, 128, 64],
        dropout_list=[0.2, 0.15, 0.1],
        device=device,
    )
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    scheduler = lr_scheduler.ExponentialLR(
        optimizer,
        gamma=0.9,
        last_epoch=-1,
        verbose=False,
    )

    for epoch in range(num_epochs):
        train_loss = train()
        valid_loss = validate(valid_loader, "Validating batch")

        print(
            "Epoch: {} - Training loss: {} - Eval Loss: {} - LR: {}".format(
                epoch + 1,
                train_loss / len(train_loader),
                valid_loss / len(valid_loader),
                optimizer.param_groups[0]["lr"],
            )
        )
        scheduler.step()

        if valid_loss < min_eval_loss:
            model_save_file = os.path.join(root_save_dir, model_name)
            torch.save(model.state_dict(), model_save_file)
            print(f"Saved model to {model_save_file}")
            min_eval_loss = valid_loss
            early_stopping_hook = 0
        else:
            early_stopping_hook += 1

            if early_stopping_hook > patience:
                break

    print("The training process has done!")
