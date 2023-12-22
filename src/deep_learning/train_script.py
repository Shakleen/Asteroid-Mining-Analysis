import os

import torch
from torch.optim import AdamW, lr_scheduler

torch.manual_seed(29)


def train(model, optimizer, scaler, train_loader, device):
    train_loss = 0
    model.train()

    for X, y in train_loader:
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


def validate(model, data_loader, device):
    model.eval()
    valid_loss = 0

    for X, y in data_loader:
        X = X.to(device)
        y = y.to(device)

        pred = model(X)

        loss = model.loss(y, pred)
        valid_loss += loss.item()

    return valid_loss


def train_epoch(
    model,
    device,
    num_epochs,
    learning_rate,
    gamma,
    patience,
    root_save_dir,
    model_name,
    train_loader,
    valid_loader,
):
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    scheduler = lr_scheduler.ExponentialLR(
        optimizer,
        gamma=gamma,
        last_epoch=-1,
        verbose=False,
    )
    model_save_file = os.path.join(root_save_dir, model_name)
    scaler = torch.cuda.amp.GradScaler()
    min_eval_loss = float("inf")
    early_stopping_hook = 0
    log_file_path = os.path.join(root_save_dir, "log.txt")

    if os.path.exists(log_file_path):
        os.remove(log_file_path)

    for epoch in range(num_epochs):
        train_loss = train(model, optimizer, scaler, train_loader, device)
        valid_loss = validate(model, valid_loader, device)

        log_file = open(log_file_path, "a")
        print(
            "Epoch: {} - Training loss: {} - Eval Loss: {} - LR: {}".format(
                epoch + 1,
                train_loss / len(train_loader),
                valid_loss / len(valid_loader),
                optimizer.param_groups[0]["lr"],
            ),
            file=log_file,
        )
        scheduler.step()

        if valid_loss < min_eval_loss:
            torch.save(model.state_dict(), model_save_file)
            print(f"Saved model to {model_save_file}", file=log_file)
            min_eval_loss = valid_loss
            early_stopping_hook = 0
        else:
            early_stopping_hook += 1

            if early_stopping_hook > patience:
                break

        log_file.close()

    print("The training process has done!")
    return model
