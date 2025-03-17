import torch
import torch.nn as nn
import torch.utils.data as data
from requests_toolbelt.multipart.encoder import total_len


def predict(model:nn.Module,
            criterion: nn.Module,
            val_loader:data.DataLoader,
            device:torch.device,
            epoch:int,
            wandb_run=None,
            log_freq=20):

    model.eval()
    running_loss = 0
    running_accuracy = 0
    total = 0
    for batch_idx, (X, y) in enumerate(val_loader, 0):
        X, y = X.to(device=device), y.to(device=device)
        pred = model(X)

        loss = criterion(pred, y)
        accuracy = torch.mean(torch.argmax(X, 1) == y).item() * 100

        running_loss += loss.item()
        running_accuracy += accuracy
        total += 1
        # Logging
        if wandb_run and batch_idx % log_freq == log_freq - 1:
           wandb_run.log(
               {
                   'validation_loss': loss.item(),
                   'validation_accuracy': accuracy,
                   'epoch': epoch
               }
           )

    val_loss = running_loss / total
    val_acc = running_accuracy / total

    print(f'[{epoch + 1}] '
          f'validation_loss: {running_loss :.5f} '
          f'validation_accuracy: {val_acc:.5f}')

    return val_loss, val_acc