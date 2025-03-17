import torch
import torch.nn as nn
import torch.utils.data as data
from tqdm import tqdm


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
    

    with tqdm(val_loader, desc=f"Epoch {epoch+1} [Valid]", unit="batch") as t:
        for batch_idx, (X, y) in enumerate(t, 0):
            X, y = X.to(device=device), y.to(device=device)
            with torch.no_grad():
                pred = model(X)
                loss = criterion(pred, y)
            accuracy = torch.mean((torch.argmax(pred, dim=1) == y).float()).item() * 100
            running_loss += loss.item()
            running_accuracy += accuracy
            total += 1
            
            t.set_postfix(loss=f"{loss.item():.4f}", accuracy=f"{accuracy:.2f}%")
            
            # Logging
            if wandb_run and batch_idx % log_freq == log_freq - 1:
                wandb_run.log({
                    'validation_loss': loss.item(),
                    'validation_accuracy': accuracy,
                    'epoch': epoch
                })
    val_loss = running_loss / total
    val_acc = running_accuracy / total
    print(f'[{epoch + 1}] '
          f'validation_loss: {val_loss:.5f} '
          f'validation_accuracy: {val_acc:.5f}')
    return val_loss, val_acc