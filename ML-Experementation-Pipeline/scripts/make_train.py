import torch
from torch import optim
from torch import nn
import torch.utils.data as data
from tqdm import tqdm

def train_epoch(model:nn.Module,
                criterion:nn.Module,
                device:torch.device,
                optimizer:optim.Optimizer,
                train_loader: data.DataLoader,
                epoch:int,
                print_freq=200,
                log_freq=20,
                wandb_run=None):
    model.train()
    running_loss = 0
    running_accuracy = 0
    
    with tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]", unit="batch") as t:
        for batch_idx, (X, y) in enumerate(t, 0):
            X, y = X.to(device=device), y.to(device=device)
            optimizer.zero_grad()
            pred = model(X)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            accuracy = torch.mean((torch.argmax(pred, dim=1) == y).float()).item() * 100
            running_loss += loss.item()
            running_accuracy += accuracy
            
            # Update tqdm with current loss and accuracy
            t.set_postfix(loss=f"{loss.item():.4f}", accuracy=f"{accuracy:.2f}%")
            
            # Logging
            if wandb_run and batch_idx % log_freq == log_freq - 1:
                wandb_run.log({
                    'train_loss': loss.item(),
                    'train_accuracy': accuracy,
                    'epoch': epoch
                })
            if batch_idx % print_freq == print_freq - 1:
                print(f'[{epoch + 1}, {batch_idx + 1:5d}] '
                      f'loss: {running_loss / print_freq:.5f} '
                      f'accuracy: {running_accuracy / print_freq:.5f}')
                running_loss = 0.0
                running_accuracy = 0.0