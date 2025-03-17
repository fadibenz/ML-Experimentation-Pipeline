import optuna.exceptions
import torch
from make_train import train_epoch
from make_predict import predict
from architecture.BasicConvNet import BasicConvNet
import torch.nn as nn
from data.make_dataset import make_dataset
from torchvision.transforms import v2
import wandb
import torch.optim as optim
from pathlib import Path

def objective(trial):

    lr = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
    batch_size = trial.suggest_categorical('batch_size', [64, 128, 256])
    dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.7)
    num_filters = trial.suggest_int('num_filters', 16, 64, step=16)
    hidden_size = trial.suggest_categorical('hidden_size', [128, 256, 512])

    wandb_run = wandb.init(
        project='optuna_wandb_CIFAR10',
        reinit=True,
        config={
            'lr': lr,
            'architecture': 'Basic ConvNet',
            'dataset': 'CIFAR10',
            'dropout_rate': dropout_rate,
            'batch_size': batch_size,
            'num_filters': num_filters,
            'hidden_size': hidden_size
        },
        tags=['optuna', 'CIFAR10', 'Basic ConvNet'],
        mode='online'
    )

    epochs = 8
    model = BasicConvNet()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device=device)
    optimizer = optim.Adam(model.parameters(),
                           lr=lr)

    transform = v2.Compose(
        [
            v2.ToTensor(),
            v2.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
    )

    train_loader, val_loader, test_loader = make_dataset(
        batch_size=batch_size,
        val_split=0.1,
        transform=transform,
        num_workers=4,
        root='../../data/processed',
    )

    criterion = nn.CrossEntropyLoss()
    best_val_accuracy = 0

    for epoch in range(epochs):
        train_epoch(model = model,
                    criterion = criterion,
                    device=device,
                    optimizer=optimizer,
                    train_loader=train_loader,
                    epoch= epoch,
                    wandb_run=wandb_run)

        val_loss, val_acc = predict(
            model=model,
            criterion=criterion,
            val_loader=val_loader,
            device=device,
            epoch=epoch,
            wandb_run=wandb_run
        )

        trial.report(val_acc, epoch)

        if trial.should_prune():
            wandb_run.finish(exit_code=1)
            raise optuna.exceptions.TrialPruned()

        if val_acc > best_val_accuracy:

            root_path = Path('../../models')
            root_path.mkdir(parents=True, exist_ok=True)

            best_val_accuracy = val_acc
            torch.save(model.state_dict(),
                       root_path / f"model_trial_{trial.number}_epoch_{epoch}.pt")

    wandb_run.finish()
    return best_val_accuracy