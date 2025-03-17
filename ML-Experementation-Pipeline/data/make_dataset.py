import torch
import torch.utils.data as data
import torchvision
from pathlib import Path

def make_dataset(batch_size:int,
                 val_split:float,
                 transform,
                 num_workers:int = 4,
                 root:str = '../../data/processed') -> tuple:


    seed = 42

    # Downloading Data
    root_path = Path(root)
    root_path.mkdir(parents=True, exist_ok=True)

    train_data = torchvision.datasets.CIFAR10(
        root=str(root_path),
        train=True,
        download=True,
        transform=transform,
    )

    test_data = torchvision.datasets.CIFAR10(
        root=str(root_path),
        train=False,
        download=True,
        transform=transform,

    )

    # Test-Val Split
    val_size = int(val_split * len(train_data))
    train_size = len(train_data) - val_size
    generator = torch.Generator().manual_seed(seed)

    train_subset, val_subset = data.random_split(train_data,
                                        [train_size, val_size],
                                                 generator=generator)
    # Data Loading
    train_loader = data.DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = data.DataLoader(
      val_subset,
      batch_size =batch_size,
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = data.DataLoader(
        test_data,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader, test_loader