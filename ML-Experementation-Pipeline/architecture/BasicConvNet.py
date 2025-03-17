import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicConvNet(nn.Module):
    def __init__(self, nb_channels=16,
                 hidden_size= 128,
                 nb_classes = 10,
                 kernel_size:int = 3,
                 pool_kernel:int = 3,
                 p_dropout:float = 0.1 ):

        super().__init__()
        self.conv1 = nn.Conv2d(3, nb_channels, kernel_size, padding=1)
        self.bn1 = nn.BatchNorm2d(nb_channels)

        self.conv2 = nn.Conv2d(nb_channels, nb_channels * 2, kernel_size, padding=1)
        self.bn2 = nn.BatchNorm2d(nb_channels * 2)

        self.conv3 = nn.Conv2d(nb_channels * 2, nb_channels * 4, kernel_size, padding=1)
        self.bn3 = nn.BatchNorm2d(nb_channels * 4)

        self.pool = nn.MaxPool2d(pool_kernel)

        self.fc1 = nn.Linear((nb_channels * 4) * 4 * 4, hidden_size)
        self.fc2 = nn.Linear(hidden_size, nb_classes)

        self.dropout = nn.Dropout(p_dropout)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))

        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x


