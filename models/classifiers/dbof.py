import torch
import torch.nn as nn


class DBoF(nn.Module):
    def __init__(self, input_dim=512, projection_dim=1024, num_classes=101, pooling='max'):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(input_dim, projection_dim),
            nn.ReLU(),
        )
        self.pooling = pooling
        self.classifier = nn.Linear(projection_dim, num_classes)

    def forward(self, x):
        x = self.projection(x)
        if self.pooling == 'max':
            x = torch.max(x, dim=1).values
        elif self.pooling == 'avg':
            x = torch.mean(x, dim=1)
        else:
            raise ValueError("pooling must be 'max' or 'avg'")
        return self.classifier(x)
