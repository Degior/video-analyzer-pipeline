import torch.nn as nn


class LogReg(nn.Module):
    def __init__(self, input_dim=512, num_classes=101):
        super().__init__()
        self.linear = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return self.linear(x)



