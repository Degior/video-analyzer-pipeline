import torch
import torch.nn as nn


class MixtureOfExperts(nn.Module):
    def __init__(self, input_dim=512, num_classes=101, num_experts=8, hidden_size=256):
        super().__init__()
        self.num_experts = num_experts
        self.num_classes = num_classes

        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, num_classes)
            ) for _ in range(num_experts)
        ])

        self.gate = nn.Sequential(
            nn.Linear(input_dim, num_experts),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        gate_outputs = self.gate(x)
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=1)
        output = torch.sum(gate_outputs.unsqueeze(2) * expert_outputs, dim=1)
        return output
