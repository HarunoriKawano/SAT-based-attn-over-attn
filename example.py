import json

import torch
from torch import nn
from torch.nn.functional import cross_entropy

from model import SATConfig, SATFramework


class ExampleLowLayers(nn.Module):
    def __init__(self, out_hidden_size: int):
        super().__init__()
        self.linear = nn.Linear(80, out_hidden_size)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden_states = self.linear(inputs)
        return hidden_states


class ExampleHighLayers(nn.Module):
    def __init__(self, input_hidden_size: int):
        super().__init__()
        self.linear1 = nn.Linear(input_hidden_size, 512)
        self.linear2 = nn.Linear(512, 512)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.linear1(hidden_states)
        hidden_states = self.linear2(hidden_states)

        return hidden_states


if __name__ == '__main__':
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Use device: {device}")

    with open("config.json", "r", encoding="utf-8") as f:
        config = SATConfig(**json.load(f))

    low_layers = ExampleLowLayers(config.hidden_size).to(device)
    high_layers = ExampleHighLayers(config.hidden_size + config.d_vector_size).to(device)

    # Memory is created by d-vector model. '(N, D)'
    memory = torch.rand(config.num_d_vectors, config.d_vector_size).to(device)

    model = SATFramework(config, low_layers, high_layers).to(device)
    model.set_memory(memory)

    # `(B, T, D)`
    inputs = torch.rand(4, 1000, 80).to(device)

    num_target_classes = 100
    targets = torch.randint(num_target_classes, size=(inputs.size(0) * inputs.size(1),)).to(device)
    out_linear = nn.Linear(512, num_target_classes).to(device)

    out = model(inputs)
    out = out_linear(out).view(out.size(0) * out.size(1), -1)

    loss = cross_entropy(out, targets)

    print(loss)
    loss.backward()
