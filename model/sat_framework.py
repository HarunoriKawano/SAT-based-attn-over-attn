import torch
from torch import nn

from model.sat_config import SATConfig


class SATFramework(nn.Module):
    def __init__(self, config: SATConfig, low_layers: nn.Module, high_layers: nn.Module):
        super().__init__()
        self.low_layers = low_layers
        self.high_layers = high_layers
        self.m_linear = nn.Linear(config.hidden_size, config.d_vector_size)
        self.memory = nn.Parameter(torch.rand(config.num_d_vectors, config.d_vector_size))
        self.memory.requires_grad = False

    def set_memory(self, memory: torch.Tensor):
        self.memory.data = memory

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs (torch.Tensor): with shape `(B, T1, D1)`

        Returns:
            torch.Tensor with shape `(B, T2, D2)`
        """
        hidden_states = self.low_layers(inputs)
        num_steps = hidden_states.size(1)

        # Add speaker vectors. `(B, T, D)` -> `(B, T, D+N)`
        speaker_vectors = self._get_speaker_vectors(hidden_states)
        hidden_states = torch.cat([hidden_states, speaker_vectors.unsqueeze(-2).repeat(1, num_steps, 1)], dim=-1)

        hidden_states = self.high_layers(hidden_states)

        return hidden_states

    def _get_speaker_vectors(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states (torch.Tensor): with shape `(B, T, D)`

        Returns:
            torch.Tensor with shape `(B, N)`
        """

        # `(B, T, N)`
        similarity_degrees = torch.matmul(self.m_linear(hidden_states), self.memory.T)

        memory_level_attentions = torch.softmax(similarity_degrees, dim=-1)

        frame_level_attentions = torch.softmax(similarity_degrees, dim=-2).sum(-1) / similarity_degrees.size(-1)

        attention_scores = torch.matmul(
            memory_level_attentions.transpose(-1, -2), frame_level_attentions.unsqueeze(-1)
        ).squeeze()

        speaker_vectors = torch.matmul(attention_scores, self.memory)

        return speaker_vectors
