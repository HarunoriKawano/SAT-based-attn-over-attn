from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    d_vector_size: int  # Dimension of d-vector
    num_d_vectors: int  # Number of d-vectors in memory
    hidden_size: int  # Dimension of input hidden size
