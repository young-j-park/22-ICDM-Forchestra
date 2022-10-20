
from dataclasses import dataclass

from args.base_args import BaseArguments


@dataclass
class ReprModelArguments(BaseArguments):
    repr_model_depth: int
    repr_model_input_dim: int
    repr_model_projection_dim: int
    repr_model_hidden_dim: int
    repr_model_output_dim: int
    repr_model_kernel_size: int
    repr_model_mask_mode: str
    repr_model_encoding_window: str

    repr_model_path: str = None
