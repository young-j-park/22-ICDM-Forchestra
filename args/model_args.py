
from dataclasses import dataclass

from args.base_args import BaseArguments


@dataclass
class ModelArguments(BaseArguments):
    train_dataset: str

    meta_model_name: str
    num_base_models: int
    num_layers: int
    emb_dim: int
    pred_len: int
    activation: str
    dropout: float
    repr_model_output_dim: int

    model_load_path: str
    model_save_path: str

    def __post_init__(self):
        self.repr_dim = self.repr_model_output_dim
