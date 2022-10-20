
from dataclasses import dataclass

from args.base_args import BaseArguments


@dataclass
class BaseModelArguments(BaseArguments):

    base_model_name: str
    base_model_num_layers: int
    base_model_emb_dim: int
    base_model_n_head: int
    base_model_hist_len: int
    pred_len: int

    activation: str
    dropout: float

    base_model_path: str

    def __post_init__(self):
        self.model_name = self.base_model_name
        self.num_layers = self.base_model_num_layers
        self.emb_dim = self.base_model_emb_dim
        self.n_head = self.base_model_n_head
        self.hist_len = self.base_model_hist_len
        self.model_load_path = self.base_model_path
