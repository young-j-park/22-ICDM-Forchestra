
from datetime import datetime, timedelta
from dataclasses import dataclass

from args.base_args import BaseArguments
from config.config import TARGET_KEY, MASK_KEY


@dataclass
class DataArguments(BaseArguments):
    base_model_path: str
    repr_model_path: str
    repr_slide_padding: int

    train_dataset: str
    train_dataset_begin: str
    train_dataset_end: str

    test_dataset: str
    test_dataset_begin: str
    test_dataset_end: str

    base_model_hist_len: int
    pred_len: int

    def __post_init__(self):
        self.dynamic_catg_feats = [MASK_KEY]
        self.dynamic_cont_feats = [TARGET_KEY]
        self.max_hist_len = max(self.repr_slide_padding, self.base_model_hist_len)

        self.test_load_begin = (
                datetime.fromisoformat(self.test_dataset_begin)
                - timedelta(self.max_hist_len)
        ).strftime('%Y-%m-%d')
