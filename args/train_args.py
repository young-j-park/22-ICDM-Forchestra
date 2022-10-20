
from typing import Dict
from dataclasses import dataclass

from args.base_args import BaseArguments


@dataclass
class TrainingArguments(BaseArguments):
    skip_train: bool

    train_dataset: str
    test_dataset: str
    output_fname_prefix: str

    base_model_hist_len: int
    pred_len: int

    num_warmup_steps: int
    num_training_steps: int
    report_interval_steps: int
    max_epoch: int
    num_cycles: int
    tol_epoch: int
    early_stop: int
    init_lr: float
    lr_gamma: float
    weight_decay: float
    batch_size: int
    validation_days: int

    train_dataset_len: int = None
    test_dataset_len: int = None

    def __post_init__(self):
        if self.output_fname_prefix is not None and self.output_fname_prefix.endswith('.npz'):
            self.output_fname_prefix = self.output_fname_prefix[:-4]

    def update_args(self, train_ds_meta: Dict, test_ds_meta: Dict):
        self.train_dataset_len = train_ds_meta['dataset_days']
        self.test_dataset_len = test_ds_meta['dataset_days']
