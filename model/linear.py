
import torch.nn as nn

from model.base import BaseModel


class LinearMetaLearner(BaseModel):
    def __init__(
            self, repr_dim, num_base_models, **kwargs
    ):
        super().__init__(**kwargs)
        self.linear = nn.Sequential(
            nn.Linear(repr_dim, num_base_models),
            self.final_activation,
        )
        self.to_device()

    def forward(self, representations):
        return self.linear(representations)
