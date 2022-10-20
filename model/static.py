
import torch
import torch.nn as nn

from model.base import BaseModel


class StaticMetaLearner(BaseModel):
    def __init__(self, num_base_models, **kwargs):
        super().__init__(**kwargs)
        self.w = nn.Parameter(torch.ones(1, num_base_models), requires_grad=True)
        self.to_device()

    def forward(self, representations):
        return self.final_activation(self.w)
