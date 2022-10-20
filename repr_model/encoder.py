
import torch
from torch import nn

from repr_model.dilated_conv import DilatedConvEncoder
from utility.ts2vec_utils import generate_binomial_mask, generate_continuous_mask


class TSEncoder(nn.Module):
    def __init__(
            self,
            input_dim,
            projection_dim,
            hidden_dim,
            output_dim,
            depth,
            kernel_size,
            mask_mode='binomial',
            eps=1e-5,
    ):
        super().__init__()
        self.input_dims = input_dim
        self.output_dims = output_dim
        self.projection_dims = projection_dim
        self.hidden_dims = hidden_dim
        self.mask_mode = mask_mode
        self.input_fc = nn.Linear(input_dim, hidden_dim)
        self.output_fc = nn.Linear(hidden_dim, output_dim)
        self.feature_extractor = DilatedConvEncoder(
            in_channels=projection_dim,
            channels=[hidden_dim] * depth,
            kernel_size=kernel_size
        )
        self.eps = eps
        self.repr_dropout = nn.Dropout(p=0.1)

    def forward(self, x, mask=None):  # x: B x T x input_dims
        nan_mask = ~x.isnan().any(axis=-1)
        x[~nan_mask] = 0
        x = self.input_fc(x)  # B x T x Ch
        # generate & apply mask
        if mask is None:
            if self.training:
                mask = self.mask_mode
            else:
                mask = 'all_true'
        if mask == 'binomial':
            mask = generate_binomial_mask(x.size(0), x.size(1)).to(x.device)
        elif mask == 'continuous':
            mask = generate_continuous_mask(x.size(0), x.size(1)).to(x.device)
        elif mask == 'all_true':
            mask = x.new_full((x.size(0), x.size(1)), True, dtype=torch.bool)
        elif mask == 'all_false':
            mask = x.new_full((x.size(0), x.size(1)), False, dtype=torch.bool)
        elif mask == 'mask_last':
            mask = x.new_full((x.size(0), x.size(1)), True, dtype=torch.bool)
            mask[:, -1] = False
        mask &= nan_mask
        x[~mask] = 0
        
        # encoder
        x = self.repr_dropout(self.feature_extractor(x))  # B x T x C
        x = self.output_fc(x)
        return x
