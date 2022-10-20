
import logging

import torch
from torch import nn
from torch.optim.swa_utils import AveragedModel

from repr_model.encoder import TSEncoder


class TS2VecModel(nn.Module):
    def __init__(
            self,
            repr_model_depth,
            repr_model_input_dim,
            repr_model_projection_dim,
            repr_model_hidden_dim,
            repr_model_output_dim,
            repr_model_kernel_size,
            repr_model_mask_mode,
            repr_model_encoding_window,
            device,
            repr_model_path,
            eps=1e-5
    ):
        super().__init__()
        self.device = device
        self.repr_model_path = repr_model_path
        self.encoding_window = repr_model_encoding_window

        self._net = TSEncoder(
            input_dim=repr_model_input_dim,
            output_dim=repr_model_output_dim,
            projection_dim=repr_model_projection_dim,
            hidden_dim=repr_model_hidden_dim,
            mask_mode=repr_model_mask_mode,
            depth=repr_model_depth,
            kernel_size=repr_model_kernel_size,
            eps=eps,
        ).to(self.device)
        self.net = AveragedModel(self._net)
        self.net.update_parameters(self._net)

    def forward(self, x, mask=None):
        out = self.net(x, mask)
        out = out[:, -1]
        return out

    def load(self):
        if self.repr_model_path is not None:
            state_dict = torch.load(self.repr_model_path, map_location=self.device)
            self.net.load_state_dict(state_dict)
            logging.info(f'RM param is loaded from: {self.repr_model_path}.')
        else:
            logging.info(f'RM path is not specified. Skip loading.')
