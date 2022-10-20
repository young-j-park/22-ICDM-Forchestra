
import torch
import torch.nn as nn
import logging


class BaseModel(nn.Module):
    def __init__(
            self, dropout, device, model_load_path, model_save_path, **kwargs
    ):
        super().__init__()
        self.dropout = dropout
        self.best_state_dict = None
        self.device = device
        self.model_load_path = model_load_path
        self.model_save_path = model_save_path
        self.final_activation = nn.Softmax(dim=1)
        self.repr_model = None
        self.base_models = None

    def save(self):
        if self.model_save_path is not None:
            torch.save(self.state_dict(), self.model_save_path)
            logging.info(f'Forchestra param is saved.')
        else:
            logging.info(f'Forchestra path is not specified. Skip saving.')

    def load(self):
        if self.model_load_path is not None:
            state_dict = torch.load(self.model_load_path, self.device)
            self.load_state_dict(state_dict)
            logging.info(f'Forchestra param is loaded.')
        else:
            logging.info(f'Forchestra path is not specified. Skip loading.')

    def save_best(self):
        self.best_state_dict = self.state_dict()

    def load_best(self):
        self.load_state_dict(self.best_state_dict)

    def to_device(self):
        self.to(self.device)

    def forward(
            self, representations
    ):
        """
        N: batch_size
        E: repr_dim
        K: num_base_models

        :param representations: Tensor(N, E)
        :return: Tensor(N, K)
        """
        raise NotImplementedError
