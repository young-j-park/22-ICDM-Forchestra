
import torch
import torch.nn as nn
import logging


class BasePredictors(nn.Module):
    def __init__(
            self, hist_len, pred_len, dropout, device,
            model_name, model_load_path, **kwargs
    ):
        super().__init__()
        self.hist_len = hist_len
        self.pred_len = pred_len
        self.dropout = dropout
        self.best_state_dict = None
        self.device = device
        self.desc = model_name
        self.model_load_path = model_load_path

    def load(self):
        if self.model_load_path is not None:
            state_dict = torch.load(self.model_load_path, self.device)
            self.load_state_dict(state_dict)
            logging.info(f'BP param is loaded from: {self.model_load_path}.')
        else:
            logging.info(f'BP path is not specified. Skip loading.')

    def save_best(self):
        self.best_state_dict = self.state_dict()

    def load_best(self):
        self.load_state_dict(self.best_state_dict)

    def to_device(self):
        self.to(self.device)

    def forward(
            self, train_dynamic_cont_feats, norm_params, batch_size, **kwargs
    ):
        """
        :param train_dynamic_cont_feats: Dict[att_key, Tensor(T_H, N, D_i)]
        :param norm_params: Dict[norm_key, Tensor(N)]
        :param batch_size: int
        :return: Tensor(T_P, N)
        """
        raise NotImplementedError

    @staticmethod
    def get_activation(method):
        method = method.lower()
        if method == 'relu':
            return nn.ReLU()
        elif method == 'tanh':
            return nn.Tanh()
        elif method == 'leaky':
            return nn.LeakyReLU()
        else:
            raise NotImplementedError
