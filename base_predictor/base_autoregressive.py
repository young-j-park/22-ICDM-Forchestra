
import torch.nn as nn

from base_predictor.base import BasePredictors
from config.config import TARGET_KEY


class BaseAutoregressive(BasePredictors):
    def __init__(
            self, n_head, num_layers, activation='relu', emb_dim=64,
            bidirectional=False, **kwargs
    ):
        super().__init__(**kwargs)
        self.emb_dim = emb_dim
        self.activation = activation
        self.n_head = n_head
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        # initial emb layers
        self.order_layer = nn.Linear(1, self.emb_dim)

        # autoregressive layer
        self._build_autoregressive_layer()

        # prediction layer
        self.pred_layer = nn.Linear(emb_dim, 1)
        self.to_device()

    def _build_autoregressive_layer(self):
        raise NotImplementedError

    def _predict(self, train_embs):
        raise NotImplementedError

    def forward(
            self, train_dynamic_catg_feats, train_dynamic_cont_feats,
            batch_size, **kwargs
    ):
        train_init_embs = self.order_layer(train_dynamic_cont_feats[TARGET_KEY])
        return self._predict(train_init_embs)

