
import torch.nn as nn

from base_predictor.base_autoregressive import BaseAutoregressive


class LSTMForecasting(BaseAutoregressive):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.pred_layer = nn.Sequential(
            nn.Dropout(p=self.dropout),
            self.get_activation(self.activation),
            nn.Linear(self.emb_dim, self.pred_len)
        )
        self.to_device()

    def _build_autoregressive_layer(self):
        d = 2 if self.bidirectional else 1
        assert self.emb_dim % d == 0
        self.lstm = nn.LSTM(
            input_size=self.emb_dim,
            hidden_size=self.emb_dim//d,
            num_layers=self.num_layers,
            dropout=self.dropout,
            bidirectional=self.bidirectional,
        )

    def _predict(self, train_embs):
        pred = self.pred_layer(self.lstm(train_embs)[0][-1])
        return pred.T
