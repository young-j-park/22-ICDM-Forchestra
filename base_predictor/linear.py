
import logging

import torch
import torch.nn as nn

from base_predictor.base import BasePredictors


class LinearForecasting(BasePredictors):
    def __init__(
            self, dynamic_catg_feats, dynamic_cont_feats_dim, **kwargs
    ):
        super().__init__(**kwargs)
        if len(dynamic_catg_feats) > 0:
            logging.warning(f'[{self.desc}] Dynamic categorical features is not implemented.')

        self.train_dynamic_cont_feat_layers = nn.ModuleDict({
            feat: self.build_cont_layer(self.hist_len * dim, self.pred_len)
            for feat, dim in dynamic_cont_feats_dim.items()
        })
        self.static_catg_feat_layers = nn.ModuleDict({
            feat: self.build_catg_layer(cardinality, self.pred_len, padding_idx=0)
            for feat, cardinality in static_catg_feats_card.items()
        })
        self.static_cont_feat_layers = nn.ModuleDict({
            feat: self.build_cont_layer(dim, self.pred_len)
            for feat, dim in static_cont_feats_dim.items()
        })
        self.feats_cnt = len(self.train_dynamic_cont_feat_layers) \
                         + len(self.static_catg_feat_layers) \
                         + len(self.static_cont_feat_layers)
        self.feats_cnt = torch.tensor(
            self.feats_cnt, dtype=torch.float32, device=self.device,
            requires_grad=False
        )
        self.to_device()

    def build_cont_layer(self, inp_dim, out_dim):
        return nn.Linear(inp_dim, out_dim)

    def build_catg_layer(self, inp_card, out_dim, padding_idx=None):
        return nn.Embedding(inp_card, out_dim, padding_idx=padding_idx)

    def forward(
            self,
            train_dynamic_cont_feats, train_dynamic_catg_feats,
            train_date_catg_feats, train_date_cont_feats,
            static_catg_feats, static_cont_feats,
            batch_size, **kwargs
    ):
        pred = torch.zeros(batch_size, self.pred_len, device=self.device)
        for key, feat_model in self.train_dynamic_cont_feat_layers.items():
            pred += feat_model(
                train_dynamic_cont_feats[key].transpose(0, 1).view(batch_size, -1)
            )
        for key, feat_model in self.static_catg_feat_layers.items():
            pred += feat_model(static_catg_feats[key])
        for key, feat_model in self.static_cont_feat_layers.items():
            pred += feat_model(static_cont_feats[key])
        pred = (pred / self.feats_cnt).T
        return pred
