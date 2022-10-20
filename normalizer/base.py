
import torch
import torch.nn as nn

from config.config import NORM_EPS


class BaseTSNormalizer:
    NORM_EPS = NORM_EPS

    def __init__(self, local_updater, clip_input, output_rectifier, **kwargs):
        self.local_updater = local_updater
        self.clip_input = clip_input
        self.norm_params = {}
        self.skip_normalize = {k for k, v in NORM_EPS.items() if v == -1}
        if output_rectifier == 'relu':
            self.rectifier = nn.ReLU()
        elif output_rectifier == 'softplus':
            self.rectifier = nn.Softplus()
        elif output_rectifier == 'identity':
            self.rectifier = nn.Identity()
        else:
            raise NotImplementedError

    def update_norm_params(self, feat_key, val):
        raise NotImplementedError

    def normalize(self, dynamic_cont_feats, update_param=False):
        norm_params = {}
        normalized_feats = {}
        for feat_key, val in dynamic_cont_feats.items():
            if feat_key in self.skip_normalize:
                normalized_feats[feat_key] = val
                norm_params[f'{feat_key}_bias'] = None
                norm_params[f'{feat_key}_scale'] = None
            else:
                if update_param:
                    self.update_norm_params(feat_key, val)
                bias, scale = self._get_norm_params(feat_key)
                normalized_feats[feat_key] = self._normalize_value(val, bias, scale)
                norm_params[f'{feat_key}_bias'] = bias[0]
                norm_params[f'{feat_key}_scale'] = scale[0]
        return normalized_feats, norm_params

    def unnormalize(self, feat_key, val):
        if feat_key in self.skip_normalize:
            return val
        bias, scale = self._get_norm_params(feat_key)
        bias = bias.squeeze(-1)
        scale = scale.squeeze(-1)
        return self.rectifier(val * scale + bias)

    def _get_norm_params(self, feat_key):
        return self.norm_params[f'{feat_key}_bias'], self.norm_params[f'{feat_key}_scale']

    def _normalize_value(self, x, m, s):
        v = (x - m) / s
        if self.clip_input is not None:
            v = torch.clamp(v, -self.clip_input, self.clip_input)
        return v
