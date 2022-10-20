
import torch

from normalizer import BaseTSNormalizer


class StandardTSNormalizer(BaseTSNormalizer):
    def update_norm_params(self, feat_key, val):
        val_mean = torch.mean(val, 0, keepdim=True)
        val_std = torch.clamp(
            torch.std(val, 0, keepdim=True),
            self.NORM_EPS.get(feat_key, 0),
            self.NORM_EPS.get(feat_key, 0)*100
        )
        self.norm_params.update({
            f'{feat_key}_bias': val_mean,
            f'{feat_key}_scale': val_std,
        })
