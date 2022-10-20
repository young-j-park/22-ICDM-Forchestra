
import numpy as np
import torch
from torch.utils.data import Dataset


class M5Dataset(Dataset):
    def __init__(
            self,
            dataset_path: str,
            subset_begin: str,
            subset_end: str,
            emb_path: str = None,
            ts_col_name: str = None,
    ):
        super().__init__()
        data = np.load(dataset_path, allow_pickle=True)
        self.ts_col_name = 'order_cnt' if ts_col_name is None else ts_col_name
        date_list = data['date_list']
        idx_begin = np.where(date_list == subset_begin)[0][0]
        idx_end = np.where(date_list == subset_end)[0][0]

        self.order_cnt = torch.tensor(
            data['order_cnt'][:, idx_begin:idx_end+1], dtype=torch.float32
        )
        self.mask = torch.tensor(
            data['mask'][:, idx_begin:idx_end+1], dtype=torch.bool
        )
        self.prod_no = data['prod_list']
        self.date_list = data['date_list'][idx_begin:idx_end+1]

        if emb_path is not None:
            raise NotImplementedError

    def __len__(self):
        return len(self.prod_no)

    def __getitem__(self, idx):
        return {
            self.ts_col_name: self.order_cnt[idx],
            'prod_no': self.prod_no[idx],
            'mask': self.mask[idx],
        }

    def get_date_list(self):
        return self.date_list

    def get_prod_list(self):
        return self.prod_no
