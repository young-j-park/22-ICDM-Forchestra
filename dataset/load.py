
from typing import Tuple, Dict
import os
from datetime import datetime
import json

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

from dataset.dataset import M5Dataset
from utility.date_utils import get_date_range
from args import DataArguments
from config.config import M5_DATASET_PATH, TARGET_KEY, DATE_KEY


def load_data(
        args: DataArguments,
        is_train: bool
) -> Tuple[M5Dataset, Dict[str, int], Dict[str, np.ndarray]]:
    if is_train:
        begin = datetime.fromisoformat(args.train_dataset_begin)
        end = datetime.fromisoformat(args.train_dataset_end)
        dataset_name = args.train_dataset
    else:
        begin = datetime.fromisoformat(args.test_load_begin)
        end = datetime.fromisoformat(args.test_dataset_end)
        dataset_name = args.test_dataset
    dataset_days = (end - begin).days + 1
    begin_str = begin.strftime('%Y-%m-%d')
    end_str = end.strftime('%Y-%m-%d')

    split = dataset_name.split('_')[-1]
    # FIXME
    if dataset_name.startswith('m5'):
        dataset_path = os.path.join(M5_DATASET_PATH, f'{split}_data.npz')
    elif dataset_name.startswith('e_commerce'):
        dataset_path = f'./dataset/e_commerce/{split}_data.npz'

    m5_dataset = M5Dataset(
        dataset_path,
        subset_begin=begin_str,
        subset_end=end_str,
        ts_col_name=TARGET_KEY,
    )
    dataset_meta = {
        'total': len(m5_dataset),
        'dataset_days': dataset_days,
    }
    target_date_list = np.array(get_date_range(begin_str, end_str))
    target_prod_list = m5_dataset.get_prod_list()
    data_gl_normed = gl_normalize_data(
        dataset_path, args.train_dataset_begin, args.train_dataset_end,
        begin_str, end_str
    )
    prod_dict = dict(zip(target_prod_list, range(len(target_prod_list))))
    cache_dict = {
        'data_gl_normed': data_gl_normed,
        'prod_dict': prod_dict,
        'date_list': target_date_list
    }
    return m5_dataset, dataset_meta, cache_dict


def load_meta(path: str) -> Dict[str, int]:
    meta_file_path = os.path.join(path, 'stat_cat_meta.json')
    if not os.path.exists(meta_file_path):
        raise ValueError(f'{meta_file_path} does not exist')
    with open(meta_file_path, 'rb') as fp:
        data = fp.read().split(b'\n')

    meta_data = {}
    for d in data[:-1]:
        meta_data.update(json.loads(d))
    return meta_data


def gl_normalize_data(
        dataset_path: str,
        train_begin: str,
        train_end: str,
        output_begin: str,
        output_end: str,
) -> np.ndarray:
    data = np.load(dataset_path, allow_pickle=True)
    target_data = data[TARGET_KEY].T
    target_date_list = data[DATE_KEY]

    train_begin_idx = np.argwhere(target_date_list == train_begin).item()
    train_end_idx = np.argwhere(target_date_list == train_end).item()
    output_begin_idx = np.argwhere(target_date_list == output_begin).item()
    output_end_idx = np.argwhere(target_date_list == output_end).item()

    target_train_data = target_data[output_begin_idx:output_end_idx+1]
    scaler = StandardScaler().fit(target_train_data)

    target_data_norm = scaler.transform(target_data).astype(np.float32)
    target_data_norm = torch.from_numpy(
        target_data_norm[output_begin_idx:output_end_idx+1].T
    ).unsqueeze(-1)
    return target_data_norm
