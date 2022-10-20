
import argparse
import logging
import dataclasses
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn

from args import DataArguments, BaseModelArguments, ReprModelArguments, ModelArguments, TrainingArguments
from dataset.load import load_data
from model import build_model
from base_predictor import build_base_model
from repr_model.ts2vec_model import TS2VecModel
from normalizer import build_normalizer
from train.trainer import Trainer, BEST_EPOCH


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    # General
    parser.add_argument('--device', type=str, default='cuda')

    # Path
    parser.add_argument('--output_fname_prefix', type=str, required=False)
    parser.add_argument('--model_load_path', type=str, required=False)
    parser.add_argument('--model_save_path', type=str, required=False)

    # Data
    parser.add_argument(
        '--train_dataset', type=str, default='m5_train',
        choices=['m5_train', 'm5_test', 'e_commerce_train', 'e_commerce_test']
    )
    parser.add_argument('--train_dataset_begin', type=str, default='2011-01-29')
    parser.add_argument('--train_dataset_end', type=str, default='2016-04-24')
    parser.add_argument(
        '--test_dataset', type=str, default='m5_train',
        choices=['m5_train', 'm5_test', 'e_commerce_train', 'e_commerce_test']
    )
    parser.add_argument('--test_dataset_begin', type=str, default='2016-04-25')
    parser.add_argument('--test_dataset_end', type=str, default='2016-05-22')
    parser.add_argument('--repr_slide_padding', type=int, default=200)

    # Model
    # Base Predictors
    parser.add_argument('--base_model_path', type=str, required=False)
    parser.add_argument('--num_base_models', type=int, default=2)
    parser.add_argument(
        '--base_model_name', type=str, default='lstm', choices=['lstm', 'bilstm']
    )
    parser.add_argument('--base_model_num_layers', type=int, default=4)
    parser.add_argument('--base_model_emb_dim', type=int, default=512)
    parser.add_argument('--base_model_n_head', type=int, default=1)
    parser.add_argument('--base_model_hist_len', type=int, default=84)
    parser.add_argument('--freeze_base_model', action='store_true')
    parser.add_argument('--random_init_base_model', action='store_true')

    # Representation Module
    parser.add_argument('--repr_model_path', type=str, required=False)
    parser.add_argument('--repr_model_depth', type=int, default=5)
    parser.add_argument('--repr_model_input_dim', type=int, default=1)
    parser.add_argument('--repr_model_projection_dim', type=int, default=64)
    parser.add_argument('--repr_model_hidden_dim', type=int, default=64)
    parser.add_argument('--repr_model_output_dim', type=int, default=32)
    parser.add_argument('--repr_model_kernel_size', type=int, default=3)
    parser.add_argument('--repr_model_mask_mode', type=str, default='binomial')
    parser.add_argument('--repr_model_encoding_window', type=str, default=5)
    parser.add_argument('--freeze_repr_model', action='store_true')
    parser.add_argument('--random_init_repr_model', action='store_true')

    # Meta Learner
    parser.add_argument(
        '--meta_model_name', type=str, default='linear',
        choices=['linear', 'static']
    )
    parser.add_argument('--num_layers', type=int, default=1)
    parser.add_argument('--emb_dim', type=int, default=64)
    parser.add_argument('--pred_len', type=int, default=28)
    parser.add_argument('--activation', type=str, default='relu')
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument(
        '--output_rectifier', type=str, default='softplus',
        choices=['identity', 'relu', 'softplus'],
        help='rectify output to avoid a negative prediction.'
    )

    # Train
    parser.add_argument('--skip_train', action='store_true')
    parser.add_argument('--report_interval_steps', type=int, default=1)
    parser.add_argument('--num_warmup_steps', type=int, default=1000)
    parser.add_argument('--num_training_steps', type=int, default=1000 * 20)
    parser.add_argument('--max_epoch', type=int, default=100)
    parser.add_argument('--num_cycles', type=int, default=5)
    parser.add_argument('--early_stop', type=int, default=5)
    parser.add_argument('--tol_epoch', type=int, default=50)
    parser.add_argument('--init_lr', type=float, default=1e-4)
    parser.add_argument('--lr_gamma', type=float, default=0.95)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--validation_days', type=int, default=28)

    args = parser.parse_args()

    # device
    if args.device != 'cpu' and not torch.cuda.is_available():
        logging.warning('CUDA is not available. Set device as cpu.')
        args.device = 'cpu'

    logging.info(f'output_fname_prefix: {args.output_fname_prefix}')
    return args


def parse_args_into_dataclasses(args: argparse.Namespace) \
        -> Tuple[DataArguments, BaseModelArguments, ReprModelArguments, ModelArguments, TrainingArguments]:
    outputs = []
    dtypes = [DataArguments, BaseModelArguments, ReprModelArguments, ModelArguments, TrainingArguments]
    for dtype in dtypes:
        keys = {f.name for f in dataclasses.fields(dtype) if f.init}
        inputs = {k: v for k, v in vars(args).items() if k in keys}
        obj = dtype(**inputs)
        outputs.append(obj)
    return (*outputs,)


def main():
    # set seed
    torch.manual_seed(1234)
    np.random.seed(1234)

    # Load args
    args = parse_args()
    data_args, default_base_model_args, repr_model_args, \
        model_args, training_args = parse_args_into_dataclasses(args)
    logging.info(args)

    # Load data
    logging.info('Load data...')
    train_ds, train_dataset_meta, train_cache_data = load_data(data_args, is_train=True)
    test_ds, test_dataset_meta, test_cache_data = load_data(data_args, is_train=False)

    training_args.update_args(train_dataset_meta, test_dataset_meta)

    # Build model
    base_models = nn.ModuleList([])
    for i in range(args.num_base_models):
        bm = build_base_model(default_base_model_args)
        if not args.random_init_base_model:
            bm.load()
        if args.freeze_base_model:
            for param in bm.parameters():
                param.requires_grad = False
        base_models.append(bm)
        bm_size = sum(p.numel() for p in bm.parameters() if p.requires_grad)
        logging.info(f'BP-{i} size: {bm_size:,}')

    repr_model = TS2VecModel(**vars(repr_model_args))
    if not args.random_init_repr_model:
        repr_model.load()
    if args.freeze_repr_model:
        for param in repr_model.parameters():
            param.requires_grad = False
    repr_model_size = sum(p.numel() for p in repr_model.parameters() if p.requires_grad)
    logging.info(f'RM size: {repr_model_size:,}')

    model = build_model(model_args)
    model.repr_model = repr_model
    model.base_models = base_models
    model.load()
    model_size = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f'Forchestra model size: {model_size:,}')

    normalizer = build_normalizer('local_standard', args.output_rectifier)

    # Train and Evaluation
    trainer = Trainer(
        training_args, data_args, normalizer, model,
        train_ds, train_cache_data, test_ds, test_cache_data
    )

    if not args.skip_train:
        logging.info(f'Train the model...')
        trainer.train()
        model.save()
    else:
        logging.info(f'Skip train...')

    # trainer.evaluate(i_epoch=BEST_EPOCH, test=False, save_output=True)  # validation period
    trainer.evaluate(i_epoch=BEST_EPOCH, test=True, save_output=True)  # test period
    logging.info('Finished.')


if __name__ == '__main__':
    main()
