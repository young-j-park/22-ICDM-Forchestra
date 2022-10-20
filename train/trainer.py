
from typing import Dict
import logging
from pprint import pprint

import numpy as np
import torch
from torch.utils.data import DataLoader as TorchDataLoader

from args import TrainingArguments, DataArguments
from dataset.dataset import M5Dataset
from normalizer.base import BaseTSNormalizer
from model.base import BaseModel
from utility.torch_utils import get_cosine_with_hard_restarts_schedule_with_warmup, to_np
from config.config import TARGET_KEY, MASK_KEY, NUM_WORKERS

TEMPORAL_AXIS = 0
ENTITY_AXIS = 1
BEST_EPOCH = 9999


class Trainer:
    def __init__(
            self,
            args: TrainingArguments,
            data_args: DataArguments,
            normalizer: BaseTSNormalizer,
            model: BaseModel,
            train_ds: M5Dataset,
            train_cache_data: Dict[str, np.ndarray],
            test_ds: M5Dataset,
            test_cache_data: Dict[str, np.ndarray]
    ):
        self.args = args
        self.data_args = data_args
        self.normalizer = normalizer
        self.model = model
        self.train_ds = train_ds
        self.test_ds = test_ds
        self.train_cache_data = train_cache_data
        self.test_cache_data = test_cache_data
        self.device = self.args.device

        self.hist_len = args.base_model_hist_len
        self.pred_len = args.pred_len
        self.repr_slide_padding = self.data_args.repr_slide_padding
        self.max_hist_len = self.data_args.max_hist_len

        self.train_begin_idx = self.max_hist_len
        self.train_end_idx = args.train_dataset_len - self.pred_len - args.validation_days
        self.train_t_idx = np.arange(self.train_begin_idx, self.train_end_idx + 1)

        self.valid_begin_idx = args.train_dataset_len - args.validation_days
        self.valid_end_idx = self.valid_begin_idx + args.validation_days - self.pred_len
        self.valid_t_idx = np.arange(self.valid_begin_idx, self.valid_end_idx + 1, self.pred_len)

        self.test_begin_idx = self.max_hist_len
        self.test_end_idx = args.test_dataset_len - self.pred_len
        self.test_t_idx = np.arange(self.test_begin_idx, self.test_end_idx + 1, self.pred_len)

        assert self.train_end_idx > self.train_begin_idx
        assert self.valid_end_idx >= self.valid_begin_idx
        assert self.test_end_idx >= self.test_begin_idx

        self.best_valid = np.inf
        self.best_valid_epoch = -1

        self.optim = torch.optim.Adam(
                model.parameters(), lr=self.args.init_lr,
                weight_decay=self.args.weight_decay
        )
        self.scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(
                    self.optim, self.args.num_warmup_steps, self.args.num_training_steps,
                    self.args.num_cycles
        )
        self.best_state_dict = None

    def train(self):
        self.model.save_best()
        for i_epoch in range(1, self.args.max_epoch + 1):
            loss = self.run_epoch()
            logging.info(f'Epoch: {i_epoch}/{self.args.max_epoch}, Train loss: {loss:.4f}')
            self.evaluate(i_epoch)
            self.evaluate(i_epoch, test=True)
            if i_epoch >= max(
                self.args.tol_epoch, self.best_valid_epoch + self.args.early_stop
            ):
                logging.info('Early Stopped.')
                break

        self.model.load_best()
        logging.info('Start evaluation.')

    def run_epoch(self):
        losses = []
        dataloader = TorchDataLoader(
            self.train_ds,
            batch_size=int(self.args.batch_size),
            num_workers=NUM_WORKERS,
            shuffle=True,
        )
        for minibatch in dataloader:
            t_samples = self.sample_t_idx(bs=len(minibatch[TARGET_KEY]))
            loss, _ = self.run_batch(
                self.model, self.optim, self.scheduler,
                t_samples, minibatch, self.train_cache_data
            )
            losses.append(loss)
        return np.mean(losses)

    def sample_t_idx(self, bs):
        t_idx = np.random.choice(self.train_t_idx, size=bs)
        return t_idx

    def make_batch_data(self, t_samples, minibatch, cache_data):
        bs = len(minibatch[TARGET_KEY])
        feat_types = [
            'dynamic_cont_feats',
            'dynamic_catg_feats',
        ]
        batch_data = {'batch_size': bs}
        for feat_type in feat_types:
            batch_data[f'train_{feat_type}'] = {}
            batch_data[f'eval_{feat_type}'] = {}

            # slice
            for feat_key in self.data_args.__getattribute__(feat_type):
                feat_val = minibatch[feat_key]
                if 'cont' in feat_type:
                    feat_val = feat_val.unsqueeze(2).float()
                else:
                    feat_val = feat_val.long()

                for prefix, ti, tf in [
                    ('train_', t_samples-self.hist_len, t_samples),
                    ('eval_', t_samples, t_samples+self.pred_len)
                ]:
                    feat_val_slice = torch.stack([
                        feat_val[i, t0:t1]
                        for i, (t0, t1) in enumerate(zip(ti, tf))
                    ], dim=ENTITY_AXIS)
                    batch_data[f'{prefix}{feat_type}'][feat_key] = feat_val_slice.to(self.device)
        # mask
        batch_data['train_exposure_mask'] = batch_data['train_dynamic_catg_feats'][MASK_KEY].bool()
        batch_data['eval_exposure_mask'] = batch_data['eval_dynamic_catg_feats'][MASK_KEY].bool()

        # post-process
        target_order_hist = batch_data['eval_dynamic_cont_feats'].get(TARGET_KEY).squeeze(-1)

        # normalize
        batch_data['train_dynamic_cont_feats'], norm_params = self.normalizer.normalize(
            batch_data['train_dynamic_cont_feats'], update_param=True
        )
        batch_data.update(norm_params)
        batch_data.pop('eval_dynamic_cont_feats')
        batch_data.pop('eval_dynamic_catg_feats')

        # make history data for the NC
        batch_prod_list = to_np(minibatch['prod_no'])
        prod_dict = cache_data['prod_dict']
        prod_idx = [prod_dict[p] for p in batch_prod_list]

        # make batch for normalized repr history
        data_gl_normed = cache_data['data_gl_normed'][prod_idx]
        ti, tf = t_samples - self.repr_slide_padding, t_samples
        data_gl_normed = torch.stack([
            data_gl_normed[i, t0:t1]
            for i, (t0, t1) in enumerate(zip(ti, tf))
        ], dim=0)
        data_gl_normed = torch.as_tensor(data_gl_normed, device=self.device)
        return batch_data, target_order_hist, data_gl_normed

    def run_batch(self, model, optim, scheduler, t_samples, minibatch, cache_data, train=True):
        if train:
            model.train()
        else:
            model.eval()

        # make batch
        batch_data, y_true, repr_hist_norm = self.make_batch_data(
            t_samples, minibatch, cache_data
        )

        # forward
        predictions = [
            self.normalizer.unnormalize(
                TARGET_KEY, bm(**batch_data)
            )
            for bm in self.model.base_models
        ]
        predictions = torch.stack(predictions, -1)
        representations = self.model.repr_model(repr_hist_norm)
        ensemble_weight = model(representations)
        y_pred = torch.sum(ensemble_weight.unsqueeze(0) * predictions, dim=2)
        pred_mask = batch_data.get('eval_exposure_mask')

        # backward
        if train:
            loss = self.compute_loss(y_true, y_pred, pred_mask)
            if loss is None:
                loss = 0.0
            else:
                optim.zero_grad()
                loss.backward()
                optim.step()
                scheduler.step()
                loss = loss.item()
        else:
            loss = self.compute_loss(y_true, y_pred, pred_mask)
            loss = 0.0 if loss is None else loss.detach().cpu().item()
            y_pred = to_np(y_pred * pred_mask)
            y_true = to_np(y_true)
        return loss, (y_true, y_pred)

    def compute_loss(self, ytrue, ypred, mask):
        if torch.any(mask).item() == 0:
            return None
        err = ypred - ytrue
        loss = torch.abs(err)
        loss = loss[mask]
        return torch.mean(loss)

    def evaluate(self, i_epoch=0, test=False, save_output=False):
        if test:
            t_idx = self.test_t_idx
            dl_name = 'test'
            ds = self.test_ds
            cache_data = self.test_cache_data
        else:
            t_idx = self.valid_t_idx
            dl_name = 'valid'
            ds = self.train_ds
            cache_data = self.train_cache_data
        results = {
            'loss_all': [],
            'y_pred_all': [],
            'y_gt_all': [],
        }
        eval_prod_list_all = []

        dataloader = TorchDataLoader(
            dataset=ds,
            batch_size=int(self.args.batch_size),
            num_workers=NUM_WORKERS,
            shuffle=False,
        )

        for minibatch in dataloader:
            eval_prod_list_all.append(to_np(minibatch['prod_no']))
            tmp = {
                'y_pred': [],
                'y_gt': [],
            }
            for t_sample in t_idx:
                loss, (y_gt, y_pred) = self.run_batch(
                    self.model, self.optim, self.scheduler,
                    np.array([t_sample]*len(minibatch['prod_no'])),
                    minibatch, cache_data, train=False
                )
                results['loss_all'].append(loss)
                tmp['y_gt'].append(y_gt)
                tmp['y_pred'].append(y_pred)
            results['y_gt_all'].append(
                np.concatenate(tmp['y_gt'], TEMPORAL_AXIS)
            )
            results['y_pred_all'].append(
                np.concatenate(tmp['y_pred'], TEMPORAL_AXIS)
            )
        eval_prod_list_all = np.concatenate(eval_prod_list_all)
        y_gt_all = np.concatenate(results['y_gt_all'], ENTITY_AXIS)
        y_pred_all = np.concatenate(results['y_pred_all'], ENTITY_AXIS)
        if save_output:
            eval_len = len(t_idx) * self.pred_len
            assert y_pred_all.shape == y_gt_all.shape \
                   and y_gt_all.shape[ENTITY_AXIS] == len(eval_prod_list_all) \
                   and y_gt_all.shape[TEMPORAL_AXIS] == eval_len
            eval_predictions = {
                'y_pred': y_pred_all,
                'date_list': cache_data['date_list'][t_idx[0]:t_idx[0]+eval_len],
                'prod_list': eval_prod_list_all,
            }
            eval_file_path = self.args.output_fname_prefix \
                             + '_{ds}.npz'.format(ds='downstream' if test else 'original')
            np.savez_compressed(
                file=eval_file_path,
                **eval_predictions
            )
            logging.info(f"Inference result saved at {eval_file_path}")
        else:
            cur_loss = np.mean(results['loss_all'])
            eval_dict = {'loss': cur_loss}
            logging.info(f"{'Test' if test else 'Valid'} loss: {cur_loss}")
            if i_epoch != BEST_EPOCH:
                logging.info({'run_type': f'{dl_name}', **eval_dict})
                if not test and cur_loss <= self.best_valid:
                    self.best_valid = cur_loss
                    self.best_valid_epoch = i_epoch
                    self.model.save_best()
                    self.model.save()
        return eval_prod_list_all, y_gt_all, y_pred_all
