
import logging
import math

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


def get_device(device: str) -> 'torch.device':
    if device != 'cpu' and not torch.cuda.is_available():
        logging.warning(f'CUDA is not available. But device "{device}" is given.')
        device = 'cpu'
    return torch.device(device)


def to_np(t):
    if isinstance(t, torch.Tensor):
        return t.cpu().detach().numpy()
    else:
        return t


def get_cosine_with_hard_restarts_schedule_with_warmup(
        optimizer: Optimizer, num_warmup_steps: int, num_training_steps: int,
        num_cycles: int = 1, last_epoch: int = -1,
):
    """
    Create a schedule with a learning rate that decreases following the values of the cosine function between the
    initial lr set in the optimizer to 0, with several hard restarts, after a warmup period during which it increases
    linearly between 0 and the initial lr set in the optimizer.

    Args:
        optimizer (:class:`~torch.optim.Optimizer`):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (:obj:`int`):
            The number of steps for the warmup phase.
        num_training_steps (:obj:`int`):
            The total number of training steps.
        num_cycles (:obj:`int`, `optional`, defaults to 1):
            The number of hard restarts to use.
        last_epoch (:obj:`int`, `optional`, defaults to -1):
            The index of the last epoch when resuming training.

    Return:
        :obj:`torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    def lr_lambda_ratio(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        if progress >= 1.0:
            return 0.0
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * ((float(num_cycles) * progress) % 1.0))))

    def lr_lambda(current_step):
        ratio = lr_lambda_ratio(current_step)
        return 0.1 + 0.9 * ratio

    return LambdaLR(optimizer, lr_lambda, last_epoch)
