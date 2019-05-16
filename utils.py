# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
import torch
import logging
from functools import partial
from torch.nn import functional as F
from torch.distributions.utils import lazy_property
from torch.distributions import utils as distr_utils
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.distributions.categorical import Categorical as TorchCategorical


class AverageMeter:
    def __init__(self):
        self.value = None
        self.avg = None
        self.sum = None
        self.count = None
        self.reset()

    def reset(self):
        self.value = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, value, n=1):
        self.value = value
        self.sum += value * n
        self.count += n
        self.avg = self.sum / self.count


def get_logger(file_name):
    logger = logging.getLogger("general_logger")
    handler = logging.FileHandler(file_name, mode='w')
    formatter = logging.Formatter("%(asctime)s - %(message)s", "%d-%m-%Y %H:%M:%S")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger


def get_lr_scheduler(logger, optimizer, mode='max', factor=0.5, patience=10, threshold=1e-4, threshold_mode='rel'):
    def reduce_lr(self, epoch):
        ReduceLROnPlateau._reduce_lr(self, epoch)
        logger.info(f"learning rate is reduced by factor {factor}!")

    lr_scheduler = ReduceLROnPlateau(optimizer, mode, factor, patience, False, threshold, threshold_mode)
    lr_scheduler._reduce_lr = partial(reduce_lr, lr_scheduler)
    return lr_scheduler


def clamp_grad(v, min_val, max_val):
    if v.requires_grad:
        v_tmp = v.expand_as(v)
        v_tmp.register_hook(lambda g: g.clamp(min_val, max_val))
        return v_tmp
    return v


def length_to_mask(length):
    with torch.no_grad():
        batch_size = length.shape[0]
        max_length = length.data.max()
        range = torch.arange(max_length, dtype=torch.int64, device=length.device)
        range_expanded = range[None, :].expand(batch_size, max_length)
        length_expanded = length[:, None].expand_as(range_expanded)
        return (range_expanded < length_expanded).float()


class Categorical:
    def __init__(self, scores, mask=None):
        self.mask = mask
        if mask is None:
            self.cat_distr = TorchCategorical(F.softmax(scores, dim=-1))
            self.n = scores.shape[0]
            self.log_n = math.log(self.n)
        else:
            self.n = self.mask.sum(dim=-1)
            self.log_n = (self.n + 1e-17).log()
            self.cat_distr = TorchCategorical(Categorical.masked_softmax(scores, self.mask))

    @lazy_property
    def probs(self):
        return self.cat_distr.probs

    @lazy_property
    def logits(self):
        return self.cat_distr.logits

    @lazy_property
    def entropy(self):
        if self.mask is None:
            return self.cat_distr.entropy() * (self.n != 1)
        else:
            entropy = - torch.sum(self.cat_distr.logits * self.cat_distr.probs * self.mask, dim=-1)
            does_not_have_one_category = (self.n != 1.0).to(dtype=torch.float32)
            # to make sure that the entropy is precisely zero when there is only one category
            return entropy * does_not_have_one_category

    @lazy_property
    def normalized_entropy(self):
        return self.entropy / (self.log_n + 1e-17)

    def sample(self):
        return self.cat_distr.sample()

    def rsample(self, temperature=None, gumbel_noise=None):
        if gumbel_noise is None:
            with torch.no_grad():
                uniforms = torch.empty_like(self.probs).uniform_()
                uniforms = distr_utils.clamp_probs(uniforms)
                gumbel_noise = -(-uniforms.log()).log()
            # TODO(serhii): This is used for debugging (to get the same samples) and is not differentiable.
            # gumbel_noise = None
            # _sample = self.cat_distr.sample()
            # sample = torch.zeros_like(self.probs)
            # sample.scatter_(-1, _sample[:, None], 1.0)
            # return sample, gumbel_noise

        elif gumbel_noise.shape != self.probs.shape:
            raise ValueError

        if temperature is None:
            with torch.no_grad():
                scores = (self.logits + gumbel_noise)
                scores = Categorical.masked_softmax(scores, self.mask)
                sample = torch.zeros_like(scores)
                sample.scatter_(-1, scores.argmax(dim=-1, keepdim=True), 1.0)
                return sample, gumbel_noise
        else:
            scores = (self.logits + gumbel_noise) / temperature
            sample = Categorical.masked_softmax(scores, self.mask)
            return sample, gumbel_noise

    def log_prob(self, value):
        if value.dtype == torch.long:
            if self.mask is None:
                return self.cat_distr.log_prob(value)
            else:
                return self.cat_distr.log_prob(value) * (self.n != 0.).to(dtype=torch.float32)
        else:
            max_values, mv_idxs = value.max(dim=-1)
            relaxed = (max_values - torch.ones_like(max_values)).sum().item() != 0.0
            if relaxed:
                raise ValueError("The log_prob can't be calculated for the relaxed sample!")
            return self.cat_distr.log_prob(mv_idxs) * (self.n != 0.).to(dtype=torch.float32)

    @staticmethod
    def masked_softmax(logits, mask):
        """
        This method will return valid probability distribution for the particular instance if its corresponding row
        in the `mask` matrix is not a zero vector. Otherwise, a uniform distribution will be returned.
        This is just a technical workaround that allows `Categorical` class usage.
        If probs doesn't sum to one there will be an exception during sampling.
        """
        probs = F.softmax(logits, dim=-1) * mask
        probs = probs + (mask.sum(dim=-1, keepdim=True) == 0.).to(dtype=torch.float32)
        Z = probs.sum(dim=-1, keepdim=True)
        return probs / Z


class EarlyStopping:
    def __init__(self, mode='max', patience=20, threshold=1e-4, threshold_mode='rel'):
        self.mode = mode
        self.patience = patience
        self.threshold = threshold
        self.threshold_mode = threshold_mode

        self.num_bad_epochs = 0
        self.mode_worse = None  # the worse value for the chosen mode
        self.is_better = None
        self.last_epoch = -1
        self.is_converged = False
        self._init_is_better(mode=mode, threshold=threshold, threshold_mode=threshold_mode)
        self.best = self.mode_worse

    def is_improved(self):
        return self.num_bad_epochs == 0

    def step(self, metrics):
        if self.is_converged:
            raise ValueError
        current = metrics
        self.last_epoch += 1

        if self.is_better(current, self.best):
            self.best = current
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1

        if self.num_bad_epochs > self.patience:
            self.is_converged = True

    def _cmp(self, mode, threshold_mode, threshold, a, best):
        if mode == 'min' and threshold_mode == 'rel':
            rel_epsilon = 1. - threshold
            return a < best * rel_epsilon
        elif mode == 'min' and threshold_mode == 'abs':
            return a < best - threshold
        elif mode == 'max' and threshold_mode == 'rel':
            rel_epsilon = threshold + 1.
            return a > best * rel_epsilon
        else:  # mode == 'max' and epsilon_mode == 'abs':
            return a > best + threshold

    def _init_is_better(self, mode, threshold, threshold_mode):
        if mode not in {'min', 'max'}:
            raise ValueError('mode ' + mode + ' is unknown!')
        if threshold_mode not in {'rel', 'abs'}:
            raise ValueError('threshold mode ' + threshold_mode + ' is unknown!')

        if mode == 'min':
            self.mode_worse = float('inf')
        else:  # mode == 'max':
            self.mode_worse = (-float('inf'))

        self.is_better = partial(self._cmp, mode, threshold_mode, threshold)
