import importlib
import logging
import os


import numpy as np
import torch
from torch.utils.data import Dataset


from args import args


def create_dir(dir_path):
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path)


class ForecastingData:
    def __init__(self):
        raw_X = self.load_data()
        self.raw_X = raw_X
        assert raw_X.ndim == 2
        self.valid_idx = int((1-args.valid_ratio-args.test_ratio)*raw_X.shape[0])
        self.test_idx = int((1-args.test_ratio)*raw_X.shape[0])
        train_set = raw_X[:self.valid_idx,:]

        if args.local_norm:
            axis = 0
        else:
            axis = None

        if args.norm_type == 'none':
            self.sc = np.ones((1, 1))
            self.mn = np.zeros((1, 1))
        elif args.norm_type == 'minmax':
            self.mn = train_set.min(axis=axis, keepdims=True)
            self.sc = train_set.max(axis=axis, keepdims=True) - self.mn
        elif args.norm_type == 'standard':
            self.mn = train_set.mean(axis=axis, keepdims=True)
            self.sc = train_set.std(axis=axis, keepdims=True)

    def load_data(self):
        data_path = os.path.join(args.data_dir, f'{args.dataset}.npy')
        raw_X = np.load(data_path, allow_pickle=True)
        args.n_series = raw_X[0].shape[-1]
        return raw_X

    def get_dataset(self, i):
        if i == 0:
            start = 0
            end = self.valid_idx
        elif i == 1:
            start = self.valid_idx
            end = self.test_idx
        else:
            start = self.test_idx
            end = self.raw_X.shape[0]
        return ForecastingDataset(self.raw_X, start, end, self.sc, self.mn)


class ForecastingDataset(Dataset):
    def __init__(self, raw_X, start, end, sc, mn):
        self.sc = sc
        self.mn = mn
        self.start = start
        self.end = end
        assert self.start < self.end

        # start = 0 indicates training set and consider the shift of series_lens
        if self.start == 0:
            assert args.series_len < self.end
            self.rse = np.sum((raw_X[args.series_len:self.end] - np.mean(raw_X[args.series_len:self.end]))**2)
        else:
            self.rse = np.sum((raw_X[self.start:self.end] - np.mean(raw_X[self.start:self.end])) ** 2)

        self.X = self.norm(raw_X).astype(np.float32)

    def norm(self, X):
        return (X - self.mn) / self.sc

    def renorm(self, X):
        return X * self.sc + self.mn

    def __len__(self):
        if self.start == 0:
            return self.end + 1 - args.series_len - args.horizon
        else:
            return np.ceil((self.end - self.start) / (args.horizon)).astype(int)

    def __getitem__(self, idx):
        if self.start == 0:
            return (self.X[idx:idx+args.series_len], self.X[idx+args.series_len:idx+args.series_len+args.horizon])
        else:
            return (self.X[(self.start + idx * args.horizon - args.series_len):(self.start + idx * args.horizon)],
                    self.X[self.start + idx*args.horizon : min(self.end, self.start + (idx+1)*args.horizon)])


