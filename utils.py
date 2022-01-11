import importlib
import logging
import os


import numpy as np
import torch
from torch.utils.data import Dataset


def create_dir(dir_path):
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path)


class ForecastingData:
    def __init__(self, raw_X, args):
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

    def get_dataset(self, i, series_len, horizon, ft_num=0):
        if i == 0:
            start = 0
            end = self.valid_idx
        elif i == 1:
            start = self.valid_idx
            end = self.test_idx
        else:
            start = self.test_idx
            end = self.raw_X.shape[0]
        return ForecastingDataset(series_len, horizon, self.raw_X, start, end, self.sc, self.mn, ft_num)


class ForecastingDataset(Dataset):
    def __init__(self, series_len, horizon, raw_X, start, end, sc, mn, ft_num=0):
        self.sc = sc
        self.mn = mn
        self.start = start
        self.end = end
        self.series_len = series_len
        self.horizon = horizon
        self.ft_num = ft_num
        if self.ft_num > 0:
            assert self.start > 0, "fine tuning mode only work on validate and test set"
        assert self.start < self.end, "start point should before end point"

        # start = 0 indicates training set and consider the shift of series_lens
        if self.start == 0:
            assert self.series_len < self.end
            self.rse = np.sum((raw_X[self.series_len:self.end] - np.mean(raw_X[self.series_len:self.end]))**2)
        else:
            self.rse = np.sum((raw_X[self.start:self.end] - np.mean(raw_X[self.start:self.end])) ** 2)

        self.X = self.norm(raw_X).astype(np.float32)

    def norm(self, X):
        return (X - self.mn) / self.sc

    def renorm(self, X):
        return X * self.sc + self.mn

    def __len__(self):
        if self.start == 0:
            return self.end + 1 - self.series_len - self.horizon
        else:
            return np.ceil((self.end - self.start) / (self.horizon)).astype(int)

    def __getitem__(self, idx):
        if self.start == 0:
            return (self.X[idx:idx+self.series_len], self.X[idx+self.series_len:idx+self.series_len+self.horizon])
        else:
            if self.ft_num == 0:
                return (self.X[(self.start + idx * self.horizon - self.series_len):(self.start + idx * self.horizon)],
                        self.X[(self.start + idx*self.horizon) : min(self.end, self.start + (idx+1)*self.horizon)])
            else:
                pt_data_x = np.array([self.X[(self.start + idx * self.horizon - self.series_len - ft_idx):(self.start + idx * self.horizon - ft_idx)] for ft_idx in range(1, self.ft_num+1)])
                pt_data_y = np.array([self.X[self.start + idx*self.horizon - ft_idx : (min(self.end, self.start + (idx+1)*self.horizon) - ft_idx) ] for ft_idx in range(1, self.ft_num+1)])

                x = self.X[(self.start + idx * self.horizon - self.series_len):(self.start + idx * self.horizon)]
                y = self.X[(self.start + idx*self.horizon) : min(self.end, self.start + (idx+1)*self.horizon)]
                return(pt_data_x, pt_data_y, x, y)



