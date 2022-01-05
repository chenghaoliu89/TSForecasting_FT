import logging
import numpy as np
import sys
import os
import copy
import pickle
import importlib


from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
import torch
import torch.nn as nn


from args import args
from utils import create_dir, ForecastingData
from runner.runner import Runner


class forecastingRunner(Runner):
    def __init__(self, model, data):
        super().__init__(model, data)
        self.criterion = nn.MSELoss()

    def run(self):
        bad_limit = 0

        if args.fine_tuning:
            model = self.bst_model
            model.load_state_dict(torch.load(args.model_loadpath))

            with torch.no_grad():
                ft_info = 'fine tuning epoch = {},'
                val_results = self.one_epoch(0, 1)
                ft_info += ' , val loss = {:.6f}'.format(val_results['loss'])
                ft_info += ' , val err = {:.6f}'.format(val_results['err'])

                tst_results = self.one_epoch(0, 2)
                ft_info += ' , tst loss = {:.6f}'.format(tst_results['loss'])
                ft_info += ' , tst err = {:.6f}'.format(tst_results['err'])
            logging.info(ft_info)

        else:
            for epoch in range(1, args.n_epochs+1):
                trn_results = self.one_epoch(epoch, 0)
                self.model_scheduler.step()

                epoch_info = 'epoch = {} , trn loss = {:.6f}'.format(epoch, trn_results['loss'])
                epoch_info += ' , trn err = {:.6f}'.format(trn_results['err'])


                with torch.no_grad():
                    val_results = self.one_epoch(epoch, 1)
                    epoch_info += ' , val loss = {:.6f}'.format(val_results['loss'])
                    epoch_info += ' , val err = {:.6f}'.format(val_results['err'])

                    tst_results = self.one_epoch(epoch, 2)
                    epoch_info += ' , tst loss = {:.6f}'.format(tst_results['loss'])
                    epoch_info += ' , tst err = {:.6f}'.format(tst_results['err'])

                logging.info(epoch_info)

                if val_results['err'] < self.bst_val_err:
                    self.bst_val_err = val_results['err']
                    bad_limit = 0
                    self.bst_model = copy.deepcopy(self.model)
                    torch.save(self.bst_model.state_dict(), os.path.join(args.output_dir, args.dataset + '_' + args.model_type + '_' + 'bstmodel.pth'))
                else:
                    bad_limit += 1
                if args.bad_limit > 0 and bad_limit >= args.bad_limit:
                    break


    def one_epoch(self, epoch, mode):

        if mode == 0:
            self.model.train()
        else:
            self.model.eval()

        sc = torch.tensor(self.dataloaders[0].dataset.sc).to(args.device)
        rse = self.dataloaders[mode].dataset.rse

        results = dict()
        epoch_err = 0
        epoch_loss = 0
        with torch.autograd.set_grad_enabled(mode==0):
            for i, (x, y) in enumerate(self.dataloaders[mode]):

                bs = x.shape[0]
                x = x.to(args.device)
                y = y.squeeze().to(args.device)

                inp = x
                prd_y = self.model(inp)

                loss = self.criterion(y, prd_y)
                epoch_loss += loss.item() * bs

                epoch_err += torch.sum((y*sc - prd_y*sc)**2).item()

                if mode == 0:
                    loss.backward()
                    self.model_opt.step()
                    self.model_opt.zero_grad()

        results['err'] = (epoch_err/rse)**0.5
        results['loss'] = epoch_loss / len(self.dataloaders[mode].dataset)
        return results
