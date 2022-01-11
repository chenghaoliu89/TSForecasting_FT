import logging
import numpy as np
import sys
import os
import copy
import pickle
import importlib
from copy import deepcopy

from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from torch.nn import functional as F


from utils import create_dir, ForecastingData
from runner.runner import Runner


class forecastingRunner(Runner):
    def __init__(self, args, model, data):
        super().__init__(args, model, data)
        self.criterion = nn.MSELoss()

    def run(self, args):
        # load model, fine tuning and evaluation
        if args.fine_tuning:
            model = self.bst_model
            model.load_state_dict(torch.load(args.model_loadpath, map_location=args.device))


            tst_results = self.fine_tuning(args, 2)
            for i in range(tst_results['loss'].shape[0]):
                ft_info = 'fine tuning epoch = {} , tst loss = {:.6f} , tst err = {:.6f}'.format(i, tst_results['loss'][i], tst_results['err'][i])
                logging.info(ft_info)
        else:
            # load model and evaluation
            if args.evaluate:
                model = self.bst_model
                model.load_state_dict(torch.load(args.model_loadpath, map_location=args.device))

                eval_info = 'evaluation epoch = {}'
                with torch.no_grad():
                    val_results = self.one_epoch(args, 1)
                    eval_info += ' , val loss = {:.6f}'.format(val_results['loss'])
                    eval_info += ' , val err = {:.6f}'.format(val_results['err'])

                    tst_results = self.one_epoch(args, 2)
                    eval_info += ' , tst loss = {:.6f}'.format(tst_results['loss'])
                    eval_info += ' , tst err = {:.6f}'.format(tst_results['err'])
                logging.info(eval_info)
            # model training and evaluation
            else:
                bad_limit = 0
                for epoch in range(1, args.n_epochs+1):
                    trn_results = self.one_epoch(args, 0)
                    self.model_scheduler.step()

                    epoch_info = 'epoch = {} , trn loss = {:.6f}'.format(epoch, trn_results['loss'])
                    epoch_info += ' , trn err = {:.6f}'.format(trn_results['err'])


                    with torch.no_grad():
                        val_results = self.one_epoch(args, 1)
                        epoch_info += ' , val loss = {:.6f}'.format(val_results['loss'])
                        epoch_info += ' , val err = {:.6f}'.format(val_results['err'])

                        tst_results = self.one_epoch(args, 2)
                        epoch_info += ' , tst loss = {:.6f}'.format(tst_results['loss'])
                        epoch_info += ' , tst err = {:.6f}'.format(tst_results['err'])

                    logging.info(epoch_info)

                    if val_results['err'] < self.bst_val_err:
                        self.bst_val_err = val_results['err']
                        bad_limit = 0
                        self.bst_model = copy.deepcopy(self.model)
                        torch.save(self.bst_model.state_dict(), os.path.join(args.output_dir, args.dataset + '_' + args.model_type + '_' + 'series_len' + str(args.series_len) + '_' + 'bstmodel.pth'))
                    else:
                        bad_limit += 1
                    if args.bad_limit > 0 and bad_limit >= args.bad_limit:
                        break


    def one_epoch(self, args, mode):

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

    def fine_tuning(self, args, mode):
        self.model.train()
        basemodel_parameteres = deepcopy(self.model.state_dict())

        sc = torch.tensor(self.dataloaders[0].dataset.sc).to(args.device)
        rse = self.dataloaders[mode].dataset.rse
        results = dict()

        example_loss = np.zeros(args.ft_steps + 1)
        example_err = np.zeros(args.ft_steps + 1)
        for (pt_datax, pt_datay, x, y) in self.dataloaders[mode]:
            bs = x.shape[0]
            x = x.to(args.device)
            y = y.squeeze().to(args.device)
            pt_datax = pt_datax.to(args.device)
            pt_datay = pt_datay.squeeze().to(args.device)

            for i in range(bs):
                self.model.load_state_dict(basemodel_parameteres)
                optimizer_ft = torch.optim.SGD(self.model.parameters(), lr=args.ft_lr)
                self.model.train()
                ft_x = pt_datax[i, :, :, :]
                ft_y = pt_datay[i, :, :]

                # without fine tuning
                with torch.set_grad_enabled(False):
                    prd_y = self.model(x[i, :, :].unsqueeze(0))
                    loss = self.criterion(y[i, :], prd_y.squeeze())
                    example_loss[0] += loss.item()
                    example_err[0] += torch.sum((y[i, :] * sc - prd_y.squeeze() * sc) ** 2).item()

                # fine tuning with args.ft_steps SGD steps
                for k in range(1, args.ft_steps+1):
                    optimizer_ft.zero_grad()
                    with torch.set_grad_enabled(True):
                        prd_y = self.model(ft_x)
                        loss_ft = self.criterion(prd_y, ft_y)
                        loss_ft.backward()
                        optimizer_ft.step()

                    with torch.set_grad_enabled(False):
                        prd_y = self.model(x[i,:,:].unsqueeze(0))
                        loss = self.criterion(y[i,:], prd_y.squeeze())
                        example_loss[k] += loss.item()
                        example_err[k] += torch.sum((y[i,:] * sc - prd_y.squeeze() * sc) ** 2).item()

        results['err'] = (example_err/rse)**0.5
        results['loss'] = example_loss / len(self.dataloaders[mode].dataset)
        return results

    # non parametric fine tuning
    def fine_tuning_np(self, args, mode):
        self.model.eval()
        sc = torch.tensor(self.dataloaders[0].dataset.sc).to(args.device)
        rse = self.dataloaders[mode].dataset.rse
        results = dict()
        return






