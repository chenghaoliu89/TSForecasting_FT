import logging
import numpy as np
import sys
import os
import importlib


import torch
import torch.nn as nn


import args
from utils import create_dir, ForecastingData


def run():

    if args.task_type == 'forecasting':
        # load raw dataset
        data_path = os.path.join(args.data_dir, f'{args.dataset}.npy')
        raw_X = np.load(data_path, allow_pickle=True)
        args.n_series = raw_X[0].shape[-1]

        data = ForecastingData(raw_X, args)

    # datasets = []
    # for i in range(3):
    #     datasets.append(data.get_dataset(i))

    # def model_decay(epoch):
    #     return args.model_decay_rate**epoch

    model_package = importlib.import_module(f'models.{args.task_type}.{args.model_type}')
    org_model = getattr(model_package, args.model_type)(args).to(args.device)
    model = org_model

    if args.model_type == 'AGCRN':
        for p in model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            else:
                nn.init.uniform_(p)

    # for name, param in model.named_parameters():
    #     logging.info(name, param.shape, param.requires_grad)

    total_num = sum([param.nelement() for param in model.parameters()])
    logging.info('total num of parameters: {}'.format(total_num))

    runner_package = importlib.import_module(f'runner.{args.task_type}_runner')
    runner = getattr(runner_package, f'{args.task_type}Runner')(args, model, data)
    runner.run(args)


if __name__ == '__main__':

    # torch.backends.cudnn.benchmark = True

    args = args.all_args()

    if not os.path.isdir(args.output_dir):
        create_dir(args.output_dir)
    if args.device == 'auto':
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # FORMAT = '%(asctime)s %(levelname)s: %(message)s'
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logfile_path = args.dataset + '_' + args.model_type + '_' + 'series_len' + str(args.series_len) + '_'
    if args.fine_tuning:
        logfile_path += 'ft-log.txt'
    elif args.evaluate:
        logfile_path += 'eval-log.txt'
    else:
        logfile_path += 'log.txt'
    output_file_handler = logging.FileHandler(os.path.join(args.output_dir, logfile_path), mode='w')
    stdout_handler = logging.StreamHandler(sys.stdout)
    logger.addHandler(output_file_handler)
    logger.addHandler(stdout_handler)

    logging.info(args)

    run()
