import os
import argparse


import task_args

def all_args(str = None):
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_decay_rate', type=float, default=1)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--data_dir', type=str, default='data')

    parser.add_argument('--device', type=str, default='auto', choices=['cpu', 'cuda', 'auto'], help='device type (assign automatically)')
    parser.add_argument('--lmbd', type=float, default=0.0, help='lambda value for regularization')
    parser.add_argument('--n_epochs', type=int, default=50)
    parser.add_argument('--bad_limit', type=int, default=0, help='# of non-improving epochs before early-stopping')
    parser.add_argument('--n_series', type=int, help='# of series in the data (assign automatically)')
    parser.add_argument('--series_len', type=int, default=64, help='window size of a single example')
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--norm_type', type=str, default='none', choices=['none', 'minmax', 'standard'])
    parser.add_argument('--local_norm', action='store_true', help='normalize data per series')
    parser.add_argument('--valid_ratio', type=float, default=0.2, help='ratio of data for validation')
    parser.add_argument('--test_ratio', type=float, default=0.2, help='ratio of data for testing')

    parser.add_argument('--fine_tuning', action='store_true', help='load pre-trained model and fine tuning')

    subparsers = parser.add_subparsers(dest='task_type', help='task_type')

    for name, task_arg in task_args.__dict__.items():
        if callable(task_arg):
            task_arg(subparsers)

    args = parser.parse_args(str)
    args.data_dir = os.path.join(args.data_dir, args.task_type)
    args.output_dir = os.path.join(args.output_dir, args.task_type)
    args.model_loadpath = os.path.join(args.output_dir, args.dataset + '_' + args.model_type + '_' + 'bstmodel.pth')

    return args

args = all_args()
