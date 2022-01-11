#!/bin/bash

for lr in 0.1 0.05 0.01 0.005 0.001 0.0005 0.0001 0.00005
do
  echo "fine tuning learning rate $lr"
  python3 main.py --output_dir "outputs"  --n_epochs 750  --bad_limit 25 --lr 3e-3  --batch_size 64  --series_len 60  --norm_type standard  --fine_tuning --ft_num 5 --ft_steps 500 --ft_lr ${lr} forecasting  --model_type "TCN" --dataset "air-quality"

done