#!/bin/bash

ep=750
model="TPA"
dataset="solar"
output_dir="outputs"

python3 main.py \
  --output_dir $output_dir \
  --n_epochs $ep \
  --bad_limit 25 \
  --lr 3e-3 \
  --rho_lr 1e-2 \
  --one_rho --inp_adj --out_adj \
  --batch_size 64 \
  --series_len 60 \
  --norm_type standard \
  forecasting \
  --model_type $model \
  --dataset $dataset

python3 main.py --output_dir "outputs"  --n_epochs 750  --bad_limit 25 --lr 3e-3  --rho_lr 1e-2  --one_rho --inp_adj --out_adj --batch_size 64  --series_len 60  --norm_type standard  forecasting  --model_type "TCN"  --dataset "exchange"
python3 main.py --output_dir "outputs"  --n_epochs 750  --bad_limit 25 --lr 3e-3  --rho_lr 1e-2  --one_rho --inp_adj --out_adj --batch_size 64  --series_len 60  --norm_type standard  forecasting  --model_type "TCN"  --dataset "exchange"



export CUDA_VISIBLE_DEVICES=0
python3 main.py --output_dir "outputs"  --n_epochs 750  --bad_limit 25 --lr 3e-3  --batch_size 64  --series_len 60  --norm_type standard  forecasting  --model_type "TCN"  --dataset "exchange"
python3 main.py --output_dir "outputs"  --n_epochs 750  --bad_limit 25 --lr 3e-3  --batch_size 64  --series_len 60  --norm_type standard  forecasting  --model_type "TCN"  --dataset "solar"
python3 main.py --output_dir "outputs"  --n_epochs 750  --bad_limit 25 --lr 3e-3  --batch_size 64  --series_len 60  --norm_type standard  forecasting  --model_type "TCN"  --dataset "electricity"
python3 main.py --output_dir "outputs"  --n_epochs 750  --bad_limit 25 --lr 3e-3  --batch_size 64  --series_len 60  --norm_type standard  forecasting  --model_type "TCN"  --dataset "air-quality"

CUDA_VISIBLE_DEVICES=1
python3 main.py --output_dir "outputs"  --n_epochs 750  --bad_limit 25 --lr 1e-3  --batch_size 64  --series_len 60  --norm_type standard  forecasting  --model_type "Trans"  --dataset "exchange"
python3 main.py --output_dir "outputs"  --n_epochs 750  --bad_limit 25 --lr 1e-3  --batch_size 64  --series_len 60  --norm_type standard  forecasting  --model_type "Trans"  --dataset "solar"
python3 main.py --output_dir "outputs"  --n_epochs 750  --bad_limit 25 --lr 1e-3  --batch_size 64  --series_len 60  --norm_type standard  forecasting  --model_type "Trans"  --dataset "electricity"
python3 main.py --output_dir "outputs"  --n_epochs 750  --bad_limit 25 --lr 1e-3  --batch_size 64  --series_len 60  --norm_type standard  forecasting  --model_type "Trans"  --dataset "air-quality"

CUDA_VISIBLE_DEVICES=2
python3 main.py --output_dir "outputs"  --n_epochs 750  --bad_limit 25 --lr 3e-3  --batch_size 64  --series_len 60  --norm_type standard  forecasting  --model_type "LSTM"  --dataset "exchange"
python3 main.py --output_dir "outputs"  --n_epochs 750  --bad_limit 25 --lr 3e-3  --batch_size 64  --series_len 60  --norm_type standard  forecasting  --model_type "LSTM"  --dataset "solar"
python3 main.py --output_dir "outputs"  --n_epochs 750  --bad_limit 25 --lr 3e-3  --batch_size 64  --series_len 60  --norm_type standard  forecasting  --model_type "LSTM"  --dataset "electricity"
python3 main.py --output_dir "outputs"  --n_epochs 750  --bad_limit 25 --lr 3e-3  --batch_size 64  --series_len 60  --norm_type standard  forecasting  --model_type "LSTM"  --dataset "air-quality"
