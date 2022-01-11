#!/usr/bin/env bash
n=50
while (( $n <= 200 ))
do
	echo "Series Length is $n."
	python3 main.py --output_dir "outputs"  --n_epochs 750  --bad_limit 25 --lr 3e-3  --batch_size 64  --series_len ${n}  --norm_type standard  forecasting  --model_type "TCN" --dataset "air-quality"
	n=$(( n+5))
done