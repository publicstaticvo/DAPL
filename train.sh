#! /bin/bash

modelpath=./bertmodel
model=dapl
data=~/data/

python train.py --data_dir $data --model $model --model_path $modelpath --num_ways 5 --num_shots 1
python train.py --data_dir $data --model $model --model_path $modelpath --num_ways 5 --num_shots 5 --seed 1 --warmup 0.05
python train.py --data_dir $data --model $model --model_path $modelpath --num_ways 10 --num_shots 1 --seed 1 --warmup 0.05
python train.py --data_dir $data --model $model --model_path $modelpath --num_ways 10 --num_shots 5 --seed 42 --warmup 0.05 --lr 2e-5

