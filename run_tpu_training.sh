#!/bin/bash

rm ./nohup.out

git stash
git pull
git stash pop
# 학습 실행
PJRT_DEVICE=TPU | ~/venv/bin/python train.py \
    --train_file ./data/train.csv \
    --test_file ./data/test.csv \
    --batch_size 20 \
    --max_len 256 \
    --max_epochs 500 \
    --lr 3e-5 \
    --gradient_clip_val 1.0 \
    --checkpoint ./checkpoint/ \
    --num_workers 16 \
    --logging_steps 10 \
    --save_epoch 10 \
    --use_wandb # if you want to use wandb, add this argument. Needed  ~/venv/bin/python -m wandb login <API_KEY> .

echo "TPU 학습이 완료되었습니다."