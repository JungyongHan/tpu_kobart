#!/bin/bash
git stash
git pull
git stash pop
# 학습 실행
PJRT_DEVICE=TPU | ~/venv/bin/python train.py \
    --train_file ./data/train.csv \
    --batch_size 16 \
    --max_len 1024 \
    --max_epochs 1000 \
    --lr 4e-5 \
    --gradient_clip_val 1.0 \
    --checkpoint ./checkpoint/ \
    --num_workers 16 \
    --save_steps 1000 \
    --logging_steps 200

echo "TPU 학습이 완료되었습니다."