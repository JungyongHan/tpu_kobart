#!/bin/bash

# KoBART 요약 모델 TPU 학습 스크립트
# 순수 PyTorch 구현 버전

# 환경 변수 설정
# export XLA_USE_BF16=1  # BF16 정밀도 사용 (TPU에 최적화)

# 학습 실행
python train.py \
    --train_file ../data/train.csv \
    --test_file ../data/test.csv \
    --batch_size 8 \
    --max_len 1024 \
    --max_epochs 10 \
    --lr 3e-5 \
    --gradient_clip_val 1.0 \
    --checkpoint ../checkpoint/ \
    --num_workers 4 \
    --save_steps 100 \
    --eval_steps 100 \
    --logging_steps 10

echo "TPU 학습이 완료되었습니다."