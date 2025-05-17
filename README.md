# KoBART 요약 모델 - 순수 PyTorch TPU 학습

이 폴더는 Lightning 프레임워크를 사용하지 않고 순수 PyTorch만을 사용하여 TPU 환경에서 KoBART 요약 모델을 학습하는 코드를 포함하고 있습니다.

## 특징

- Lightning 대신 순수 PyTorch 사용
- TPU 슬라이스 환경에 최적화된 분산 학습 구현
- `torch.distributed`와 `torch.nn.parallel.DistributedDataParallel` 사용
- `xla_multiprocessing` 대신 `xla.launch` 사용
- 체크포인트 저장 및 복원 기능 구현
- 학습 중 정기적인 평가 및 최고 성능 모델 저장

## 설치

```bash
pip install -r requirements.txt
```

## 데이터 준비

학습 데이터는 다음 형식의 CSV 파일이어야 합니다:

```
article,script
원문 텍스트,요약 텍스트
...
```

## 학습 실행

```bash
python train.py \
    --train_file ../data/train.tsv \
    --test_file ../data/test.tsv \
    --batch_size 32 \
    --max_len 512 \
    --max_epochs 10 \
    --lr 3e-5 \
    --num_tpu_cores 16 \
    --gradient_clip_val 1.0 \
    --checkpoint ../checkpoint/pure_torch \
    --num_workers 4 \
    --save_steps 100 \
    --eval_steps 100 \
    --logging_steps 10
```

## 주요 매개변수

- `--train_file`: 학습 데이터 파일 경로
- `--test_file`: 검증 데이터 파일 경로
- `--batch_size`: TPU 코어당 배치 크기
- `--max_len`: 최대 시퀀스 길이
- `--max_epochs`: 학습 에폭 수
- `--lr`: 학습률
- `--num_tpu_cores`: TPU 코어/슬라이스 수
- `--gradient_clip_val`: 그래디언트 클리핑 값
- `--checkpoint`: 체크포인트 저장 디렉토리
- `--resume_from_checkpoint`: 마지막 체크포인트에서 학습 재개 (플래그)
- `--save_steps`: 모델 저장 간격 (스텝 단위)
- `--eval_steps`: 평가 간격 (스텝 단위)
- `--logging_steps`: 로깅 간격 (스텝 단위)

## 구현 차이점

### Lightning vs 순수 PyTorch

이 구현은 Lightning 프레임워크를 사용하지 않고 순수 PyTorch API만을 사용하여 TPU 학습을 구현했습니다. 주요 차이점은 다음과 같습니다:

1. 모델 정의: `nn.Module`을 상속받아 직접 모델 클래스 구현
2. 학습 루프: 수동으로 학습/평가 루프 구현
3. 분산 학습: `DistributedDataParallel`을 사용한 명시적 분산 학습 설정
4. 체크포인트: 수동으로 체크포인트 저장/로드 로직 구현

### TPU 최적화

- `xla_multiprocessing` 대신 `xla.launch`와 `torch.distributed` 사용
- `MpDeviceLoader`를 사용하여 TPU에 최적화된 데이터 로딩
- `xm.optimizer_step()`을 사용하여 TPU에 최적화된 옵티마이저 스텝
- `xm.mesh_reduce()`를 사용하여 분산 환경에서 값 동기화

## 모델 저장 구조

- `{checkpoint}/model_epoch_{epoch}_step_{step}`: 정기적으로 저장되는 체크포인트
- `{checkpoint}/best_model`: 검증 손실이 가장 낮은 모델
- `{checkpoint}/final_model`: 학습 완료 후 최종 모델
- `{checkpoint}/last.pt`: 학습 재개를 위한 상태 정보