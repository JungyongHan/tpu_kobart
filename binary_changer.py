import argparse
import os
import torch
from transformers import BartForConditionalGeneration, PreTrainedTokenizerFast
from loguru import logger


def convert_checkpoint_to_huggingface(checkpoint_path, output_dir, model_name="gogamza/kobart-base-v2"):
    """
    PyTorch 체크포인트(.pt) 파일을 Hugging Face 형식으로 변환합니다.
    
    Args:
        checkpoint_path (str): 변환할 체크포인트 파일 경로
        output_dir (str): 변환된 모델을 저장할 디렉토리 경로
        model_name (str): 기본 모델 이름 (기본값: 'gogamza/kobart-base-v2')
    """
    # 출력 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)
    
    logger.info(f"체크포인트 로드 중: {checkpoint_path}")
    # 체크포인트 로드
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    
    # 모델 상태 딕셔너리 추출
    if "model" in checkpoint:
        state_dict = checkpoint["model"]
        logger.info("체크포인트에서 모델 상태 딕셔너리를 추출했습니다.")
    else:
        logger.error("체크포인트에서 모델 상태 딕셔너리를 찾을 수 없습니다.")
        return False
    
    # 토크나이저 로드
    logger.info(f"토크나이저 로드 중: {model_name}")
    tokenizer = PreTrainedTokenizerFast.from_pretrained(model_name)
    
    # 특수 토큰 추가 (KoBART 요약 모델에서 사용)
    special_tokens_dict = {'additional_special_tokens': ['<LF>']}
    tokenizer.add_special_tokens(special_tokens_dict)
    
    # 기본 모델 로드
    logger.info(f"기본 모델 로드 중: {model_name}")
    model = BartForConditionalGeneration.from_pretrained(model_name)
    
    # 토크나이저에 맞게 임베딩 크기 조정
    model.resize_token_embeddings(len(tokenizer))
    
    # 체크포인트의 가중치를 모델에 로드
    # 모델 키 이름 변환 (필요한 경우)
    new_state_dict = {}
    missing_keys = []
    unexpected_keys = []
    
    # 모델 상태 딕셔너리의 키 확인
    model_keys = set(model.state_dict().keys())
    checkpoint_keys = set(state_dict.keys())
    
    # 모델에 있지만 체크포인트에 없는 키
    missing_keys = list(model_keys - checkpoint_keys)
    # 체크포인트에 있지만 모델에 없는 키
    unexpected_keys = list(checkpoint_keys - model_keys)
    
    if missing_keys:
        logger.warning(f"모델에는 있지만 체크포인트에 없는 키: {len(missing_keys)}개")
        for key in missing_keys[:5]:  # 처음 5개만 출력
            logger.warning(f"  - {key}")
        if len(missing_keys) > 5:
            logger.warning(f"  ... 그리고 {len(missing_keys) - 5}개 더")
    
    if unexpected_keys:
        logger.warning(f"체크포인트에는 있지만 모델에 없는 키: {len(unexpected_keys)}개")
        for key in unexpected_keys[:5]:  # 처음 5개만 출력
            logger.warning(f"  - {key}")
        if len(unexpected_keys) > 5:
            logger.warning(f"  ... 그리고 {len(unexpected_keys) - 5}개 더")
    
    # 키 매핑 및 로드
    for k, v in state_dict.items():
        if k in model.state_dict():
            if v.shape == model.state_dict()[k].shape:
                new_state_dict[k] = v
            else:
                logger.warning(f"형태 불일치 무시: {k}, 체크포인트: {v.shape}, 모델: {model.state_dict()[k].shape}")
    
    # 모델에 가중치 로드
    model.load_state_dict(new_state_dict, strict=False)
    logger.info(f"모델에 체크포인트 가중치를 로드했습니다.")
    
    # Hugging Face 형식으로 저장
    logger.info(f"모델을 Hugging Face 형식으로 저장 중: {output_dir}")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    # 모델 구성 정보 저장 (필요한 경우 추가 정보 저장)
    with open(os.path.join(output_dir, "training_info.txt"), "w", encoding="utf-8") as f:
        f.write(f"Original checkpoint: {checkpoint_path}\n")
        f.write(f"Base model: {model_name}\n")
        if "epoch" in checkpoint:
            f.write(f"Epoch: {checkpoint['epoch']}\n")
        if "step" in checkpoint:
            f.write(f"Step: {checkpoint['step']}\n")
    
    logger.info(f"변환 완료! 모델이 {output_dir}에 저장되었습니다.")
    return True


def main():
    parser = argparse.ArgumentParser(description="PyTorch 체크포인트(.pt)를 Hugging Face 형식으로 변환")
    parser.add_argument(
        "--checkpoint", 
        type=str, 
        required=True,
        help="변환할 체크포인트 파일 경로"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        required=True,
        help="변환된 모델을 저장할 디렉토리 경로"
    )
    parser.add_argument(
        "--model_name", 
        type=str, 
        default="gogamza/kobart-base-v2",
        help="기본 모델 이름 (기본값: 'gogamza/kobart-base-v2')"
    )
    
    args = parser.parse_args()
    
    # 체크포인트 파일 존재 확인
    if not os.path.exists(args.checkpoint):
        logger.error(f"체크포인트 파일을 찾을 수 없습니다: {args.checkpoint}")
        return
    
    # 변환 실행
    convert_checkpoint_to_huggingface(
        args.checkpoint,
        args.output_dir,
        args.model_name
    )


if __name__ == "__main__":
    main()