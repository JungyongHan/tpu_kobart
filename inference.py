import argparse
import torch
import torch_xla.core.xla_model as xm
import pandas as pd
import numpy as np
from transformers import BartForConditionalGeneration, PreTrainedTokenizerFast
from loguru import logger
import os

parser = argparse.ArgumentParser(description='KoBART 요약 모델 추론')
parser.add_argument('--model_dir', type=str, default='../checkpoint/pure_torch/final_model', help='추론에 사용할 모델 경로')
parser.add_argument('--input_file', type=str, default=None, help='입력 데이터 파일 (없으면 대화형 모드)')
parser.add_argument('--output_file', type=str, default='inference_results.csv', help='결과 저장 파일')
parser.add_argument('--max_length', type=int, default=128, help='생성할 요약의 최대 길이')
parser.add_argument('--num_beams', type=int, default=4, help='빔 서치 크기')
parser.add_argument('--use_tpu', action='store_true', help='TPU 사용 여부')
args = parser.parse_args()

def load_model_and_tokenizer(model_dir, use_tpu=False):
    # 모델 및 토크나이저 로드
    model = BartForConditionalGeneration.from_pretrained(model_dir)
    tokenizer = PreTrainedTokenizerFast.from_pretrained(model_dir)
    
    # TPU 사용 시 디바이스 설정
    if use_tpu:
        device = xm.xla_device()
        model.to(device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
    
    model.eval()
    return model, tokenizer, device

def generate_summary(text, model, tokenizer, device, max_length=128, num_beams=4):
    # 개행문자 처리
    text = text.replace('\n', '<newline>')
    
    # 토큰화 및 모델 입력 준비
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # 요약 생성
    with torch.no_grad():
        generated_ids = model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=max_length,
            num_beams=num_beams,
            early_stopping=True
        )
    
    # 생성된 요약 디코딩
    summary = tokenizer.decode(generated_ids[0], skip_special_tokens=False)
    
    # 특수 토큰 처리
    summary = summary.replace('<s>', '').replace('</s>', '').replace('<pad>', '')
    summary = summary.replace('<newline>', '\n')
    
    return summary.strip()

def batch_inference(input_file, output_file, model, tokenizer, device, max_length=128, num_beams=4):
    # 입력 데이터 로드
    if input_file.endswith('.csv'):
        data = pd.read_csv(input_file)
    elif input_file.endswith('.tsv'):
        data = pd.read_csv(input_file, sep='\t')
    else:
        raise ValueError("지원되지 않는 파일 형식입니다. CSV 또는 TSV 파일을 사용하세요.")
    
    # 원문 컬럼 확인
    if 'article' in data.columns:
        text_column = 'article'
    else:
        text_column = data.columns[0]  # 첫 번째 컬럼을 텍스트로 가정
    
    logger.info(f"총 {len(data)}개의 텍스트에 대해 요약을 생성합니다.")
    
    # 결과 저장을 위한 리스트
    original_texts = []
    generated_summaries = []
    
    # 각 텍스트에 대해 요약 생성
    for i, row in enumerate(data[text_column]):
        if i % 10 == 0:
            logger.info(f"진행 상황: {i}/{len(data)}")
        
        text = str(row)
        summary = generate_summary(text, model, tokenizer, device, max_length, num_beams)
        
        original_texts.append(text)
        generated_summaries.append(summary)
    
    # 결과를 데이터프레임으로 변환하여 저장
    results_df = pd.DataFrame({
        'original_text': original_texts,
        'generated_summary': generated_summaries
    })
    
    # 참조 요약이 있는 경우 추가
    if 'script' in data.columns:
        results_df['reference_summary'] = data['script'].tolist()
    
    # 결과 저장
    results_df.to_csv(output_file, index=False, encoding='utf-8')
    logger.info(f"추론 결과가 {output_file}에 저장되었습니다.")

def interactive_mode(model, tokenizer, device, max_length=128, num_beams=4):
    logger.info("대화형 모드를 시작합니다. 종료하려면 'exit' 또는 'quit'를 입력하세요.")
    
    while True:
        text = input("\n원문을 입력하세요: ")
        
        if text.lower() in ['exit', 'quit']:
            break
        
        if not text.strip():
            continue
        
        summary = generate_summary(text, model, tokenizer, device, max_length, num_beams)
        print("\n생성된 요약:")
        print(summary)

def main():
    logger.info(f"모델 경로: {args.model_dir}")
    
    # 모델 및 토크나이저 로드
    model, tokenizer, device = load_model_and_tokenizer(args.model_dir, args.use_tpu)
    logger.info(f"모델이 {device}에 로드되었습니다.")
    
    # 입력 파일이 있으면 배치 추론, 없으면 대화형 모드
    if args.input_file:
        logger.info(f"입력 파일: {args.input_file}")
        batch_inference(args.input_file, args.output_file, model, tokenizer, device, args.max_length, args.num_beams)
    else:
        interactive_mode(model, tokenizer, device, args.max_length, args.num_beams)

if __name__ == "__main__":
    main()