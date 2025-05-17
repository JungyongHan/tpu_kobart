import argparse
import numpy as np
import pandas as pd
import os
import time
from loguru import logger

import torch
import torch.nn as nn
import torch.distributed as dist
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.distributed.parallel_loader as pl
import torch_xla.distributed.xla_launch as xla_launch

from transformers import BartForConditionalGeneration, PreTrainedTokenizerFast
from transformers.optimization import get_linear_schedule_with_warmup

# 데이터셋 클래스 임포트
sys_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if sys_path not in sys.path:
    sys.path.append(sys_path)

import sys
from dataset import KoBARTSummaryDataset

parser = argparse.ArgumentParser(description='KoBART Summarization for TPU with Pure PyTorch')

class ArgsBase():
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(
            parents=[parent_parser], add_help=False)
        parser.add_argument('--train_file',
                            type=str,
                            default='../data/test.tsv',
                            help='train file')
        parser.add_argument('--test_file',
                            type=str,
                            default='../data/test.tsv',
                            help='test file')
        parser.add_argument('--batch_size',
                            type=int,
                            default=32,
                            help='batch size per TPU core')
        parser.add_argument('--checkpoint',
                            type=str,
                            default='../checkpoint',
                            help='checkpoint directory')
        parser.add_argument('--max_len',
                            type=int,
                            default=512,
                            help='max seq len')
        parser.add_argument('--max_epochs',
                            type=int,
                            default=10,
                            help='train epochs')
        parser.add_argument('--lr',
                            type=float,
                            default=3e-5,
                            help='The initial learning rate')
        parser.add_argument('--num_tpu_cores',
                            type=int,
                            default=16,
                            help='number of TPU cores/slices')
        parser.add_argument('--gradient_clip_val',
                            type=float,
                            default=1.0,
                            help='gradient_clipping')
        parser.add_argument('--resume_from_checkpoint',
                            action='store_true',
                            help='resume training from last checkpoint')
        parser.add_argument('--num_workers',
                            type=int,
                            default=4,
                            help='num of worker for dataloader')
        parser.add_argument('--warmup_ratio',
                            type=float,
                            default=0.1,
                            help='warmup ratio for scheduler')
        parser.add_argument('--save_steps',
                            type=int,
                            default=100,
                            help='steps interval for saving model')
        parser.add_argument('--eval_steps',
                            type=int,
                            default=100,
                            help='steps interval for evaluation')
        parser.add_argument('--logging_steps',
                            type=int,
                            default=10,
                            help='steps interval for logging')

        return parser


class KoBARTSummaryModel(nn.Module):
    def __init__(self, tokenizer):
        super().__init__()
        self.model = BartForConditionalGeneration.from_pretrained('gogamza/kobart-base-v1')
        self.model.resize_token_embeddings(len(tokenizer))
        self.pad_token_id = tokenizer.pad_token_id
        
    def forward(self, input_ids, decoder_input_ids, labels):
        attention_mask = input_ids.ne(self.pad_token_id).float()
        decoder_attention_mask = decoder_input_ids.ne(self.pad_token_id).float()
        
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            labels=labels,
            return_dict=True
        )


def train_step(model, batch, optimizer, device):
    model.train()
    
    # 데이터를 디바이스로 이동
    input_ids = batch['input_ids'].to(device)
    decoder_input_ids = batch['decoder_input_ids'].to(device)
    labels = batch['labels'].to(device)
    
    # 순전파
    outputs = model(input_ids, decoder_input_ids, labels)
    loss = outputs.loss
    
    # 역전파
    loss.backward()
    
    return loss


def validate(model, val_loader, device):
    model.eval()
    total_loss = 0
    total_samples = 0
    
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            decoder_input_ids = batch['decoder_input_ids'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids, decoder_input_ids, labels)
            loss = outputs.loss
            
            batch_size = input_ids.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size
    
    return total_loss / total_samples


def save_checkpoint(model, tokenizer, optimizer, scheduler, epoch, step, args, val_loss=None):
    checkpoint_path = os.path.join(args.checkpoint, f"model_epoch_{epoch}_step_{step}")
    os.makedirs(checkpoint_path, exist_ok=True)
    
    # 모델 저장
    model.model.save_pretrained(checkpoint_path)
    tokenizer.save_pretrained(checkpoint_path)
    
    # 옵티마이저, 스케줄러 상태 저장
    xm.save({
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'epoch': epoch,
        'step': step,
        'val_loss': val_loss
    }, os.path.join(checkpoint_path, 'training_state.pt'))
    
    # last.pt 파일 생성 (이어서 학습하기 위한 용도)
    last_path = os.path.join(args.checkpoint, "last.pt")
    xm.save({
        'model_path': checkpoint_path,
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'epoch': epoch,
        'step': step,
        'val_loss': val_loss
    }, last_path)
    
    logger.info(f"Checkpoint saved at {checkpoint_path}")


def train_kobart(rank, args):
    # 시드 설정
    torch.manual_seed(42 + rank)
    np.random.seed(42 + rank)
    dist.init_process_group("xla", init_method='xla://')
    # 디바이스 설정
    device = xm.xla_device()
    
    # 마스터 프로세스 확인
    is_master = xm.is_master_ordinal(local=False)
    is_local_master = xm.is_master_ordinal()
    
    if is_local_master:
        logger.info(f"Starting training on TPU core {rank}")
        os.makedirs(args.checkpoint, exist_ok=True)
    
    # 토크나이저 설정
    tokenizer = PreTrainedTokenizerFast.from_pretrained('gogamza/kobart-base-v1')
    special_tokens_dict = {'additional_special_tokens': ['<newline>']}
    tokenizer.add_special_tokens(special_tokens_dict)
    
    # 데이터셋 및 데이터로더 설정
    train_dataset = KoBARTSummaryDataset(args.train_file, tokenizer, args.max_len)
    val_dataset = KoBARTSummaryDataset(args.test_file, tokenizer, args.max_len)
    
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset,
        num_replicas=args.num_tpu_cores,
        rank=rank,
        shuffle=True
    )
    
    val_sampler = torch.utils.data.distributed.DistributedSampler(
        val_dataset,
        num_replicas=args.num_tpu_cores,
        rank=rank,
        shuffle=False
    )
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=args.num_workers,
        drop_last=True,
        pin_memory=True
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        sampler=val_sampler,
        num_workers=args.num_workers,
        drop_last=True,
        pin_memory=True
    )
    
    # TPU에 최적화된 데이터로더
    train_loader = pl.MpDeviceLoader(train_loader, device)
    val_loader = pl.MpDeviceLoader(val_loader, device)
    
    # 모델 설정
    model = KoBARTSummaryModel(tokenizer)
    model.to(device)

    xm.broadcast_master_param(model)
    
    # DDP 모델 설정
    model = torch.nn.parallel.DistributedDataParallel(
        model,
        gradient_as_bucket_view=True
    )
    
    # 옵티마이저 설정
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.lr)



    # 총 학습 스텝 계산
    total_steps = len(train_loader) * args.max_epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    
    # 스케줄러 설정
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    
    # 체크포인트에서 이어서 학습
    start_epoch = 0
    global_step = 0
    best_val_loss = float('inf')
    
    if args.resume_from_checkpoint:
        last_path = os.path.join(args.checkpoint, "last.pt")
        if os.path.exists(last_path):
            checkpoint = torch.load(last_path, map_location='cpu')
            model_path = checkpoint['model_path']
            
            if is_master:
                logger.info(f"Loading model from {model_path}")
            
            # 모델 로드
            model.module.model = BartForConditionalGeneration.from_pretrained(model_path)
            model.module.model.to(device)
            
            # 옵티마이저, 스케줄러 상태 로드
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            
            start_epoch = checkpoint['epoch']
            global_step = checkpoint['step']
            best_val_loss = checkpoint.get('val_loss', float('inf'))
            
            if is_master:
                logger.info(f"Resuming from epoch {start_epoch}, step {global_step}")
    
    def _log_summary(epoch, step, avg_loss, elapsed):
        print(f"Epoch: {epoch}, Step: {step}, Loss: {avg_loss:.4f}, Time: {elapsed:.2f}s")

    # 학습 루프
    for epoch in range(start_epoch, args.max_epochs):
        train_sampler.set_epoch(epoch)
        epoch_loss = 0
        epoch_steps = 0
        
        # 에폭 시작 시간
        start_time = time.time()
        
        for step, batch in enumerate(train_loader):
            # 그래디언트 초기화
            optimizer.zero_grad()
            
            # 학습 스텝
            loss = train_step(model.module, batch, optimizer, device)
            
            # 그래디언트 클리핑
            if args.gradient_clip_val > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip_val)
            
            # 옵티마이저 스텝
            # xm.optimizer_step(optimizer)
            optimizer.step()
            xm.mark_step()
            # 손실 누적
            epoch_loss += loss.item()
            epoch_steps += 1
            global_step += 1
            
            # 로깅
            if is_master and global_step % args.logging_steps == 0:
                avg_loss = epoch_loss / epoch_steps
                elapsed = time.time() - start_time
                # logger.info(f"Epoch: {epoch}, Step: {global_step}, Loss: {avg_loss:.4f}, Time: {elapsed:.2f}s")
                xm.add_step_closure(
                    _log_summary,
                    args=(epoch, global_step, avg_loss, elapsed)
                )
                        
            # 정기적인 체크포인트 저장
            if is_local_master and global_step % args.save_steps == 0:
                save_checkpoint(model.module, tokenizer, optimizer, scheduler, epoch, global_step, args)
        
        # 에폭 종료 후 평가
        val_loss = validate(model.module, val_loader, device)
        
        # 모든 프로세스에서 동기화
        val_loss = xm.mesh_reduce('val_loss', val_loss, lambda x: sum(x) / len(x))
        
        if is_local_master:
            epoch_avg_loss = epoch_loss / epoch_steps
            epoch_time = time.time() - start_time
            logger.info(f"Epoch: {epoch}, Step: {global_step}, Loss: {epoch_avg_loss:.4f}, Val Loss: {val_loss:.4f}, Time: {epoch_time:.2f}s")
            # 에폭 종료 후 체크포인트 저장
            save_checkpoint(model.module, tokenizer, optimizer, scheduler, epoch + 1, global_step, args, val_loss)
            
            # 최고 성능 모델 저장
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_path = os.path.join(args.checkpoint, f"best_model")
                os.makedirs(best_path, exist_ok=True)
                model.module.model.save_pretrained(best_path)
                tokenizer.save_pretrained(best_path)
                logger.info(f"New best model saved with val_loss: {val_loss:.4f}")
    
    # 학습 완료 후 최종 모델 저장
    if is_local_master:
        logger.info("Training completed, saving final model...")
        final_path = os.path.join(args.checkpoint, "final_model")
        os.makedirs(final_path, exist_ok=True)
        model.module.model.save_pretrained(final_path)
        tokenizer.save_pretrained(final_path)
        logger.info(f"Final model saved to {final_path}")


def _mp_fn(index, args):
    # TPU 프로세스 시작
    train_kobart(index, args)


if __name__ == '__main__':
    parser = ArgsBase.add_model_specific_args(parser)
    args = parser.parse_args()
    
    # 학습 시작
    logger.info("Starting TPU distributed training with pure PyTorch")
    logger.info(args)
    
    # TPU 분산 학습 시작 (xla.launch 사용)
    xla_launch.launch(_mp_fn, args=(args,), nprocs=args.num_tpu_cores)