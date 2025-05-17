from torch_xla import runtime as xr
import argparse
import numpy as np
import os
import time
from loguru import logger

import torch
import torch.nn as nn
import torch.distributed as dist
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.distributed.parallel_loader as pl
import torch_xla.distributed.xla_backend

from transformers import BartForConditionalGeneration, PreTrainedTokenizerFast
from transformers.optimization import get_linear_schedule_with_warmup

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
        parser.add_argument('--logging_steps',
                            type=int,
                            default=10,
                            help='steps interval for logging')

        return parser


class KoBARTSummaryModel(nn.Module):
    def __init__(self, tokenizer):
        super().__init__()
        self.model = BartForConditionalGeneration.from_pretrained('gogamza/kobart-base-v2')
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
    
    def save_pretrained(self, path):
        """내부 모델을 저장하는 메서드"""
        self.model.save_model(path)



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
    if hasattr(model, "module"):
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()
    
    # 옵티마이저, 스케줄러 상태 저장
    xm.save({
        'model':state_dict,
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'epoch': epoch,
        'step': step,
    }, os.path.join(args.checkpoint, f'checkpoint_{epoch}.pt'))
    

def train_kobart(rank, args):
    # 시드 설정
    torch.manual_seed(42)
    np.random.seed(42)
    dist.init_process_group("xla", init_method='xla://')
    device = xm.xla_device()
    
    # 마스터 프로세스 확인
        
    if is_local_master := xm.is_master_ordinal():
        logger.info(f"Starting training on TPU core {rank}")
        os.makedirs(args.checkpoint, exist_ok=True)
        
    
    if is_master := xm.is_master_ordinal(False):
        logger.info("Starting training on TPU core 0")
        # os.environ["PT_XLA_DEBUG_LEVEL"] = 2
        is_local_master = True
    
    # 토크나이저 설정
    tokenizer = PreTrainedTokenizerFast.from_pretrained('gogamza/kobart-base-v2')
    special_tokens_dict = {'additional_special_tokens': ['<LF>']}
    tokenizer.add_special_tokens(special_tokens_dict)
    
    # 데이터셋 및 데이터로더 설정
    train_dataset = KoBARTSummaryDataset(args.train_file, tokenizer, args.max_len)
    if os.path.exists(args.train_file):
        val_dataset = KoBARTSummaryDataset(args.test_file, tokenizer, args.max_len)

        val_sampler = torch.utils.data.distributed.DistributedSampler(
            val_dataset,
            num_replicas=xr.world_size(),
            rank=xr.global_ordinal(),
            shuffle=False
        )

        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            sampler=val_sampler,
            num_workers=args.num_workers,
            drop_last=True,
            persistent_workers=True,
            prefetch_factor=16
        )

        val_loader = pl.MpDeviceLoader(
            val_loader, 
            device,
            loader_prefetch_size=128,
            device_prefetch_size=1,
            host_to_device_transfer_threads=4
        )
    else:
        val_loader = None

    
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset,
        num_replicas=xr.world_size(),
        rank=xr.global_ordinal(),
        shuffle=False
    )
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=args.num_workers,
        drop_last=True,
        persistent_workers=True,
        prefetch_factor=16
    )
    
    # TPU에 최적화된 데이터로더
    train_loader = pl.MpDeviceLoader(
        train_loader, 
        device,
        loader_prefetch_size=128,
        device_prefetch_size=1,
        host_to_device_transfer_threads=4
    )
    
    # 모델 설정
    model = KoBARTSummaryModel(tokenizer)
    model.to(device)

    xm.broadcast_master_param(model)
    
    # DDP 모델 설정
    model = torch.nn.parallel.DistributedDataParallel(
        model,
        broadcast_buffers=False,
        gradient_as_bucket_view=True
    )
    
    # 옵티마이저 설정
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.lr * xr.world_size())



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
    
    if args.resume_from_checkpoint:
        # 마지막 체크포인트 로드 checkpoint_{epoch}.pt
        files = os.listdir(args.checkpoint)
        checkpoint_files = [f for f in files if f.startswith('checkpoint_') and f.endswith('.pt')]
        checkpoint_files.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))
        model_path = checkpoint_files[-1] if checkpoint_files else None

        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location='cpu')
            
            if is_master:
                logger.info(f"Loading model from {model_path}")
            
            if hasattr(model, "module"):
                state_dict = model.module.state_dict()
            else:
                state_dict = model.state_dict()
            check_model_state = checkpoint['model']
            new_state_dict = {}

            for k, v in state_dict.items():
                try:
                    new_state_dict[k] = check_model_state[k]
                    assert v.shape == check_model_state[k].shape, (
                        check_model_state[k].shape,
                        v.shape
                    )
                except:
                    print(f"{k} is not int the checkpoint model")
                    new_state_dict[k] = v
            if hasattr(model, "module"):
                model.module.load_state_dict(new_state_dict)
            else:
                model.load_state_dict(new_state_dict)
            
            # 옵티마이저, 스케줄러 상태 로드
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            
            start_epoch = checkpoint['epoch']
            global_step = checkpoint['step']
            
            if is_master:
                logger.info(f"Resuming from epoch {start_epoch}, step {global_step}")
    
    def _log_summary(epoch, step, total_steps, avg_loss, elapsed):
        print(f"Epoch: {epoch}, Step: {step}/{total_steps}, Loss: {avg_loss:.4f}, Time: {elapsed:.2f}s", flush=True)

    total_steps = len(train_loader) # 로깅을 위해 배치당 스텝 수 계산
    # 학습 루프
    for epoch in range(start_epoch, args.max_epochs):
        epoch_loss = 0
        epoch_steps = 0
        # 에폭 시작 시간
        start_time = time.time()
        
        for step, batch in enumerate(train_loader):
            with torch_xla.step():
                # 그래디언트 초기화
                optimizer.zero_grad()
                
                # 학습 스텝
                loss = train_step(model.module, batch, optimizer, device)

                # 그래디언트 클리핑
                if args.gradient_clip_val > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip_val)
                
                optimizer.step()
                scheduler.step()

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
                        args=(epoch, step, total_steps, avg_loss, elapsed),
                        run_async=True
                    )

        if is_local_master:
            save_checkpoint(model, tokenizer, optimizer, scheduler, epoch + 1, global_step, args)


    if is_local_master:
        logger.info("Training completed")


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
    torch_xla.launch(_mp_fn, args=(args,))