# https://github.com/huggingface/transformers/issues/6821#issuecomment-1175736525
# https://github.com/HFAiLab/pytorch_distributed/blob/main/resnet_ddp_apex.py
from __future__ import absolute_import, division, print_function

import hf_env
hf_env.set_env('202111')

# import torch.multiprocessing as mp
# import torch.distributed as dist

import hfai
import hfai.nccl.distributed as dist
from hfai.nn.parallel import DistributedDataParallel

import argparse
import torch

import argparse
import glob
import logging
import os
import time
from pathlib import Path
import pickle
import random
import re
import shutil
import json

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler,TensorDataset
from torch.utils.data.distributed import DistributedSampler
from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR
from dataset_wo_err import editDataset_wo_err
from beam import Beam

try:
    from torch.utils.tensorboard import SummaryWriter
except:
    from tensorboardX import SummaryWriter

from torch.nn import CrossEntropyLoss

from bleu import _bleu
from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup,
                          BertConfig, BertForMaskedLM, BertTokenizer,
                          GPT2Config, GPT2LMHeadModel, GPT2Tokenizer,
                          OpenAIGPTConfig, OpenAIGPTLMHeadModel, OpenAIGPTTokenizer,
                          RobertaConfig, RobertaForMaskedLM, RobertaTokenizer,
                          DistilBertConfig, DistilBertForMaskedLM, DistilBertTokenizer,
                          AutoConfig, AutoModelForCausalLM, AutoTokenizer)
from codegen_tokenizer import CodeGenTokenizer
from codegen_model import CodeGenForCausalLM
from codegen_config import CodeGenConfig

from xglm_tokenizer import XGLMTokenizer
from xglm_model import XGLMForCausalLM
from xglm_config import XGLMConfig

from collate_utils import DataCollatePad
from tqdm import tqdm
import multiprocessing

logger = logging.getLogger(__name__)

MODEL_CLASSES = {
    'gpt2': (GPT2Config, GPT2LMHeadModel, GPT2Tokenizer),
    'openai-gpt': (OpenAIGPTConfig, OpenAIGPTLMHeadModel, OpenAIGPTTokenizer),
    'bert': (BertConfig, BertForMaskedLM, BertTokenizer),
    'roberta': (RobertaConfig, RobertaForMaskedLM, RobertaTokenizer),
    'distilbert': (DistilBertConfig, DistilBertForMaskedLM, DistilBertTokenizer),
    'auto': (AutoConfig, AutoModelForCausalLM, AutoTokenizer),
    'codegen': (CodeGenConfig, CodeGenForCausalLM, CodeGenTokenizer),
    'xglm': (XGLMConfig, XGLMForCausalLM, AutoTokenizer),
}




parser = argparse.ArgumentParser()

parser.add_argument("--config_file", default=None, type=str, required=True,
                    help="The config json file for train.")

args = parser.parse_args()
with open(args.config_file, 'r') as f:
    add_args = json.load(f)
for k, v in add_args.items():
    setattr(args, k, v)
if not hasattr(args, "with_id"):
    args.with_id = False

args.output_dir += "without_err_msg"

config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
# parser = argparse.ArgumentParser()

# parser.add_argument('--nodes', type=int, default=1)  # how many nodes (machines) you have
# parser.add_argument('--gpus', type=int, default=-1, help='num gpus per node')
# parser.add_argument('--nr', type=int, default=0, help='ranking within the nodes')
# args = parser.parse_args()
pretrained = args.pretrain_dir

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
def update_config(model, tokenizer):
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
# def load_and_cache_examples(args, tokenizer, evaluate=False):
#     if args.use_pool:
#         cpu_count = multiprocessing.cpu_count()
#         pool = multiprocessing.Pool(cpu_count)
#     else:
#         pool = None
#     dataset = editDataset_wo_err(tokenizer, args, logger, file_type='dev' if evaluate else 'train',
#                           block_size=args.block_size, pool=None)
#     return dataset

set_seed(args)

def save_model(output_dir, model, tokenizer, optimizer, scheduler, acc = None, epoch = None, step = None):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    pretrain_dir = output_dir
    scheduler_path = os.path.join(output_dir, 'scheduler.pt')
    optimizer_path = os.path.join(output_dir, 'optimizer.pt')
    state_dict_path = os.path.join(output_dir, 'state_dict.pt')
    if optimizer is not None:
        torch.save(optimizer.state_dict(), optimizer_path)
    if scheduler is not None:
        torch.save(scheduler.state_dict(), scheduler_path)
    if model is not None:
        model_to_save = (
            model.module if hasattr(model, "module") else model
        )  # Take care of distributed/parallel training
        model_to_save.save_pretrained(pretrain_dir)
    if tokenizer is not None:
        tokenizer.save_pretrained(pretrain_dir)
    
    state = {
        'epoch': epoch,
        'step': step,
        'acc': acc,
    }
    torch.save(state, state_dict_path)

def load_model(output_dir):
    pretrain_dir = output_dir
    scheduler_path = os.path.join(output_dir, 'scheduler.pt')
    optimizer_path = os.path.join(output_dir, 'optimizer.pt')
    state_dict_path = os.path.join(output_dir, 'state_dict.pt')
    if os.path.exists(state_dict_path):
        state = torch.load(state_dict_path)
        epoch = state['epoch']
        step = state['step']
        acc = state['acc']
    else:
        epoch = None
        step = None
        acc = None

    if os.path.exists(optimizer_path):
        optimizer_path = os.path.join(output_dir, 'optimizer.pt')
    else:
        optimizer_path = None
    if os.path.exists(scheduler_path):
        scheduler_path = os.path.join(output_dir, 'scheduler.pt')
    else:
        scheduler_path = None
    
    if os.path.exists(output_dir) and os.listdir(output_dir):
        pretrain_dir = output_dir
    else:
        pretrain_dir = None
    
    return dict(pretrain_dir=pretrain_dir, optimizer_path=optimizer_path, scheduler_path=scheduler_path, epoch=epoch, step=step, acc=acc)

    

def train(dataloader, model, optimizer, scheduler, loss_scaler, epoch, local_rank, start_step, best_acc, save_path, args):
    model.train()
    for step, train_data in enumerate(dataloader):
        if step < start_step:
            continue

        if args.with_id:
            batch, token_labels, ids = train_data
            ids = ids.to(args.device, non_blocking=True)
        else:
            batch, token_labels = train_data
        if step == start_step: # logging
            if args.with_id:
                logger.info("rank: %s, batch: %s, token_labels: %s, ids: %s", args.local_rank, batch.shape, token_labels.shape, ids.shape)
            else:
                logger.info("rank: %s, batch: %s, token_labels: %s", args.local_rank, batch.shape, token_labels.shape)

        inputs = batch.cuda(non_blocking=True)
        token_labels = token_labels.cuda(non_blocking=True)
        attn_mask = torch.tensor(token_labels.clone().detach() != 0, dtype=torch.uint8).to(args.device, non_blocking=True)
        loss_mask = torch.tensor(token_labels.clone().detach() == 2, dtype=torch.uint8).to(args.device, non_blocking=True)

        with torch.cuda.amp.autocast():
            outputs = model(inputs, attention_mask=attn_mask)
            logits = outputs[0]
            labels = inputs
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss(reduction="none")
            flatten_shift_loss_mask = loss_mask[..., :-1].contiguous().view(-1)
            ids = torch.nonzero(flatten_shift_loss_mask).view(-1)

            p_shift_logits_softmax = shift_logits.view(-1, shift_logits.size(-1)).clone().detach()
            p_shift_logits_softmax = torch.softmax(p_shift_logits_softmax, dim=-1)
            p_shift_logits_softmax_on_labels = torch.gather(p_shift_logits_softmax, 1, shift_labels.view(-1, 1))
            p_shift_logits_softmax_on_labels = p_shift_logits_softmax_on_labels[ids].view(-1)

            
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1))[ids], shift_labels.view(-1)[ids])
            loss = loss * p_shift_logits_softmax_on_labels
            loss = loss.mean()

        loss_scaler.scale(loss).backward()
        
        # TODO: add gradient clip
        # torch.nn.utils.clip_grad_norm_(torch.cuda.amp.master_params(optimizer), args.max_grad_norm)


        loss_scaler.step(optimizer)
        loss_scaler.update()
        optimizer.zero_grad()

        torch.cuda.synchronize()
        torch.cuda.empty_cache()

        if step % args.logging_steps == 0:
            logger.info("rank: %s, step: %s, len(dataloader): %s, loss: %s", args.local_rank, step, len(dataloader), loss.item())

        # 保存
        if dist.get_rank() == 0 and local_rank == 0 and hfai.client.receive_suspend_command():
            checkpoint_last = os.path.join(args.output_dir, 'checkpoint-last')
            save_model(checkpoint_last, model, tokenizer, optimizer, scheduler, best_acc, epoch, step)
            time.sleep(5)
            hfai.client.go_suspend()


def main(local_rank, args):
    # 超参数设置
    epochs = int(args.num_train_epochs)
    batch_size = int(args.per_gpu_train_batch_size)
    num_workers = args.num_workers
    lr = args.learning_rate
    eps = args.adam_epsilon
    warmup_steps = args.warmup_steps
    weight_decay = 1e-4
    save_path = Path(args.output_dir)
    save_path.mkdir(exist_ok=True, parents=True)

    # 多机通信
    ip = os.environ['MASTER_IP']
    port = os.environ['MASTER_PORT']
    hosts = int(os.environ['WORLD_SIZE'])  # 机器个数
    rank = int(os.environ['RANK'])  # 当前机器编号
    gpus = torch.cuda.device_count()  # 每台机器的GPU个数

    
    tokenizer = tokenizer_class.from_pretrained(args.pretrain_dir, do_lower_case=args.do_lower_case, bos_token='<s>', eos_token='</s>', pad_token='<pad>', unk_token='<|UNKNOWN|>', sep_token='concode_elem_sep')
    special_tokens_dict = {'additional_special_tokens': ['<question>','<code>','<error>']}
    tokenizer.add_special_tokens(special_tokens_dict)
    # world_size是全局GPU个数，rank是当前GPU全局编号
    dist.init_process_group(backend='nccl', init_method="tcp://" + str(ip) + ":" + str(port), world_size=hosts * gpus, rank=rank * gpus + local_rank)
    torch.cuda.set_device(local_rank)
    args.device = torch.device("cuda", local_rank)
    args.world_size = hosts * gpus
    args.local_rank = local_rank
    world_size = args.world_size

    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO if local_rank in [-1, 0] else logging.WARN)
    logger.info(tokenizer.encode("<s> hello world <pad> </s>"))
    logger.info("args: %s", args)

    # 数据
    # train_dataset = load_and_cache_examples(args, tokenizer, evaluate=False)
    if args.use_pool:
        cpu_count = multiprocessing.cpu_count()
        pool = multiprocessing.Pool(cpu_count)
    else:
        pool = None
    train_dataset = editDataset_wo_err(tokenizer, args, logger, 'train', block_size=args.block_size, pool=pool)

    # train_datasampler = DistributedSampler(train_dataset)
    train_datasampler = RandomSampler(train_dataset)
    collate_fn = DataCollatePad(pad_ids=[tokenizer.pad_token_id, 0], without_pad_idxs=[2])
    train_dataloader = DataLoader(train_dataset,batch_size=args.per_gpu_train_batch_size, collate_fn=collate_fn , sampler=train_datasampler, num_workers=num_workers, pin_memory=True)
    

    total_examples = len(train_dataset) * world_size
    args.batch_size = args.per_gpu_train_batch_size * world_size
    batch_size = args.batch_size
    if args.num_train_epochs > 0:
        t_total = int(total_examples // batch_size * int(args.num_train_epochs))

    # 模型、优化器
    # 加载历史
    checkpoint_last = os.path.join(args.output_dir, 'checkpoint-last')
    load_dict = load_model(checkpoint_last)
    pretrained_dir, optimizer_path, scheduler_path, epoch, step, acc = load_dict['pretrain_dir'], load_dict['optimizer_path'], load_dict['scheduler_path'], load_dict['epoch'], load_dict['step'], load_dict['acc']
    logger.info("load history dict: {}".format(str(load_dict)))

    if pretrained_dir is not None:
        pretrained = pretrained_dir
    else:
        pretrained = args.pretrain_dir
    logger.info("pretrained from {}".format(pretrained))
    model = model_class.from_pretrained(pretrained)
    model.resize_token_embeddings(len(tokenizer))
    update_config(model, tokenizer)
    model = model.cuda()
    logger.info("model config: {}".format(model.config))
    model = DistributedDataParallel(model)
    
    loss_scaler = torch.cuda.amp.GradScaler(enabled=True) # 自动精度
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    if optimizer_path is not None:
        optimizer.load_state_dict(torch.load(optimizer_path, map_location="cpu"))
        logger.info("load optimizer from {}".format(optimizer_path))
    
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,
                                                num_training_steps=t_total)
    if scheduler_path is not None:
        scheduler.load_state_dict(torch.load(scheduler_path, map_location="cpu"))
        logger.info("scheduler loaded from {}".format(scheduler_path))
    
    # 加载
    best_acc, start_epoch, start_step = 0, 0, 0
    if epoch is not None and epoch >= 0 and step is not None and step >= 0:
        start_epoch = epoch
        start_step = step
        # best_acc = acc
        logger.info("load model from checkpoint-last, epoch: {}, step: {}".format(epoch, step))

    # 训练、验证
    for epoch in range(start_epoch, epochs):
        t1 = time.time()
        # train_datasampler.set_epoch(epoch)
        train(train_dataloader, model, optimizer, scheduler, loss_scaler, epoch, local_rank, start_step, best_acc, save_path, args)
        start_step = 0 
        scheduler.step()
        # acc = validate(val_dataloader, model, criterion, epoch, local_rank)
        acc = 0
        t2 = time.time()

        # 保存
        if rank == 0 and local_rank == 0:
            print("cost time per epoch: {:.4f} s".format(t2-t1))
            checkpoint_last = os.path.join(args.output_dir, 'checkpoint-last')
            save_model(checkpoint_last, model, tokenizer, optimizer, scheduler, 0, epoch, 0)
            checkpoint_prefix = "checkpoint" + "-epoch" + str(epoch)
            checkpoint_prefix = os.path.join(args.output_dir, checkpoint_prefix)
            save_model(checkpoint_prefix, model, tokenizer, optimizer, scheduler, 0, epoch, 0)
            # if acc > best_acc:
            #     best_acc = acc
            #     print(f'New Best Acc: {100*acc:.2f}%!')
            #     torch.save(model.module.state_dict(), save_path / 'best.pt')


        
if __name__ == "__main__":
    ngpus = torch.cuda.device_count()
    hfai.multiprocessing.spawn(main, args=(args, ), nprocs=ngpus, bind_numa=True)