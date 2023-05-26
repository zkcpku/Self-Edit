# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Text to code generation pipeline in CodeXGLUE
"""

from __future__ import absolute_import, division, print_function

import hf_env
hf_env.set_env('202111') 

import argparse
import glob
import logging
import os
import pickle
import random
import re
import shutil
import json
import multiprocessing
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler,TensorDataset
from torch.utils.data.distributed import DistributedSampler
from dataset import editDataset
from beam import Beam
import hfai


# try:
#     from torch.utils.tensorboard import SummaryWriter
# except:
#     from tensorboardX import SummaryWriter

from torch.nn import CrossEntropyLoss

from bleu import _bleu
from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup,
                          BertConfig, BertForMaskedLM, BertTokenizer,
                          GPT2Config, GPT2LMHeadModel, GPT2Tokenizer,
                          OpenAIGPTConfig, OpenAIGPTLMHeadModel, OpenAIGPTTokenizer,
                          RobertaConfig, RobertaForMaskedLM, RobertaTokenizer,
                          DistilBertConfig, DistilBertForMaskedLM, DistilBertTokenizer,
                          AutoConfig, AutoModelForCausalLM, AutoTokenizer)

logger = logging.getLogger(__name__)

MODEL_CLASSES = {
    'gpt2': (GPT2Config, GPT2LMHeadModel, GPT2Tokenizer),
    'openai-gpt': (OpenAIGPTConfig, OpenAIGPTLMHeadModel, OpenAIGPTTokenizer),
    'bert': (BertConfig, BertForMaskedLM, BertTokenizer),
    'roberta': (RobertaConfig, RobertaForMaskedLM, RobertaTokenizer),
    'distilbert': (DistilBertConfig, DistilBertForMaskedLM, DistilBertTokenizer),
    'auto': (AutoConfig, AutoModelForCausalLM, AutoTokenizer)
}



def load_and_cache_examples(args, tokenizer, evaluate=False):
    if args.use_pool:
        cpu_count = multiprocessing.cpu_count()
        pool = multiprocessing.Pool(cpu_count)
    else:
        pool = None
    dataset = editDataset(tokenizer, args, logger, file_type='dev' if evaluate else 'train',
                          block_size=args.block_size, pool=pool)
    return dataset


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def update_config(model, tokenizer):
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id


def train(args, train_dataset, model, tokenizer, pool):
    """ Train the model """
    if args.local_rank in [-1, 0]:
        args.tensorboard_dir = os.path.join(args.output_dir, 'tensorboard')
        if not os.path.exists(args.tensorboard_dir):
            os.makedirs(args.tensorboard_dir)
        # tb_writer = SummaryWriter(args.tensorboard_dir)
    
    args.batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset)
    
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.batch_size, drop_last=True)
    total_examples = len(train_dataset) * (
                    torch.distributed.get_world_size() if args.local_rank != -1 else 1)
    batch_size = args.batch_size * args.gradient_accumulation_steps * (
                    torch.distributed.get_world_size() if args.local_rank != -1 else 1)
    # if args.max_steps > 0:
    #     t_total = args.max_steps
    #     args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    if args.num_train_epochs > 0:
        t_total = total_examples // batch_size * args.num_train_epochs
    args.max_steps = t_total
    model.to(args.device)
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  
    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,
                                                num_training_steps=t_total)
    checkpoint_last = os.path.join(args.output_dir, 'checkpoint-last')
    scheduler_last = os.path.join(checkpoint_last, 'scheduler.pt')
    optimizer_last = os.path.join(checkpoint_last, 'optimizer.pt')
    if os.path.exists(scheduler_last):
        scheduler.load_state_dict(torch.load(scheduler_last, map_location="cpu"))
    if os.path.exists(optimizer_last):
        optimizer.load_state_dict(torch.load(optimizer_last, map_location="cpu"))   
    if args.local_rank == 0:
        torch.distributed.barrier()   
    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank%args.gpu_per_node],
                                                          output_device=args.local_rank%args.gpu_per_node,
                                                          find_unused_parameters=True)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", total_examples )
    logger.info("  Num epoch = %d", t_total*batch_size//total_examples)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d", batch_size)
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)
    
    global_step = args.start_step
    tr_loss, logging_loss,avg_loss,tr_nb = 0.0, 0.0,0.0,0
    # model.resize_token_embeddings(len(tokenizer))
    model.zero_grad()
    set_seed(args)  # Added here for reproducibility (even between python 2 and 3)

    best_bleu = 0.0
 
    for idx in range(args.start_epoch, int(args.num_train_epochs)): 
        for step, (batch, token_labels) in enumerate(train_dataloader):
            inputs = batch.to(args.device)
            attn_mask = torch.tensor(token_labels.clone().detach() != 0, dtype=torch.uint8, device=args.device)
            loss_mask = torch.tensor(token_labels.clone().detach() == 2, dtype=torch.uint8, device=args.device)
            model.train()
            # outputs = model(inputs, attention_mask=attn_mask, labels=inputs, loss_mask=loss_mask)
            # loss = outputs[0]
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

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            tr_loss += loss.item()
                
            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()  
                global_step += 1
                output_flag=True
                avg_loss=round(np.exp((tr_loss - logging_loss) /(global_step- tr_nb)),4)
                if global_step % args.logging_steps == 0:
                    logger.info("  steps: %s  ppl: %s", global_step, round(avg_loss,5))
                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    # Log metrics
                    # tb_writer.add_scalar('lr', scheduler.get_last_lr()[0], global_step)
                    # tb_writer.add_scalar('loss', (tr_loss - logging_loss) / args.logging_steps, global_step)
                    logger.info("lr: %s at %s, loss: %s at %s", scheduler.get_last_lr()[0], global_step, (tr_loss - logging_loss) / args.logging_steps, global_step)
                    logging_loss = tr_loss
                    tr_nb=global_step
                # TODO
                # if hfai.distributed.get_rank() == 0 and gpu_id == 0 and hfai.client.receive_suspend_command():
                #     last_output_dir = os.path.join(args.output_dir, 'checkpoint-last')
                #     if not os.path.exists(last_output_dir):
                #         os.makedirs(last_output_dir)
                #     model_to_save.save_pretrained(last_output_dir)
                #     tokenizer.save_pretrained(last_output_dir)
                #     idx_file = os.path.join(last_output_dir, 'idx_file.txt')
                #     with open(idx_file, 'w', encoding='utf-8') as idxf:
                #         idxf.write(str(0) + '\n')

                #     torch.save(optimizer.state_dict(), os.path.join(last_output_dir, "optimizer.pt"))
                #     torch.save(scheduler.state_dict(), os.path.join(last_output_dir, "scheduler.pt"))
                #     logger.info("Saving optimizer and scheduler states to %s", last_output_dir)

                #     step_file = os.path.join(last_output_dir, 'step_file.txt')
                #     with open(step_file, 'w', encoding='utf-8') as stepf:
                #         stepf.write(str(global_step) + '\n')
                #     hfai.client.go_suspend()

                # if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
        checkpoint_prefix = "checkpoint" + "-epoch" + str(idx)
        # Save model checkpoint
        if args.evaluate_during_training:  # Only evaluate when single GPU otherwise metrics may not average well
            test_bleu, test_EM = eval_and_save(args, model.module if hasattr(model, "module") else model, tokenizer, file_type='test_no_code', save_prefix="e" + str(idx))
            # logger.info(f"test bleu: {test_bleu}, test EM: {test_EM}")
            test_bleu, test_EM = eval_and_save(args, model.module if hasattr(model, "module") else model, tokenizer, file_type='train_no_code', save_prefix="e" + str(idx))
            # logger.info(f"test bleu: {test_bleu}, test EM: {test_EM}")
            # test_bleu, test_EM = eval_and_save(args, model, tokenizer, file_type='train', save_prefix="e" + str(idx))
            # logger.info(f"test bleu: {test_bleu}, test EM: {test_EM}")

        output_dir = os.path.join(args.output_dir, "{}-{}".format(checkpoint_prefix, global_step))
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        model_to_save = (
            model.module if hasattr(model, "module") else model
        )  # Take care of distributed/parallel training
        model_to_save.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)

        torch.save(args, os.path.join(output_dir, "training_args.bin"))
        logger.info("Saving model checkpoint to %s", output_dir)

        # _rotate_checkpoints(args, checkpoint_prefix)
        last_output_dir = os.path.join(args.output_dir, 'checkpoint-last')
        if not os.path.exists(last_output_dir):
            os.makedirs(last_output_dir)
        model_to_save.save_pretrained(last_output_dir)
        tokenizer.save_pretrained(last_output_dir)
        idx_file = os.path.join(last_output_dir, 'idx_file.txt')
        with open(idx_file, 'w', encoding='utf-8') as idxf:
            idxf.write(str(idx) + '\n')

        torch.save(optimizer.state_dict(), os.path.join(last_output_dir, "optimizer.pt"))
        torch.save(scheduler.state_dict(), os.path.join(last_output_dir, "scheduler.pt"))
        logger.info("Saving optimizer and scheduler states to %s", last_output_dir)

        step_file = os.path.join(last_output_dir, 'step_file.txt')
        with open(step_file, 'w', encoding='utf-8') as stepf:
            stepf.write(str(global_step) + '\n')

                    # torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                    # torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                    # logger.info("Saving optimizer and scheduler states to %s", output_dir)
                    

        #     if args.max_steps > 0 and global_step > args.max_steps:
        #         break
        # if args.max_steps > 0 and global_step > args.max_steps:
        #     break

    if args.local_rank in [-1, 0]:
        # tb_writer.close()
        pass

    return global_step, tr_loss / global_step


def eval_and_save(args, model, tokenizer, file_type='test', save_prefix = 'last',num=20000000):
    if args.use_pool:
        cpu_count = multiprocessing.cpu_count()
        pool = multiprocessing.Pool(cpu_count)
    else:
        pool = None
    dataset = editDataset(tokenizer, args, logger, file_type=file_type, block_size=args.block_size, mode='test',pool=pool)
    test_sampler = SequentialSampler(dataset)
    test_dataloader = DataLoader(dataset, sampler=test_sampler, batch_size=1)
    model.to(args.device)
    model.zero_grad()
    model.eval()

    preds = []
    golds = []
    max_gen_len = 100
    for step, (batch, token_labels) in enumerate(test_dataloader):
        if step >= num:
            break
        inputs = batch.to(args.device)
        with torch.no_grad():
            if args.sample_type == "greedy":
                outputs = model.generate(inputs, max_length=args.block_size, do_sample = False, num_beams=1, temperature=0.7, early_stopping=False, top_k=70, \
                      bos_token_id=tokenizer.bos_token_id, eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id)
                generation = tokenizer.decode(outputs[0])[len(tokenizer.decode(inputs[0])):]
                this_input = tokenizer.decode(inputs[0])
                golds.append(this_input)
                preds.append(generation.rstrip("<pad>"))
            elif "top_p" in args.sample_type:
                # sample_type = "top_p:200"
                sample_num = int(args.sample_type.split(":")[-1])
                outputs = model.generate(inputs, max_length=args.block_size, do_sample=True, temperature=0.7, top_k=70, top_p=0.95, num_return_sequences=sample_num, bos_token_id=tokenizer.bos_token_id, eos_token_id=tokenizer.pad_token_id, pad_token_id=tokenizer.pad_token_id)

                generation = tokenizer.batch_decode(outputs)
                this_input = tokenizer.batch_decode(inputs)
                golds.append(this_input)
                encode_len = len(tokenizer.decode(inputs[0]))
                generation = [e[encode_len:].rstrip("<pad>") for e in generation]
                preds.append(generation)
            else:
                assert False, "sample type not supported"
            
        #     # outputs = model.generate(inputs, max_length=args.block_size, num_beams=10, temperature=0.7, early_stopping=False, top_k=70)
        #     # outputs = model.generate(inputs, max_length=args.block_size, do_sample=True, temperature=0.7, top_k=70, top_p=0.95)
            
        
        # with torch.no_grad():
        #     beam_size = 1
        #     m = torch.nn.LogSoftmax(dim=-1)
        #     outputs = model(inputs)[1]
        #     p = []       
        #     zero = torch.cuda.LongTensor(1).fill_(0)
        #     for i in range(inputs.shape[0]):
        #         # Compatible with transformers version 3.3.0 and 4.13.0
        #         past = [torch.cat([x[0].unsqueeze(0),x[1].unsqueeze(0)],dim=0) if type(x)==tuple else x for x in outputs]
        #         past_hidden = [x[:, i:i+1].expand(-1, beam_size, -1, -1, -1) for x in past]
        #         # context_mask=source_mask[i:i+1,:].expand(beam_size,-1)
        #         beam = Beam(beam_size, tokenizer.bos_token_id, tokenizer.eos_token_id)
        #         input_ids = None
        #         for _ in range(max_gen_len): 
        #             if beam.done():
        #                 break
        #             input_ids = beam.getCurrentState()    
        #             # context_mask=torch.cat((context_mask,input_ids*0+1),-1)
        #             # mask=context_mask.unsqueeze(0).unsqueeze(-2).unsqueeze(-2).expand(self.config.n_layer, -1, -1, -1, -1)
        #             transformer_outputs = model(input_ids, past_key_values=past_hidden)
        #             out = m(transformer_outputs[0][:, -1, :]).data
        #             # out = self.lsm(self.lm_head(transformer_outputs[0][:,-1,:])).data
        #             beam.advance(out)
        #             past = [torch.cat([x[0].unsqueeze(0),x[1].unsqueeze(0)],dim=0) if type(x)==tuple else x for x in transformer_outputs[1]]
        #             past_hidden = [x.data.index_select(1, beam.getCurrentOrigin()) for x in past]
        #         hyp = beam.getHyp(beam.getFinal())
        #         pred  =beam.buildTargetTokens(hyp)[:beam_size]

        #         pred = [torch.cat([x.view(-1) for x in p]+[zero]*(max_gen_len-len(p))).view(1,-1) for p in pred]
        #         p.append(torch.cat(pred, 0).unsqueeze(0))
        #     p = torch.cat(p, 0)
        #     for pred in p:
        #         t = pred[0].cpu().numpy()
        #         t = list(t)
        #         if 0 in t:
        #             t = t[:t.index(0)]
        #         text = tokenizer.decode(t, clean_up_tokenization_spaces=False)
        #         # print(text)
        #         preds.append(text)
        
        if step % args.logging_steps == 0:
            logger.info(f"{step} are done!")
    
    # golds = []
    # datafile = os.path.join(args.data_dir, f"{file_type}.json")
    # datas = open(datafile).readlines()
    # for x in datas[:num]:
    #     x = json.loads(x)
    #     golds.append(str(x["nl"]))
    
    assert len(preds) == len(golds)

    with open(os.path.join(args.output_dir, f"{file_type}-"+save_prefix+".skeleton"), 'w') as f:
        json.dump(preds, f)
    with open(os.path.join(args.output_dir, f"{file_type}-"+save_prefix+".src"), 'w') as f1:
        json.dump(golds, f1)

    # EM = []
    # with open(os.path.join(args.output_dir, f"{file_type}-"+save_prefix+".skeleton"), 'w') as f, open(os.path.join(args.output_dir, f"{file_type}-"+save_prefix+".src"), 'w') as f1:
    #     for pred, gold in zip(preds, golds):
    #         f.write(pred+'\n')
    #         f1.write(gold+'\n')

    return 0,0


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--config_file", default=None, type=str, required=True,
                        help="The config json file for train.")
    
    pool = None
    args = parser.parse_args()
    with open(args.config_file, 'r') as f:
        add_args = json.load(f)
    for k, v in add_args.items():
        setattr(args, k, v)

    # args.output_dir = os.path.join(args.output_dir, args.dataset)

    if args.model_type in ["bert", "roberta", "distilbert"] and not args.mlm:
        raise ValueError("BERT and RoBERTa do not have LM heads but masked LM heads. They must be run using the --mlm "
                         "flag (masked language modeling).")

    # if os.path.exists(args.output_dir) and os.listdir(
    #         args.output_dir) and args.do_train and not args.overwrite_output_dir:
    #     raise ValueError(
    #         "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
    #             args.output_dir))

    # Setup distant debugging if needed
    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd
        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    logger.warning("local_rank: %d, node_index: %d, gpu_per_node: %d"%(args.local_rank, args.node_index, args.gpu_per_node))
    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.local_rank += args.node_index * args.gpu_per_node
        args.n_gpu = 1
    args.device = device
    # args.batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)

    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s, world size: %s",
                   args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), args.fp16,
                   torch.distributed.get_world_size() if args.local_rank != -1 else 1)
    logger.info("Training/evaluation parameters %s", args)

    # 使用FileHandler输出到文件
    # fh = logging.FileHandler(args.log_file)
    # logger.addHandler(fh)

    # Set seed
    set_seed(args)

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Barrier to make sure only the first process in distributed training download model & vocab

    args.start_epoch = 0
    args.start_step = 0
    checkpoint_last = os.path.join(args.output_dir, 'checkpoint-last')
    if args.do_train and os.path.exists(checkpoint_last) and os.listdir(checkpoint_last):
        args.pretrain_dir = os.path.join(checkpoint_last)
        args.config_name = os.path.join(checkpoint_last, 'config.json')
        idx_file = os.path.join(checkpoint_last, 'idx_file.txt')
        with open(idx_file, encoding='utf-8') as idxf:
            args.start_epoch = int(idxf.readlines()[0].strip()) + 1

        step_file = os.path.join(checkpoint_last, 'step_file.txt')
        if os.path.exists(step_file):
            with open(step_file, encoding='utf-8') as stepf:
                args.start_step = int(stepf.readlines()[-1].strip())

        logger.info("[Attention!] reload model from {}, resume from {} epoch".format(checkpoint_last, args.start_epoch))

    # Load pre-trained model
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    pretrained = args.pretrain_dir
    if pretrained:
        print(pretrained)
        tokenizer = tokenizer_class.from_pretrained(pretrained, do_lower_case=args.do_lower_case, bos_token='<s>', eos_token='</s>', pad_token='<pad>', unk_token='<|UNKNOWN|>', sep_token='concode_elem_sep')
        special_tokens_dict = {'additional_special_tokens': ['<question>','<code>','<error>']}
        tokenizer.add_special_tokens(special_tokens_dict)
        logger.info(tokenizer.encode("<question> hello world <code>hello<error>world </s>"))
        model = model_class.from_pretrained(pretrained)
        model.resize_token_embeddings(len(tokenizer))
        update_config(model, tokenizer)
        logger.info(model.config)
    else:
        tokenizer = tokenizer_class.from_pretrained(args.tokenizer_dir, bos_token='<s>', eos_token='</s>', pad_token='<pad>', unk_token='<|UNKNOWN|>', sep_token='concode_elem_sep')
        special_tokens_dict = {'additional_special_tokens': ['<question>','<code>','<error>']}
        tokenizer.add_special_tokens(special_tokens_dict)
        args.vocab_size = tokenizer.vocab_size
        config = config_class.from_pretrained(args.config_dir)
        model = model_class(config)
        model.resize_token_embeddings(len(tokenizer))
        update_config(model, tokenizer)

    model_parameters = model.parameters()
    num_params = sum([np.prod(p.size()) for p in model_parameters])
    logger.info(f"Model has a total of {num_params} trainable parameters")

    if args.local_rank == 0:
        torch.distributed.barrier()  # End of barrier to make sure only the first process in distributed training download model & vocab

    logger.info("Training/evaluation parameters %s", args)

    # Training
    if args.do_train:
        train_dataset = load_and_cache_examples(args, tokenizer, evaluate=False)

        global_step, tr_loss = train(args, train_dataset, model, tokenizer, pool)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

    if args.do_eval:            # only works on 1 GPU
        dev_bleu, dev_EM = eval_and_save(args, model, tokenizer, file_type='test_no_code')
        logger.info(f"dev bleu: {dev_bleu}, dev EM: {dev_EM}")

    if args.do_infer:            # only works on 1 GPU
        test_bleu, test_EM = eval_and_save(args, model, tokenizer, file_type='test_no_code', save_prefix=args.infer_prefix)
        # logger.info(f"test bleu: {test_bleu}, test EM: {test_EM}")
        # test_bleu, test_EM = eval_and_save(args, model, tokenizer, file_type='train_no_code', save_prefix=args.infer_prefix)
        # logger.info(f"test bleu: {test_bleu}, test EM: {test_EM}")


if __name__ == "__main__":
    main()
