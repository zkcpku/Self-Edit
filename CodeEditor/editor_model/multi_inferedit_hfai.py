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
import pickle
import random
import re
import shutil
import json

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler,TensorDataset
from torch.utils.data.distributed import DistributedSampler
from dataset import editDataset
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

import traceback


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




logger = logging.getLogger(__name__)



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


config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
# parser = argparse.ArgumentParser()

# parser.add_argument('--nodes', type=int, default=1)  # how many nodes (machines) you have
# parser.add_argument('--gpus', type=int, default=-1, help='num gpus per node')
# parser.add_argument('--nr', type=int, default=0, help='ranking within the nodes')
# args = parser.parse_args()
pretrained = args.pretrain_dir

tokenizer = tokenizer_class.from_pretrained(pretrained, do_lower_case=args.do_lower_case, bos_token='<s>', eos_token='</s>', pad_token='<pad>', unk_token='<|UNKNOWN|>', sep_token='concode_elem_sep')
special_tokens_dict = {'additional_special_tokens': ['<question>','<code>','<error>']}
tokenizer.add_special_tokens(special_tokens_dict)

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
def update_config(model, tokenizer):
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id

set_seed(args)


def test_model_generation(local_gpu_rank, args):

    ip = os.environ['MASTER_IP']
    port = os.environ['MASTER_PORT']
    hosts = int(os.environ['WORLD_SIZE'])  # 机器个数
    rank = int(os.environ['RANK'])  # 当前机器编号
    gpus = torch.cuda.device_count()  # 每台机器的GPU个数

    dist.init_process_group(backend='nccl', init_method="tcp://" + str(ip) + ":" + str(port), world_size=hosts * gpus, rank=rank * gpus + local_gpu_rank)
    torch.cuda.set_device(local_gpu_rank)
    args.device = torch.device("cuda", local_gpu_rank)
    args.world_size = hosts * gpus
    args.local_rank = local_gpu_rank
    
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO if local_gpu_rank in [-1, 0] else logging.WARN)
    logger.info(tokenizer.encode("<s> hello world <pad> </s>"))
    logger.info("args: %s", args)
    
    model = model_class.from_pretrained(args.pretrain_dir)
    model.resize_token_embeddings(len(tokenizer))
    update_config(model, tokenizer)
    logger.info("model config: {}".format(model.config))

    model_parameters = model.parameters()
    num_params = sum([np.prod(p.size()) for p in model_parameters])
    logger.info("Model has a total of " + str(num_params) + " trainable parameters")

    # model_state_dict = torch.load("../output/model/gen.ddp.pt")  
    # model.load_state_dict(model_state_dict, strict=True)  # load model

    model.to(args.device)  # move the model to GPU
    # model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_gpu_rank], find_unused_parameters=True) # no need when eval
    model.eval()

    test_dataset = editDataset(tokenizer, args, logger, file_type=args.infer_file_type, block_size=args.block_size, mode='test', with_id=args.with_id)
    # test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset, shuffle=False)
    test_sampler = SequentialSampler(test_dataset)
    # if not hasattr(args, "per_gpu_eval_batch_size"):
    # collate_fn = DataCollatePad(pad_ids=[tokenizer.pad_token_id, 0], without_pad_idxs=[2])
    args.per_gpu_eval_batch_size = 1
    test_dataloader = DataLoader(test_dataset, batch_size=args.per_gpu_eval_batch_size, sampler=test_sampler)  # the batch size on each GPU


    logger.info("***** Running generation *****")
    logger.info("  Num examples = %d", len(test_dataset))
    logger.info("Per Batch size = %d", args.per_gpu_eval_batch_size)
    logger.warning("rank: %s, device: %s", local_gpu_rank, args.device)
    logger.info("Start generating...")

    # reload step
    reload_step_path = os.path.join(args.output_dir, str(args.infer_file_type) + "-"+str(args.infer_prefix)+".multi.infer.step")
    reload_step = -1
    all_lines = []
    try:
        if os.path.exists(reload_step_path):
            with open(reload_step_path, "r") as f:
                all_step = f.readlines()
                all_step = [json.loads(x) for x in all_step]
            if len(all_step) > 0:
                all_step = all_step[-1]
                logger.info(str(all_step))
                reload_step = all_step["step"]
                reload_jsonl_path = os.path.join(args.output_dir, str(args.infer_file_type) + "-"+str(args.infer_prefix)+".multi.infer.jsonl")
                with open(reload_jsonl_path,'r') as f:
                    all_lines = json.load(f)
                logger.info("reload from step: %s, len: %s", reload_step, len(all_lines))
                logger.info("reload_step_path: %s", reload_step_path)
    except Exception as e:
        logger.info("reload error: %s", e)
        logger.info(traceback.format_exc())
        logger.info("reload_step_path: %s", reload_step_path)
        logger.info("reload_step_path exists: %s", os.path.exists(reload_step_path))
        reload_jsonl_path = os.path.join(args.output_dir, str(args.infer_file_type) + "-"+str(args.infer_prefix)+".multi.infer.jsonl")
        logger.info("reload_jsonl_path: %s", reload_jsonl_path)
        logger.info("reload_jsonl_path exists: %s", os.path.exists(reload_jsonl_path))
        reload_step = -1
        all_lines = []

    logger.info("Reload step: %d", reload_step)

    if local_gpu_rank == 0:
        count = len(all_lines)
        save_preds = all_lines
        logger.info("len of save_preds: %s", len(save_preds))
        logger.info("Start generating...")
        # fw = open(os.path.join(args.output_dir, str(args.infer_file_type) + "-"+str(args.infer_prefix)+".multi.infer.jsonl"), 'a')
    with torch.no_grad():
        # test_sampler.set_epoch(0)  # keep all data the same on all GPUs, it is usually used in training, I'm not sure if it is necessary in inference
        for step, test_data in enumerate(test_dataloader):
            if step <= reload_step:
                continue
            # for key in test_data.keys():
            #     test_data[key] = test_data[key].to(args.device)
            if args.with_id:
                batch, token_labels, ids = test_data
                ids = ids.to(args.device)
            else:
                batch, token_labels = test_data
            if step == 0: # logging
                if args.with_id:
                    logger.info("rank: %s, batch: %s, token_labels: %s, ids: %s", local_gpu_rank, batch.shape, token_labels.shape, ids.shape)
                else:
                    logger.info("rank: %s, batch: %s, token_labels: %s", local_gpu_rank, batch.shape, token_labels.shape)
            inputs = batch.to(args.device)
            
            # if args.sample_type == "greedy":
            #     outputs = model.generate(inputs, max_length=args.block_size, do_sample = False, num_beams=1, temperature=args.temperature, early_stopping=False, top_k=70, \
            #           bos_token_id=tokenizer.bos_token_id, eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id)
            if "top_p" in args.sample_type:
                # sample_type = "top_p:200"
                sample_num = int(args.sample_type.split(":")[-1])
                outputs = model.generate(inputs, max_length=args.block_size, do_sample=True, temperature=0.7, top_k=70, top_p=0.95, num_return_sequences=sample_num, bos_token_id=tokenizer.bos_token_id, eos_token_id=tokenizer.pad_token_id, pad_token_id=tokenizer.pad_token_id)
                aligned_inputs = inputs.repeat_interleave(sample_num, dim=0) # repeat to match the outputs
                if args.with_id:
                    aligned_ids = ids.repeat_interleave(sample_num, dim=0)
            else:
                assert False, "sample type not supported"

            if outputs.size(1) < args.block_size:  # need padding because the lengths from different GPUs may be different
                batch_pred_padding = torch.full((outputs.size(0), args.block_size - outputs.size(1)), tokenizer.pad_token_id, dtype=outputs.dtype, device=outputs.device) # use the padding token of BART, and its token id is 1. Be careful with the data type.
                outputs = torch.cat([outputs, batch_pred_padding], dim=1)
            if aligned_inputs.size(1) < args.block_size:  # need padding because the lengths from different GPUs may be different
                batch_inp_padding = torch.full((aligned_inputs.size(0), args.block_size - aligned_inputs.size(1)), tokenizer.pad_token_id, dtype=aligned_inputs.dtype, device=aligned_inputs.device) # use the padding token of BART, and its token id is 1. Be careful with the data type.
                aligned_inputs = torch.cat([aligned_inputs, batch_inp_padding], dim=1)


            batch_pred = [torch.zeros_like(outputs, dtype=outputs.dtype).cuda() for _ in range(args.world_size)]  # initialized a list for collecting tensors from all GPUs. Be careful with the data type.
            batch_inputs = [torch.zeros_like(aligned_inputs, dtype=aligned_inputs.dtype).cuda() for _ in range(args.world_size)]  # initialized a list for collecting tensors from all GPUs. Be careful with the data type.

            dist.all_gather(batch_pred, outputs)  # collect data
            dist.all_gather(batch_inputs, aligned_inputs)  # collect data
            if args.with_id:
                batch_ids = [torch.zeros_like(aligned_ids, dtype=aligned_ids.dtype).cuda() for _ in range(args.world_size)]
                dist.all_gather(batch_ids, aligned_ids)  # collect data
                batch_ids = [e.unsqueeze(1) for e in batch_ids]
                batch_ids = torch.stack(batch_ids, dim=1)
                batch_ids = batch_ids.view(-1)
                batch_ids = batch_ids.cpu().numpy().tolist()


            batch_pred = torch.stack(batch_pred, dim=1)  # use stack, take care of the dimension
            batch_pred = batch_pred.reshape(-1, args.block_size)
            batch_inputs = torch.stack(batch_inputs, dim=1)  # use stack, take care of the dimension
            batch_inputs = batch_inputs.reshape(-1, args.block_size)
            
            if local_gpu_rank == 0:
                batch_out_sentences = tokenizer.batch_decode(batch_pred, skip_special_tokens=True, clean_up_tokenization_spaces=False)  # decode the token id to token
                batch_in_sentences = tokenizer.batch_decode(batch_inputs, skip_special_tokens=True, clean_up_tokenization_spaces=False)  # decode the token id to token
                if not args.with_id:
                    batch_ids = [-1] * len(batch_out_sentences)

                assert len(batch_out_sentences) == len(batch_in_sentences) == len(batch_ids), "len of batch_out_sentences and batch_in_sentences should be the same, but got {} and {} and {}".format(len(batch_out_sentences), len(batch_in_sentences), len(batch_ids))

                for each_id, each_in, each_out in zip(batch_ids,batch_in_sentences, batch_out_sentences):
                    save_preds.append({"id": each_id,"input": each_in, "output": each_out})
                fw_path = os.path.join(args.output_dir, str(args.infer_file_type) + "-"+str(args.infer_prefix)+".multi.infer.jsonl")
                with open(fw_path, 'w') as f:
                    json.dump(save_preds, f)
                count += len(batch_out_sentences)
                with open(os.path.join(args.output_dir, str(args.infer_file_type) + "-"+str(args.infer_prefix)+".multi.infer.step"), 'w') as fw_step:
                    fw_step.write(json.dumps({"step": step, "count": count}) + "\n")
                logger.info("rank0 step: {}, count: {}, len of preds: {}".format(step, count, len(save_preds)))
                if step == 0:
                    logger.info("save_preds[-1]: {}".format(save_preds[-1]))
            # logger.warning(">>>rank: {}, step: {}".format(local_gpu_rank, step))
    if local_gpu_rank == 0:
        logger.info("Finished generating {} lines.".format(count))
        save_to_path = os.path.join(args.output_dir, str(args.infer_file_type) + "-"+str(args.infer_prefix)+".multi.infer.jsonl")
        logger.info("Saving to {}".format(save_to_path))
        logger.info("Finish!")

        
if __name__ == "__main__":
    ngpus = torch.cuda.device_count()
    hfai.multiprocessing.spawn(test_model_generation, args=(args, ), nprocs=ngpus, bind_numa=True)