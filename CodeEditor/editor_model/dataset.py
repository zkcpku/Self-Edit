# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
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
import gc
import shutil
import json
from tqdm import tqdm

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler,TensorDataset
from torch.utils.data.distributed import DistributedSampler
import multiprocessing

# try:
#     from torch.utils.tensorboard import SummaryWriter
# except:
#     from tensorboardX import SummaryWriter

from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup,
                          BertConfig, BertForMaskedLM, BertTokenizer,
                          GPT2Config, GPT2LMHeadModel, GPT2Tokenizer,
                          OpenAIGPTConfig, OpenAIGPTLMHeadModel, OpenAIGPTTokenizer,
                          RobertaConfig, RobertaForMaskedLM, RobertaTokenizer,
                          DistilBertConfig, DistilBertForMaskedLM, DistilBertTokenizer)


class editDataset(Dataset):
    def __init__(self, tokenizer, args, logger, file_type='train', block_size=512, mode='train', pool = None, with_id=False):
        self.with_id = with_id
        if args.local_rank==-1:
            local_rank=0
            world_size=1
        else:
            local_rank=args.local_rank
            world_size=torch.distributed.get_world_size()
        logger.warning("local_rank: %s, world_size: %s", local_rank, world_size)

        self.block_size = block_size
        self.mode = mode

        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
        cached_file = os.path.join(args.output_dir, file_type+"_blocksize_%d"%(block_size)+"_wordsize_%d"%(world_size)+"_rank_%d"%(local_rank)+"_small_k_%d"%(args.small_k)+"_max_sol_num_%d"%(args.max_sol_num))
        if mode != 'test' and os.path.exists(cached_file) and not args.overwrite_cache:
            if file_type == 'train':
                logger.info("Loading features from cached file %s", cached_file)
            with open(cached_file, 'rb') as handle:
                data = pickle.load(handle)
                self.inputs = data['inputs']
                self.token_labels = data['token_labels']
                if "ids" in data:
                    self.ids = data['ids']
                else:
                    self.ids = [-1] * len(self.inputs)

        else:
            self.inputs = []
            self.token_labels = []
            self.ids = []
            
            self.code_and_msg = args.code_and_msg
            self.nl_sol = args.nl_sol
            self.additional_sol = args.additional_sol

            with open(self.code_and_msg, "r") as f:
                code_and_msg = f.readlines()
            code_and_msg = [json.loads(x) for x in code_and_msg]

            code_and_msg = top_bigk2smallk(code_and_msg, small_k=args.small_k)
            
            logger.info("code_and_msg size: %d", len(code_and_msg))
            logger.info("create code_and_msg from %s", self.code_and_msg)

            with open(self.nl_sol,'r') as f:
                nl_sol = json.load(f)

            logger.info("nl_sol size: %d", len(nl_sol))
            logger.info("create nl_sol from %s", self.nl_sol)

            with open(self.additional_sol,'r') as f:
                additional_sol = json.load(f)
            
            logger.info("additional_sol size: %d", len(additional_sol))
            logger.info("create additional_sol from %s", self.additional_sol)
            
            def process_each_code_and_msg(each_code_and_msg, max_sol_num = 10):
                this_dataset = []
                question_id = each_code_and_msg['question_id']
                question_id = str(question_id)
                gen_code = each_code_and_msg['gen_code']
                error_msg = each_code_and_msg['error_msg']
                question = nl_sol[question_id]['question']
                if "no_code" not in file_type:
                    correct_solutions = nl_sol[question_id]['solutions']
                    if question_id in additional_sol:
                        correct_solutions += additional_sol[question_id]
                else:
                    correct_solutions = [""]
                correct_solutions = list(set(correct_solutions))
                # random sample
                random.shuffle(correct_solutions)
                for each_sol in correct_solutions[:max_sol_num]:
                    code = each_sol
                    nl = '<question>' +  question + '<code>' + gen_code + '<error>' + error_msg
                    # code = tokenizer.encode(code)
                    # nl = tokenizer.encode(nl)
                    # this_codes.append(code)
                    # this_nls.append(nl)
                    this_dataset.append((code, nl, tokenizer, self.mode, self.block_size))
                return this_dataset

            all_dataset = []
            for each_code_and_msg in tqdm(code_and_msg):
                this_dataset = process_each_code_and_msg(each_code_and_msg, max_sol_num = args.max_sol_num)
                all_dataset.extend(this_dataset)
            length = len(all_dataset)
            logger.info("all_dataset size: %d", length)
            self.ids = list(range(length))

            if world_size > 1:
                old_length = length
                additional_num = length % world_size
                if additional_num != 0:
                    need_add_num = world_size - additional_num
                    all_dataset = all_dataset + [all_dataset[-1]] * need_add_num
                    self.ids = self.ids + [-1] * need_add_num
                assert len(all_dataset) % world_size == 0
                assert len(self.ids) % world_size == 0
                assert len(all_dataset) == len(self.ids)

            if world_size > 1:
                all_dataset = all_dataset[local_rank::world_size]
                length = len(all_dataset)
                logger.warning("local_rank: %s, world_size: %s, length: %s", local_rank, world_size, length)
                self.ids = self.ids[local_rank::world_size]
                assert len(all_dataset) == len(self.ids)

            if pool is not None:
                all_inputs_and_labels = pool.map(get_example, tqdm(all_dataset, total=len(all_dataset)))
                self.inputs = [x[0] for x in all_inputs_and_labels]
                self.token_labels = [x[1] for x in all_inputs_and_labels]
            else:
                all_inputs_and_labels = []
                for each_dataset in tqdm(all_dataset):
                    all_inputs_and_labels.append(pad_and_get_mask_with_tokenizer(*each_dataset))
                self.inputs = [x[0] for x in all_inputs_and_labels]
                self.token_labels = [x[1] for x in all_inputs_and_labels]

            if file_type == 'train':
                logger.warning("Rank %d Training %d token, %d samples"%(local_rank, length, len(self.inputs)))
                logger.info("Saving features into cached file %s", cached_file)
            if mode != 'test':
                # if args.local_rank in [-1, 0]:
                    # with open(cached_file, 'wb') as handle:
                    #     pickle.dump({'inputs': self.inputs, 'token_labels': self.token_labels}, handle)
                with open(cached_file, 'wb') as handle:
                    pickle.dump({'inputs': self.inputs, 'token_labels': self.token_labels, "ids": self.ids}, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def pad_and_get_mask(self, code, nl, tokenizer):
        if self.mode == 'test':
            code = []
        while (len(code) + len(nl) + 2 > self.block_size):
            if (len(code) > len(nl)):
                code = code[:-1]
            else:
                nl = nl[:-1]
        if self.mode == 'train':
            inputs = nl + [tokenizer.bos_token_id] + code + [tokenizer.eos_token_id]
            labels = [1] * len(nl) + [2] * (len(code)+1) + [0]
        else:
            inputs = nl + [tokenizer.bos_token_id]
            labels = [1] * len(nl) + [2]
            return inputs, labels
        assert len(inputs) <= self.block_size
        pad_len = self.block_size - len(inputs)
        inputs += [tokenizer.pad_token_id] * pad_len
        labels += [0] * pad_len
        assert len(inputs) == len(labels)
        return inputs, labels


    def __len__(self):
        return len(self.inputs)

    
    def getitem(self, item):
        return torch.tensor(self.inputs[item]), torch.tensor(self.token_labels[item])
    def getitem_with_id(self, item):
        return torch.tensor(self.inputs[item]), torch.tensor(self.token_labels[item]), torch.tensor(self.ids[item])

    def __getitem__(self, item):
        if self.with_id:
            return self.getitem_with_id(item)
        else:
            return self.getitem(item)
        return torch.tensor(self.inputs[item]), torch.tensor(self.token_labels[item])

    # def __getitem__(self, item):
    #     return torch.tensor(self.inputs[item]), torch.tensor(self.token_labels[item])


def pad_and_get_mask_with_tokenizer(code, nl, tokenizer, mode, block_size):
    code = tokenizer.encode(code,truncation=True, max_length=block_size)
    nl = tokenizer.encode(nl,truncation=True, max_length=block_size)
    if mode == 'test':
        code = []
    while (len(code) + len(nl) + 2 > block_size):
        if (len(code) > len(nl)):
            code = code[:-1]
        else:
            nl = nl[:-1]
    if mode == 'train':
        inputs = nl + [tokenizer.bos_token_id] + code + [tokenizer.eos_token_id]
        labels = [1] * len(nl) + [2] * (len(code)+1) + [0]
    else:
        inputs = nl + [tokenizer.bos_token_id]
        labels = [1] * len(nl) + [2]
        return inputs, labels
    assert len(inputs) <= block_size
    pad_len = block_size - len(inputs)
    inputs += [tokenizer.pad_token_id] * pad_len
    labels += [0] * pad_len
    assert len(inputs) == len(labels)
    return inputs, labels
def get_example(item):
    return pad_and_get_mask_with_tokenizer(*item)

def top_bigk2smallk(code_and_msg,small_k):
    code_and_msg = sorted(code_and_msg, key=lambda x: int(x['question_id']))
    new_code_and_msg = []
    now_question_id = code_and_msg[0]['question_id']
    start_index = 0
    for i in range(len(code_and_msg)):
        if code_and_msg[i]['question_id'] != now_question_id:
            new_code_and_msg.extend(code_and_msg[start_index:i][:small_k])
            start_index = i
            now_question_id = code_and_msg[i]['question_id']
        elif i == len(code_and_msg) - 1:
            new_code_and_msg.extend(code_and_msg[start_index:i+1][:small_k])

    return new_code_and_msg


if __name__ == "__main__":
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

    local_rank = -1
    args.world_size = 8
    args.local_rank = local_rank
    
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO if local_rank in [-1, 0] else logging.WARN)
    

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

    
    if args.use_pool:
        # cpu_count = multiprocessing.cpu_count()
        cpu_count = 16
        pool = multiprocessing.Pool(cpu_count)
    else:
        pool = None
    train_dataset = editDataset(tokenizer, args, logger, 'train', block_size=args.block_size, pool=pool)