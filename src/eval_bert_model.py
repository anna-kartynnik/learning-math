import argparse
import io
import logging
import math
import os
import pprint
import sys
import json
import time
import transformers
import numpy as np
import csv
import pandas as pd

from tqdm import tqdm

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.multiprocessing as mp

from torch.nn.parallel import DistributedDataParallel as DDP

from preprocess_dataset.MATH_BERT import MATHDataset
from utils.utils import last_boxed_only_string
from utils.equivalent import strip_string
# os.environ['CUDA_VISIBLE_DEVICES'] = 'None'


# Test dataset will not contain any answers.
# Output of run_eval method should only be outputs array. Nothing else.

def get_dataset(args):
    all_datasets = []
    if args.math_dataroot is not None:
        all_datasets.append(
            MATHDataset(
                dataroot=args.math_dataroot,
                tokenizer=None,  # Set in run_training(), not in dataset creation
                max_tokens=512,
                mode='bert-eval'
            )
        )

    data = torch.utils.data.ConcatDataset(all_datasets)
    return data


def get_level_type(fname):
    """
    Somewhat inefficient, but much easier than changing dataloader and probably fine for evaluation
    """
    with open(fname, 'r') as fp:
        try:
            problem_data = json.load(fp)
        except Exception as e:
            print(f"Error loading JSON from {fname}", e)
            raise e
    level, prob_type = problem_data['level'], problem_data['type']
    try:
        level = int(level.split("Level ")[1])
    except:
        level = None
    return level, prob_type


def dict_to_gpu(d, device_id=None):
    new_dict = dict()
    for key, value in d.items():
        # Only move to GPU is cuda() is a function
        if 'cuda' in dir(value):
            new_dict[key] = value.cuda(device_id)
        else:
            new_dict[key] = value
    return new_dict


def get_model_output(context, full_output, tokenizer):
    """
    Given the context and the full model output (context + generated),
    extract just the generated tokens.
    Remove the last token if it is <|endoftext|>
    """
    ret = full_output[len(context):]
    # return ret
    if ret[-1] == 102:
        ret = ret[:-1]
    return ret


def run_eval(args):

    argsdict = vars(args)
    print(pprint.pformat(argsdict))

    tokenizer = transformers.BertTokenizer.from_pretrained("bert-base-uncased")

    # SHOULD NOT CONTAIN ANY ANSWER
    eval_data = get_dataset(args)
    for inner_dset in eval_data.datasets:
        inner_dset.tokenizer = tokenizer

    dataloader = torch.utils.data.DataLoader(
        eval_data,
        batch_size=1,
        num_workers=0,
        pin_memory=True,
    )

    # Set up model
    if args.load is None:
        model = transformers.BertLMHeadModel.from_pretrained("bert-base-uncased", is_decoder=True)
    else:
        print(f"Loading model from {args.load}")
        model = transformers.BertLMHeadModel.from_pretrained(args.load, is_decoder=True)
        print(f"Successfully loaded model from {args.load}")

    model = model.eval()
    model = model.cuda()  # Uncomment out the line if working on GPUs

    loss_moving_average = 0

    outputs = []
    types = []
    levels = []
    fnames_list = []

    cors = {}
    subject_cors = {}
    level_cors = {}

    with torch.no_grad():
        correct = 0
        total = 0
        skipped = 0
        mean_max_probs_correct = []
        mean_max_probs_wrong   = []
        for i, batch in enumerate(tqdm(dataloader)):

            if torch.sum(batch['input_ids']) == 0:
                skipped += 1
                print("SKIPPING", batch['fnames'][0])
                continue

            fnames = batch['fnames'][0]
            assert len(fnames) == 1
            fnames_list.append(fnames[0])
            prob_level, prob_type = get_level_type(fnames[0])
            batch = dict_to_gpu(batch, device_id=0)   # Uncomment if GPU is available

            output_ids = model.generate(
                batch['input_ids'],
                num_beams=args.num_beams,
                early_stopping=True,
                temperature=1.0,
                max_length=512
            )

            mean_probs_sol = 0

            output_tokens = get_model_output(batch['input_ids'][0], output_ids[0], tokenizer)

            # Print this iteration
            output_str = tokenizer.decode(output_tokens)
            output_full = strip_string(output_str)

            print("Problem String:")
            print(tokenizer.decode(batch['input_ids'][0]) + "\n")
            print("Model output:")
            print(output_full)
            print("fname")
            print(fnames)
            print("--------------------------------------------")

            outputs.append(output_full)
    return outputs

def remove_boxed(s):
    left = "\\boxed{"
    try:
        assert s[:len(left)] == left
        assert s[-1] == "}"
        return s[len(left):-1]
    except:
        return None

def generate_csv(outputs):
    fieldnames = ['Id', 'Predicted']
    ## Replace any strings by 0 for the time being:
    for outidx, outs in enumerate(outputs):
        try:
            float(outs)
        except ValueError:
            outputs[outidx] = 0

    outputs_real = [float(s) for s in outputs]
    rows = {'Id': range(len(outputs_real)), 'Predicted': outputs_real}
    dataframe = pd.DataFrame(rows, columns=fieldnames)
    dataframe.to_csv('predicted.csv', index=False)

def main():
    parser = argparse.ArgumentParser(description="Language Modelling on Code")
    parser.add_argument('--arch', default='bert', choices=transformers.BERT_PRETRAINED_MODEL_ARCHIVE_LIST)
    parser.add_argument('--load', default=None, type=str)
    parser.add_argument('--num-beams', default=20, type=int)

    # Dataloading
    parser.add_argument('--math-dataroot', default=None, type=str)
    parser.add_argument('--math-mode', default='bert-eval', type=str)

    # Others
    parser.add_argument('--workers', default=4, type=int)

    args = parser.parse_args()

    outputs = run_eval(args)
    generate_csv(outputs)
    
if __name__ == "__main__":
    main()
