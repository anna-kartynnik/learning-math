import argparse
import io
import logging
import math
import os
import regex as re
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

from preprocess_dataset.MATH import MATHDataset
from utils.utils import last_boxed_only_string
from utils.equivalent import strip_string


# Test dataset will not contain any answers.
# Output of run_eval method should only be outputs array. Nothing else.

def get_dataset(args):
    all_datasets = []
    if args.math_dataroot is not None:
        all_datasets.append(
            MATHDataset(
                dataroot=args.math_dataroot,
                tokenizer=None,  # Set in run_training(), not in dataset creation
                max_tokens=384 if args.arch == 'gpt2-xl' else 1024,
                mode='gpt2-eval'
            )
        )

    data = torch.utils.data.ConcatDataset(all_datasets)
    return data


def get_level_type_solution(fname):
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
    if 'solution' in problem_data:
        solution = problem_data['solution']
    else:
        solution = None 
    try:
        level = int(level.split("Level ")[1])
    except:
        level = None
    return level, prob_type, solution


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
    if ret[-1] == tokenizer.eos_token_id:
        ret = ret[:-1]
    return ret


def get_float_answer(output):
    float_answer = 0.0
    try:
        float_answer = float(output)
    except ValueError as e:
        print(e)
        if 'frac' in output:
            frac = re.findall(r'\d+', output)
            float_answer = float(frac[0]) / float(frac[1])

    return round(float_answer, ndigits=2)  # [TODO] 2 to config? 

def run_eval(args):

    argsdict = vars(args)
    print(pprint.pformat(argsdict))

    tokenizer = transformers.GPT2Tokenizer.from_pretrained(args.arch)

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
        model = transformers.GPT2LMHeadModel.from_pretrained(args.arch)
    else:
        print(f"Loading model from {args.load}")
        model = transformers.GPT2LMHeadModel.from_pretrained(args.load)
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
            prob_level, prob_type, prob_solution = get_level_type_solution(fnames[0])
            batch = dict_to_gpu(batch, device_id=0)   # Uncomment if GPU is available

            output_ids = model.generate(
                batch['input_ids'],
                num_beams=args.num_beams,
                early_stopping=True,
                temperature=1.0,
                max_length=384 if args.arch == 'gpt2-xl' else 1024
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

            total += 1
            if prob_solution is not None:
                predicted = get_float_answer(output_full)
                real = get_float_answer(prob_solution)

                if predicted == real:
                    correct += 1

            outputs.append(output_full)


    print("Final statistics:")
    print(f'correct {correct}')
    print(f'skipped {skipped}')
    print(f'total {total}')
    print(f'score {float(correct) / total}')

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
    rows = {'Id': range(len(outputs)), 'Predicted': outputs}
    dataframe = pd.DataFrame(rows, columns=fieldnames)
    dataframe.to_csv('predicted_raw.csv', index=False)

    outputs_real = []
    for output in outputs:
        float_answer = get_float_answer(output)

        outputs_real.append(float_answer)
    
    rows = {'Id': range(len(outputs_real)), 'Predicted': outputs_real}
    dataframe = pd.DataFrame(rows, columns=fieldnames)
    dataframe.to_csv('predicted.csv', index=False)

def main():
    parser = argparse.ArgumentParser(description="Language Modelling on Code")
    parser.add_argument('--arch', default='gpt2', choices=transformers.GPT2_PRETRAINED_MODEL_ARCHIVE_LIST)
    parser.add_argument('--load', default=None, type=str)
    parser.add_argument('--num-beams', default=20, type=int)

    # Dataloading
    parser.add_argument('--math-dataroot', default=None, type=str)
    parser.add_argument('--math-mode', default='gpt2-eval', type=str)

    # Others
    parser.add_argument('--workers', default=4, type=int)

    args = parser.parse_args()

    outputs = run_eval(args)
    generate_csv(outputs)
    
if __name__ == "__main__":
    main()
