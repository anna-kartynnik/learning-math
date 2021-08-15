import io
import logging
import math
import os
import pprint
import sys
import time
import json
import argparse

import transformers

from tqdm import tqdm
from datetime import datetime

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.multiprocessing as mp

import pickle

import glob
import re
from utils.utils import last_boxed_only, _clean_numbers, last_boxed_only_string, only_until_first_boxed_from_tokens

## Helper functions from original code
def tryint(s):
    try:
        return int(s)
    except:
        return s

def alphanum_key(s):
    """ Turn a string into a list of string and number chunks.
        "z23a" -> ["z", 23, "a"]
    """
    return [ tryint(c) for c in re.split('([0-9]+)', s) ]


def export_data(args):

    ## Define the tokenizer
    if args.tokenizer == 'roberta':
        tokenizer = transformers.RobertaTokenizer.from_pretrained('roberta-large')
    elif args.tokenizer == 'bert':
        tokenizer = transformers.BertTokenizer.from_pretrained('bert-base-uncased')
    elif args.tokenizer == 'bigbird':
        tokenizer = transformers.BigBirdTokenizer.from_pretrained('google/bigbird-roberta-base')
    elif args.tokenizer == 'gpt2':
        tokenizer = transformers.GPT2Tokenizer.from_pretrained('gpt2')


    ## Get a list of all the training json files
    all_filenames = sorted(glob.glob(args.MATH_dataroot), key=alphanum_key)
    samples_raw = []
    samples_parsed = []

    ## Loop over every file
    for fname in all_filenames:
        with open(fname, 'r') as fp:
            try:
                problem_data = json.load(fp)
                print(fname)
            except Exception as e:
                print(f"Error loading JSON from {fname}", e)
                raise e

        ## Extract all the keys from the json file (includes new equation key)
        if args.mode == 'train':
            curr_sample_raw = (problem_data['problem'], problem_data['solution'],
                            problem_data['level'], problem_data['type'],
                            problem_data['explanation'], problem_data['equation'],
                            fname)
        else:
            curr_sample_raw = (problem_data['problem'], '',
                            problem_data['level'], problem_data['type'],
                            '', '',
                            fname)

        samples_raw.append(curr_sample_raw)

        print(curr_sample_raw)
        question, answer, difficulty, mathcategory, answer_explanation, equation, origfile = curr_sample_raw

        ## Clean the numbers for both the question and answer
        question = _clean_numbers(question)
        answer = _clean_numbers(answer)
        answer_final = answer

        ## Encode the question
        question_ids = torch.LongTensor(tokenizer.encode("\nQUESTION:\n" + question, verbose=False))


        ## Encode the question and return the attention mask
        if args.tokenizer != 'gpt2':
            question_encoding = tokenizer.encode_plus("\nQUESTION:\n" + question,
                                            add_special_tokens=True,
                                            max_length=512,
                                            truncation=True,
                                            return_tensors='pt',
                                            return_token_type_ids=False,
                                            return_attention_mask=True,
                                            padding='max_length')
        else:
            question_encoding = tokenizer.encode_plus("\nQUESTION:\n" + question,
                                            add_special_tokens=True,
                                            max_length=512,
                                            truncation=True,
                                            return_tensors='pt',
                                            return_token_type_ids=False,
                                            return_attention_mask=True)

        ## Encode the answer
        sep_ids_1 = torch.LongTensor(tokenizer.encode("\nFINAL ANSWER:\n", verbose=False))
        answer_final_ids = tokenizer.encode(answer_final, verbose=False)


        answer_encoding = tokenizer.encode("\nFINAL ANSWER:\n"+answer_final,
                                                ).append(tokenizer.eos_token_id)

        ## Encode the equation
        equation_encoding = tokenizer.encode("\nEQUATION:\n"+equation, verbose=False)


        answer_final_ids = torch.LongTensor(answer_final_ids)

        input_ids = torch.cat([
            question_ids,
            sep_ids_1,
            answer_final_ids,
        ], dim=0)

        # Only answer_ids contribute to the loss
        label_ids = torch.cat([
            torch.ones_like(question_ids) * -100,
            torch.ones_like(sep_ids_1) * -100,
            answer_final_ids.clone(),
        ], dim=0)


        input_ids = input_ids.tolist()
        label_ids = label_ids.tolist()



        # print(question_encoding)
        samples_parsed.append( {
            'input_question': question,
            'input_ids_list': question_encoding['input_ids'].flatten(),
            'attention_mask': question_encoding['attention_mask'].flatten(),
            'label_ids_list': label_ids,
            'output_answer': answer_final,
            'equation': equation,
            'equation_ids_list': equation_encoding
        })

    print(samples_raw)
    print(samples_parsed)

    ## Save the encoded dataset as a pickle that can be loaded before training
    with open('all-math-data.pickle', 'wb') as pick:
        pickle.dump(samples_parsed, pick)





def main():
   

    parser = argparse.ArgumentParser(description="Preprocess and export the MATH dataset")
    parser.add_argument('--tokenizer', default='gpt2', choices=['gpt2', 'bert', 'roberta', 'bigbird', 't5'])
    parser.add_argument('--MATH-dataroot', default='./dataset/train/*/*', type=str)
    parser.add_argument('--mode', default='train', choices=['train', 'test'])


    args = parser.parse_args()

    argsdict = vars(args)
    print(pprint.pformat(argsdict))


    export_data(args)

if __name__ == "__main__":
    main()
