import pickle
from preprocess_dataset.MATH_EXPORT import MATHDataset
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

MAXLEN = 512
BATCH_SIZE = 8

class CreateDataset(torch.utils.data.Dataset):
    def __init__(self, data, max_tokens):
        self.data = data
        self.max_tokens = max_tokens

    def __len__(self):
        return MAXLEN

    def __getitem__(self, item):
        question = self.data[item]['input_ids_list']
        answer = self.data[item]['label_ids_list']

        print(len(question))
        print(len(answer))

        question = question[:self.max_tokens]
        answer = answer[:self.max_tokens]

        ## Make sure the answer has the same length for each
        num_to_pad = self.max_tokens - len(answer)
        answer = F.pad(torch.Tensor(answer), [0, num_to_pad], mode='constant', value=-100)

        return{
            'question': question,
            'answer': answer
        }

def create_data_loader(dataset):
    ds = CreateDataset(dataset, MAXLEN)

    return torch.utils.data.DataLoader(ds, batch_size=BATCH_SIZE, num_workers=4)


def load_data(args):

    

    with open('all-math-data.pickle', 'rb') as pick:
        dataset = pickle.load(pick)

    print(dataset)

    

    train_data_loader = create_data_loader(dataset)

    check_data = next(iter(train_data_loader))
    print(check_data.keys())


def main():
   

    parser = argparse.ArgumentParser(description="Load the preprocessed MATH dataset")
    parser.add_argument('--tokenizer', default='gpt2', choices=['gpt2', 'bert', 'roberta', 'bigbird', 't5'])
    parser.add_argument('--mode', default='train', choices=['train', 'test'])


    args = parser.parse_args()

    argsdict = vars(args)
    print(pprint.pformat(argsdict))


    load_data(args)

if __name__ == "__main__":
    main()

