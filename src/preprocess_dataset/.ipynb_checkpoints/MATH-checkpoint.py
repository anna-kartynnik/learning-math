import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import glob
import logging
import io
import random
import numpy as np
import os
import time

from utils.utils import last_boxed_only, _clean_numbers, last_boxed_only_string, only_until_first_boxed_from_tokens

from multiprocessing import Manager

from torch.multiprocessing import Pool


# from dataset.base_math_dataset import BaseMathDataset


class MATHDataset(torch.utils.data.Dataset):
    def __init__(self, dataroot, tokenizer, max_tokens, mode, mode_answer='default', len_multiplier=1.0, packing=None,
                 randomize=None, pack_end=None, clean_numbers=False, latex_mask=False, peek_fraction=(0.1, 1.0)):
        self.dataroot = dataroot
        self.tokenizer = tokenizer  # Set in run_training(), not in dataset creation
        self.max_tokens = max_tokens
        self.mode = mode
        # self.mode_answer = mode_answer # Used in subclass
        self.len_multiplier = len_multiplier
        self.clean_numbers = clean_numbers
        self.latex_mask = latex_mask
        # self.peek_fraction = peek_fraction

        if self.mode in {'gpt2'}:
            self.clean_sample = self.clean_filter_sample_gpt
            self.packing = True
            self.randomize = True
            self.include_fnames = False
            self.pack_end = True
            self.initialize()
        elif self.mode in {'gpt2-eval'}:
            self.clean_sample = self.clean_filter_sample_gpt_eval
            self.packing = False
            self.randomize = False
            self.include_fnames = True
            self.pack_end = True
            self.initialize(test=True)
        else:
            raise NotImplementedError()
        
        if packing != None:
            print("Overriding packing to be", packing)
            self.packing = packing
        if randomize != None:
            print("Overriding randomize to be", randomize)
            self.randomize = randomize
        if pack_end != None:
            print("Overriding pack_end to be", pack_end)
            self.pack_end = pack_end

#         self.initialize()
        self.bad_fnames = set()
        self.i = 0
        
    def __len__(self):
        return int(len(self.samples) * self.len_multiplier)

    def initialize(self, test=False):
        """
        Set up self.samples by loading from the dataroot
        """

        all_filenames = glob.glob(self.dataroot)
        samples_raw = []
        for fname in all_filenames:
            with open(fname, 'r') as fp:
                try:
                    problem_data = json.load(fp)
                    print(fname)
                except Exception as e:
                    print(f"Error loading JSON from {fname}", e)
                    raise e
            if test:
                curr_sample_raw = (problem_data['problem'], fname)
            else:
                curr_sample_raw = (problem_data['problem'], problem_data['solution'], fname)
            for e in curr_sample_raw:
                assert e
            samples_raw.append(curr_sample_raw)

        manager = Manager()
        samples_raw = manager.list(samples_raw)
        self.samples = samples_raw
        del samples_raw

        print(f"{self.__class__.__name__}: Loaded {len(self.samples)} samples.")

    def clean_filter_sample_gpt(self, sample):
        """
        Does the actual tokenization. Should be parallelized because it can be a bit slow.
        """
        if sample is None:
            return None

        question, answer = sample

        if self.clean_numbers:
            question = _clean_numbers(question)
            answer = _clean_numbers(answer)
        # MAY NOT BE REQUIRED IF THE SOLUTION IS ONLY A NUMBER
#         answer_final = last_boxed_only_string(answer)
        answer_final = answer
        if not answer_final:
            print("ERROR FROM", question, answer_final)
            return None

        question_ids = torch.LongTensor(self.tokenizer.encode("\nQUESTION:\n" + question, verbose=False))

        sep_ids_1 = torch.LongTensor(self.tokenizer.encode("\nFINAL ANSWER:\n", verbose=False))
        answer_final_ids = self.tokenizer.encode(answer_final, verbose=False)
        answer_final_ids.append(self.tokenizer.eos_token_id)
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

        # Stop early if this Q,A pair is too long
        if input_ids.shape[0] > self.max_tokens:
            return None

        input_ids = input_ids.tolist()
        label_ids = label_ids.tolist()

        return {
            'input_ids_list': input_ids,
            'label_ids_list': label_ids
        }

    def clean_filter_sample_gpt_eval(self, sample):
        """
        Does tokenization for final model evaluation. This should return
        input_ids as the context and labels as the true answer.
        """

        if sample is None:
            return None

#         if self.mode_answer == 'eval_peeking':
#             return self.clean_filter_sample_peeking_gpt_eval(sample)
#         elif self.mode_answer == 'eval_nopack_padding':
#             return self.clean_filter_sample_nopackpadding_gpt_eval(sample)

#         question, answer = sample
        print(sample)
        question = sample

        if self.clean_numbers:
            question = _clean_numbers(question)
#             answer   = _clean_numbers(answer)
        # MAY NOT BE REQUIRED 
#         answer_final = last_boxed_only_string(answer)
#         answer_final = answer

#         assert not answer.isspace()

        question_ids = torch.LongTensor(self.tokenizer.encode("\nQUESTION:\n" + question, verbose=False))
        sep_ids      = torch.LongTensor(self.tokenizer.encode("\FULL SOLUTION:\n", verbose=False))
#         answer_final_ids   = torch.LongTensor(self.tokenizer.encode(answer_final, verbose=False)) # Loss only counted on these tokens.

        input_ids = torch.cat([
            question_ids, 
            sep_ids, 
        ], dim=0)

#         label_ids = torch.cat([
#             answer_final_ids.clone()
#         ], dim=0)
        
        # Stop early if this Q,A pair is too long
#         if input_ids.shape[0] + label_ids.shape[0] > self.max_tokens:
        if input_ids.shape[0] > self.max_tokens:
            return None
        
        return {
            'input_ids_list' : input_ids.tolist()
#             'label_ids_list' : label_ids.tolist()
        }

    def __getitem__(self, index):

        # Each worker needs a different seed....
        random.seed(os.getpid() + time.time() + random.random())

        # Sampling with replacement.
        # We need to pack random elements to get close to self.max_tokens
        curr_input_ids = []
        curr_label_ids = []
        curr_fnames = []
        num_samples = 0
        while len(curr_input_ids) + 1 <= self.max_tokens and len(curr_label_ids) + 1 <= self.max_tokens:
            curr_sample, fname = self.get_random_sample()
            if curr_sample is None:
                # This only happens in eval modes
                return {
                    "input_ids": torch.zeros([self.max_tokens]),
                    "labels": torch.zeros([self.max_tokens]),
                    "fnames": [fname]
                }
            if "label_ids_list" not in curr_sample:
                curr_sample['label_ids_list'] = torch.zeros([len(curr_sample['input_ids_list'])])

            if not self.pack_end and (
                    (len(curr_input_ids) + 1 + len(curr_sample['input_ids_list']) > self.max_tokens) or
                    (len(curr_label_ids) + 1 + len(curr_sample['label_ids_list']) > self.max_tokens)
            ):
                # Do not include curr_sample if either the input_ids or the label_ids will run off the end.
                break

            # Add curr_sample to the current inputs and labels
            curr_input_ids.extend(curr_sample['input_ids_list'])
            curr_label_ids.extend(curr_sample['label_ids_list'])
            curr_fnames.append(fname)

            num_samples += 1

            # Break on the first iteration if we don't want to do packing.
            if not self.packing:
                break

        input_ids = torch.LongTensor(curr_input_ids)
        label_ids = torch.LongTensor(curr_label_ids)

        # Sanity check
        if 'eval' not in self.mode:
            assert len(curr_input_ids) == len(curr_label_ids)

        input_ids = input_ids[:self.max_tokens]
        label_ids = label_ids[:self.max_tokens]

        if len(curr_input_ids) < self.max_tokens and 'eval' not in self.mode:
            # Pad
            num_to_pad = self.max_tokens - len(curr_input_ids)
            input_ids = F.pad(input_ids, [0, num_to_pad], mode='constant', value=self.tokenizer.pad_token_id)

        if len(curr_label_ids) < self.max_tokens and 'eval' not in self.mode:
            num_to_pad = self.max_tokens - len(curr_label_ids)
            label_ids = F.pad(label_ids, [0, num_to_pad], mode='constant', value=-100)

        # Sanity check
        if 'eval' not in self.mode:
            assert input_ids.shape[0] == label_ids.shape[
                0] == self.max_tokens, f"{input_ids.shape[0]}, {label_ids.shape[0]}, {self.max_tokens}"

        if self.include_fnames:
            return {
                "input_ids": input_ids,
                "labels": label_ids,
                "fnames": curr_fnames
            }
        else:
            # This is the format required by our GPT2Trainer class
            return {
                "input_ids": input_ids,
                "labels": label_ids
            }

    def get_random_sample(self):
        """
        Get a full on random sample (used for training)
        """
        random_sample = None
        while random_sample is None:
            if self.randomize:
                sample = random.choice(self.samples)
                if len(sample) == 3:
                    q, a, fname = sample
                    random_sample = self.clean_sample((q, a))
                else:
                    q, fname = sample
                    random_sample = self.clean_sample((q))
            else:
                sample = self.samples[self.i]
                if len(sample) == 3:
                    q, a, fname = sample
                    random_sample = self.clean_sample((q, a))
                else:
                    q, fname = sample
                    random_sample = self.clean_sample((q))
#                 q, a, fname = self.samples[self.i]
                self.i = (self.i + 1) % len(self.samples)

#             random_sample = self.clean_sample((q, a))

            if not self.randomize:
                break

        return random_sample, fname
