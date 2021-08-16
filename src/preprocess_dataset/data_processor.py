import json
import glob
import logging
import io
import random
import numpy as np
import os
import time
import re
import copy

from nltk import word_tokenize
from .MATH import alphanum_key
from .lang import InputLang, OutputLang
from expression_tree import *
from .batch import MATHBatchItem, MATHBatch
from .MATHProblem import MATHProblem
from .data_utils import indexes_from_sentence


class DataProcessor(object):
	NUM_REPLACER = 'NUM'
	TRIM_MIN_COUNT = 5

	def __init__(self, data, trim_min_count=None):
		self.data = data
		self.trim_min_count = trim_min_count if trim_min_count is not None else DataProcessor.TRIM_MIN_COUNT
		self.var_nums = []

		pairs, self.generate_nums, self.copy_nums = self.transfer_nums(self.data)

		for pair in pairs:
			ept = ExpressionTree()
			ept.build_tree_from_infix_expression(pair.output_seq)
			pair.output_seq = ept.get_prefix_expression()
	
		self.pairs = pairs

	def initialize(self, train_pairs, test_pairs):
		"""Initializes input and output languages and prepares the given pairs for training."""
		self.input_lang, self.output_lang, self.train_pairs, self.test_pairs = self.prepare_data(train_pairs, test_pairs, tree=True)

		self.generate_num_ids = []
		for num in self.generate_nums:
			self.generate_num_ids.append(self.output_lang.word2index[num])

		self.var_num_ids = []
		for var in self.var_nums:
			if var in self.output_lang.word2index.keys():
				self.var_num_ids.append(self.output_lang.word2index[var])

	def prepare_pairs(self, pairs, input_lang, output_lang, tree=False):
		"""Creates num stacks, input and output token ids and a graph for all the given pairs."""
		prepared_pairs = []

		for pair in pairs:
			input_seq, eq_segs, nums = pair.input_seq, pair.output_seq, pair.nums
			num_stack = []
			for word in eq_segs:
				temp_num = []
				flag_not = True
				if word not in output_lang.index2word:
					flag_not = False
					for num_index, num in enumerate(nums):
						if num == word:
							temp_num.append(num_index)

				if not flag_not and len(temp_num) != 0:
					num_stack.append(temp_num)
				if not flag_not and len(temp_num) == 0:
					num_stack.append([_ for _ in range(len(nums))])

			num_stack.reverse()

			input_cell = indexes_from_sentence(input_lang, input_seq)
			output_cell = indexes_from_sentence(output_lang, eq_segs, tree)

			pair.input_tokens = input_cell
			pair.output_tokens = output_cell
			pair.num_stack = num_stack
			pair.build_graph()

			prepared_pairs.append(pair)

		return prepared_pairs	

	def prepare_data(self, train_pairs, test_pairs, tree=False):
		"""Builds input and output languages and populates the data with required additional attributes (token ids, num stacks and graphs)."""
		input_lang = InputLang()
		output_lang = OutputLang()

		print('Indexing words...')
		for pair in train_pairs:
			#input_seq, eq_segs, _, num_pos, _, _ = pair
			if not tree or pair.num_pos: # not empty num_pos?
				input_lang.add_sen_to_vocab(pair.input_seq)
				output_lang.add_sen_to_vocab(pair.output_seq)

		# For BERT only
		for pair in test_pairs:
			if not tree or pair.num_pos:
				input_lang.add_sen_to_vocab(pair.input_seq)
		# END


		input_lang.build_input_lang(self.trim_min_count)
		if tree:
			output_lang.build_output_lang_for_tree(self.generate_nums, self.copy_nums)
		else:
			output_lang.build_output_lang(self.generate_nums, self.copy_nums)

		print('Indexed {} words in input language, {} words in output'.format(input_lang.n_words, output_lang.n_words))

		train_prepared_pairs = self.prepare_pairs(train_pairs, input_lang, output_lang, tree)
		print('Number of data {}'.format(len(train_prepared_pairs)))

		test_prepared_pairs = self.prepare_pairs(test_pairs, input_lang, output_lang, tree)
		print('Number of testing data %d' % (len(test_prepared_pairs)))

		return input_lang, output_lang, train_prepared_pairs, test_prepared_pairs

	def prepare_batches(self, pairs, batch_size):
		"""Prepares batches of size `batch_size` from the given pairs."""
		return MATHBatch.create_from_items(pairs, batch_size)


	def _replace_numbers(self, data_item):
		"""Looks for numbers in the problem statement; replaces them by NUM."""
		number_pattern = re.compile("\d+,\d+|\d+\.\d+|\d+|\d+\.\d+%?|\d+%?")
		nums = []
		input_seq = []
		
		word_tokens = data_item.problem.strip().split()

		# Looking for numbers.
		for word_token in word_tokens:
			numbers_match = re.search(number_pattern, word_token)
			if numbers_match is not None:
				# If there are digits in the token, we need to replace them.
				if numbers_match.start() > 0:
					input_seq.append(word_token[:numbers_match.start()])

				num = word_token[numbers_match.start(): numbers_match.end()]

				nums.append(num) #.replace(",", ""))
				input_seq.append(DataProcessor.NUM_REPLACER)
				if numbers_match.end() < len(word_token):
					input_seq.append(word_token[numbers_match.end():])
			else:
				# There are no digits in the token, we can safely append it to the input sequence.
				input_seq.append(word_token)

		return nums, input_seq

	def _get_nums_from_fractions(self, nums):
		"""Converts numbers to floats; parses fractions."""
		nums_fraction = []

		for num in nums:
			if re.search("\d*\(\d+/\d+\)\d*", num):
				nums_fraction.append(num)
		nums_fraction = sorted(nums_fraction, key=lambda x: len(x), reverse=True)

		float_nums = []
		for num in nums:
			if ',' in num:
				# [TODO] It can be points with comma like (0, 1)...
				new_num = []
				for c in num:
					if c == ',':
						continue
					new_num.append(c)
				num = ''.join(new_num)
				float_num = num
			else:
				float_num = num

			try:
				float_nums.append(str(float(eval(float_num))))
			except:
				if float_num.startswith('0'):
					while float_num.startswith('0'):
						float_num = float_num[1:]
					float_nums.append(str(float(eval(float_num))))

		float_nums_fraction = []
		for num in nums_fraction:
			if ',' in num:
				new_num = []
				for c in num:
					if c == ',':
						continue
					new_num.append(c)
				num = ''.join(new_num)
				float_nums_fraction.append(str(float(eval(num))))
			else:
				float_nums_fraction.append(str(float(eval(num))))

		return float_nums, float_nums_fraction

	def transfer_nums(self, data, is_test=False):
		"""
		Transfering all numbers into 'NUM'.

		Returns
			`pairs` of form (input_seq, eq_segs, nums, num_pos, group_nums)
			`temp_g` a list of numbers met in equations but not in problem statements (not all, with threshold 5)
			`copy_nums` max number of numbers in any problem statement
		"""
		print("Transfer numbers...")
		skipped = 0

		pairs = []
		generate_nums = []
		generate_nums_dict = {}
		copy_nums = 0

		for data_item in data:
			skip = False
			# if data_item.no_expression:
			# 	skipped += 1
			# 	continue

			for var_num in data_item.var_nums:
				if len(var_num) != 1:
					skip = True
			if skip:
				skipped += 1
				continue

			self.var_nums += data_item.var_nums

			nums, input_seq = self._replace_numbers(data_item)

			if copy_nums < len(nums):
				copy_nums = len(nums)

			nums, nums_fraction = self._get_nums_from_fractions(nums)

			equations = data_item.equation
			def seg_and_tag(st):
				res = []
				for n in nums_fraction:
					if n in st:
						p_start = st.find(n)
						p_end = p_start + len(n)
						if p_start > 0:
							res += seg_and_tag(st[:p_start])
						if nums.count(n) == 1:
							res.append("N"+str(nums.index(n)))
						elif nums.count(n) > 1:
							res.append("N"+str(nums.index(n)))
						else:
							res.append(n)
						if p_end < len(st):
							res += seg_and_tag(st[p_end:])
						return res

				pos_st = re.search("\d+\.\d+%?|\d+%?", st)
				if pos_st:
					p_start = pos_st.start()
					p_end = pos_st.end()
					if p_start > 0:
						res += seg_and_tag(st[:p_start])
					st_num = st[p_start:p_end]
					if nums.count(st_num) == 1:
						res.append("N"+str(nums.index(st_num)))
					elif nums.count(st_num) > 1:
						res.append("N"+str(nums.index(st_num)))
					else:
						res.append(st_num)
					if p_end < len(st):
						res += seg_and_tag(st[p_end:])
					return res
				for ss in st:
					res.append(ss)
				return res

			try:
				out_seq = seg_and_tag(equations)
				new_out_seq = []
				for seq in out_seq:
					if seq == ' ' or seq == '':
						continue
					if seq == ';':
						new_out_seq.append('SEP')
						continue
					new_out_seq.append(seq)
				out_seq = new_out_seq
			except:
				out_seq = data_item.solution

			for s in out_seq:  # tag the num which is generated
				if s[0].isdigit() and s not in generate_nums and s not in nums:
					generate_nums.append(s)
					generate_nums_dict[s] = 0
				if s in generate_nums and s not in nums:
					generate_nums_dict[s] = generate_nums_dict[s] + 1


			num_pos = []
			group_nums = []
			for i, token in enumerate(input_seq):
				if token == DataProcessor.NUM_REPLACER:
					num_pos.append(i)
					if i > 0:
						group_nums.append(i - 1)
					group_nums.append(i)
					if i < len(input_seq) - 1:
						group_nums.append(i + 1)

			if len(nums) != len(num_pos):
				skipped += 1
				continue

			pairs.append(
				MATHBatchItem(
					input_seq,
					out_seq,
					nums,
					num_pos,
					[],
					group_nums,
					answers=data_item.solution,
					filename=data_item.filename,
				)
			)

		temp_g = []
		for g in generate_nums:
			if generate_nums_dict[g] >= 5:
				temp_g.append(g)

		print('Skipped in transfer_num ', skipped)

		self.var_nums = list(set(self.var_nums))

		return pairs, temp_g, copy_nums


def read_files(dataroot, is_test=False, preprocess=False, aug_dataset_path=None):
	"""Reads dataset files and returns a list of MATHProblem items."""
	all_filenames = sorted(glob.glob(dataroot), key=alphanum_key)
	samples_raw = []
	preprocessed = 0
	skipped = 0

	for fname in all_filenames:
		with open(fname, 'r') as fp:
			try:
				problem_data = json.load(fp)
				#print(fname)
			except Exception as e:
				print(f"Error loading JSON from {fname}", e)
				raise e

		problem = MATHProblem(
			fname,
			problem_data['problem'],
			problem_data['type'],
			problem_data['level'],
			explanation=problem_data['explanation'] if not is_test else None,
			solution=problem_data['solution'] if not is_test else None,
			equation=problem_data['equation'] if not is_test else None,
			json_data=problem_data,
			preprocess=preprocess,
		)

		if problem.no_expression:
			skipped += 1
		samples_raw.append(problem)

	if not is_test and aug_dataset_path:
		print('Adding another dataset to the samples...')
		with open(aug_dataset_path) as fp:
			data = json.load(fp)
			for data_item in data:
				problem = MATHProblem(
					'augmented',
					data_item['sQuestion'],
					'MWP',
					'NA',
					solution=str(data_item['lSolutions'][0]),
					equation=data_item['new_equation'],
				)

				if problem.no_expression:
					skipped += 1
				samples_raw.append(problem)	

	print('There are no expressions for {} in {} items'.format(skipped, len(samples_raw)))
	return samples_raw


def prepare_folds(pairs, seed, is_test=False, number_of_folds=5):
	"""Prepares folds."""
	random.seed(seed)
	random.shuffle(pairs)
	folds = []
	for fold_index in range(number_of_folds):
		train_pairs = [pairs[i] for i in range(len(pairs)) if i % number_of_folds != fold_index]
		test_pairs = [pairs[i] for i in range(len(pairs)) if i % number_of_folds == fold_index]

		folds.append((train_pairs, test_pairs))

	return folds


