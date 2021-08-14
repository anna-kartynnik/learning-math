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

# # ================= Taken from Graph2Tree repo [START] ========================= 

# # Return a list of indexes, one for each word in the sentence, plus EOS
# def indexes_from_sentence(lang, sentence, tree=False):
# 	res = []
# 	for word in sentence:
# 		if len(word) == 0:
# 			continue
# 		if word in lang.word2index:
# 			res.append(lang.word2index[word])
# 		else:
# 			res.append(lang.word2index["UNK"])
# 	if "EOS" in lang.index2word and not tree:
# 		res.append(lang.word2index["EOS"])
# 	return res

# # ================= Taken from Graph2Tree repo [END] ========================= 




class DataProcessor(object):
	NUM_REPLACER = 'NUM'
	TRIM_MIN_COUNT = 5

	def __init__(self, data, trim_min_count=None):
		self.data = data
		self.trim_min_count = trim_min_count if trim_min_count is not None else DataProcessor.TRIM_MIN_COUNT

		self.var_nums = []

		pairs, self.generate_nums, self.copy_nums = self.transfer_nums(self.data)

		print('pairs ', pairs[:2])

		for pair in pairs:
			ept = ExpressionTree()
			ept.build_tree_from_infix_expression(pair.output_seq)
			pair.output_seq = ept.get_prefix_expression()
	
		self.pairs = pairs

	def initialize(self, train_pairs, test_pairs):
		self.input_lang, self.output_lang, self.train_pairs, self.test_pairs = self.prepare_data(train_pairs, test_pairs, tree=True)

		self.generate_num_ids = []
		for num in self.generate_nums:
			self.generate_num_ids.append(self.output_lang.word2index[num])

		self.var_num_ids = []
		for var in self.var_nums:
			if var in self.output_lang.word2index.keys():
				self.var_num_ids.append(self.output_lang.word2index[var])


		print('num_start ', self.output_lang.num_start)
		print('n_words ', self.output_lang.n_words)
		print('generate_nums ', self.generate_nums)
		print('copy_nums ', self.copy_nums)
		print('pairs ', self.pairs[:2])
		print('var_nums ', self.var_nums)

	def prepare_pairs(self, pairs, input_lang, output_lang, tree=False):
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

			prepared_pairs.append(pair)

		return prepared_pairs	

	def prepare_data(self, train_pairs, test_pairs, tree=False):
		input_lang = InputLang()
		output_lang = OutputLang()

		print('Indexing words...')
		for pair in train_pairs:
			#input_seq, eq_segs, _, num_pos, _, _ = pair
			if not tree or pair.num_pos: # not empty num_pos?
				input_lang.add_sen_to_vocab(pair.input_seq)
				output_lang.add_sen_to_vocab(pair.output_seq)

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
		return MATHBatch.create_from_items(pairs, batch_size)


	def _replace_numbers(self, data_item):
		number_pattern = re.compile("\d+,\d+|\d+\.\d+|\d+|\d+\.\d+%?|\d+%?")
		nums = []
		input_seq = []
		try:
			word_tokens = data_item.problem.strip().split() #word_tokenize(data_item.problem)
		except AttributeError as e:
			print(e)
			print(data_item.filename)


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

	def _find_num_in_problem_nums(self, problem_nums, eq_num, generate_nums):
		count_eq = []
		for n_idx, n in enumerate(problem_nums):
			if abs(float(n) - float(eq_num)) < 1e-4:
				count_eq.append(n_idx)
				if n != eq_num:
					problem_nums[n_idx] = eq_num
		if len(count_eq) == 0:
			flag = True
			for gn in generate_nums:
				if abs(float(gn) - float(eq_num)) < 1e-4:
					generate_nums[gn] += 1
					if eq_num != gn:
						eq_num = gn
					flag = False
			if flag:
				generate_nums[eq_num] = 0
			return eq_num
		elif len(count_eq) == 1:
			return 'N' + str(count_eq[0])
		else:
			return eq_num


	def transfer_nums(self, data, is_test=False):  # transfer num into "NUM"
		"""
		Transfering all numbers into 'NUM'.

		Returns
			`pairs` of form (input_seq, eq_segs, nums, num_pos, group_nums)
			`temp_g` a list of numbers met in equations but not in problem statements (not all, with threshold 5)
			`copy_nums` [TODO] ? max number of numbers in one problem statement?
		"""
		print("Transfer numbers...")
		skipped = 0

		pairs = []
		generate_nums = []
		generate_nums_dict = {}
		copy_nums = 0

		for data_item in data:
			skip = False
			if data_item.no_expression:
				skipped += 1
				continue

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

			nums_fraction = []

			for num in nums:
				if re.search("\d*\(\d+/\d+\)\d*", num):
					nums_fraction.append(num)
			nums_fraction = sorted(nums_fraction, key=lambda x: len(x), reverse=True)
			# print(nums)
			# print(nums_fraction)
			float_nums = []
			for num in nums:
				if ',' in num:
					# It can be points with comma like (0, 1)...
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
					#print('error 2 ', float_num)
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
			# print(float_nums)
			# print(float_nums_fraction)
			nums = float_nums
			nums_fraction = float_nums_fraction


			equations = data_item.equation
			def seg_and_tag(st):  # seg the equation and tag the num
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

				try:
					pos_st = re.search("\d+\.\d+%?|\d+%?", st)
				except:
					print('error', st)
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
			# print(equations)
			# print(' '.join(out_seq))
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
				#print(data_item)
				#print(nums)
				#print(num_pos)
				#print(data_item.problem)
				skipped += 1
				continue

			#assert len(nums) == len(num_pos)
			# if len(nums) != 0:
			#     pairs.append((input_seq, eq_segs, nums, num_pos, group_nums))
			#pairs.append((input_seq, eq_segs, nums, num_pos, group_nums))
			# [TODO] no solution for test data 
			

			#pairs.append((input_seq, out_seq, nums, num_pos, data_item.solution, group_nums, data_item.filename))
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

		# for pair in pairs[20:30]:
		# 	print('transfer_num pairs ', pair)
		# 	print()

		#raise Exception('todo')
		return pairs, temp_g, copy_nums

class GraphBuilder(object):
	def __init__(self):
		super(GraphBuilder, self).__init__()

	@staticmethod
	def get_single_batch_graph(input_batch, input_length, group, num_value, num_pos):
		batch_graph = []
		max_len = max(input_length)
		for i in range(len(input_length)):
			input_batch_t = input_batch[i]
			sentence_length = input_length[i]
			quantity_cell_list = group[i]
			num_list = num_value[i]
			id_num_list = num_pos[i]
			graph_attbet = GraphBuilder.get_attribute_between_graph(input_batch_t, max_len, id_num_list, sentence_length, quantity_cell_list)
			# More graphs?
			graph_total = [graph_attbet.tolist()]
			batch_graph.append(graph_total)
		batch_graph = np.array(batch_graph)
		return batch_graph

	@staticmethod
	def get_attribute_between_graph(input_batch, max_len, id_num_list, sentence_length, quantity_cell_list, contain_zh_flag=True):
		diag_ele = np.zeros(max_len)
		for i in range(sentence_length):
			diag_ele[i] = 1
		graph = np.diag(diag_ele)

		if not contain_zh_flag:
			return graph
		for i in id_num_list:
			for j in quantity_cell_list:
				if i < max_len and j < max_len and j not in id_num_list and abs(i-j) < 4:
					graph[i][j] = 1
					graph[j][i] = 1
		for i in quantity_cell_list:
			for j in quantity_cell_list:
				if i < max_len and j < max_len:
					if input_batch[i] == input_batch[j]:
						graph[i][j] = 1
						graph[j][i] = 1
		return graph

def read_files(dataroot, is_test=False):
	all_filenames = sorted(glob.glob(dataroot), key=alphanum_key)
	samples_raw = []
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
		)
		if problem.no_expression:
			skipped += 1
		samples_raw.append(problem)

	print('There are no expressions for {} in {} items'.format(skipped, len(samples_raw)))
	return samples_raw

def prepare_folds(pairs, seed, is_test=False):
	random.seed(seed)
	random.shuffle(pairs)
	folds = []
	for fold_index in range(5):
		train_pairs = [pairs[i] for i in range(len(pairs)) if i % 5 != fold_index]
		test_pairs = [pairs[i] for i in range(len(pairs)) if i % 5 == fold_index]

		folds.append((train_pairs, test_pairs))

	return folds


