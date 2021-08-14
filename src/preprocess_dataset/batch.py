import random
import copy

from .data_utils import pad_seq
from .graph_builder import GraphBuilder


class MATHBatchItem(object):
	def __init__(self, input_seq, output_seq, nums, num_pos, num_stack, group_nums, answers=[], filename=None):
		self.input_seq = input_seq
		self.output_seq = output_seq
		self.nums = nums
		self.num_pos = num_pos
		self.num_stack = num_stack
		self.group_nums = group_nums
		self.answers = answers
		self.filename = filename

		# Will be initialized later.
		self.input_tokens = []
		self.output_tokens = []
		self.graph = None

	def build_graph(self):
		assert len(self.input_tokens) > 0

		self.graph = GraphBuilder.get_single_batch_graph(
			[self.input_tokens],
			[len(self.input_tokens)],
			[self.group_nums],
			[self.nums],
			[self.num_pos]
		)

	def __repr__(self):
		return 'input_seq: {}, input_tokens: {}, output_seq: {}, output_tokens: {}, nums: {}, num_pos: {}, \
			num_stack: {}, group_nums: {}, answers: {}, filename: {}'.format(
				self.input_seq,
				self.input_tokens,
				self.output_seq,
				self.output_tokens,
				self.nums,
				self.num_pos,
				self.num_stack,
				self.group_nums,
				self.answers,
				self.filename
			)


class MATHBatch(object):
	def __init__(self, input_batch, input_lengths, output_batch, output_lengths, num_batch,
		num_stack_batch, num_pos_batch, num_size_batch, group_batch, graph_batch, answers_batch):
		self.input_batch = input_batch
		self.input_lengths = input_lengths
		self.output_batch = output_batch
		self.output_lengths = output_lengths
		self.num_batch = num_batch
		self.num_stack_batch = num_stack_batch
		self.num_pos_batch = num_pos_batch
		self.num_size_batch = num_size_batch
		self.answers_batch = answers_batch
		self.group_batch = group_batch
		self.graph_batch = graph_batch


	@staticmethod
	def create_from_items(items, batch_size):
		# items is a List of MATHBatchItem elements
		items = copy.deepcopy(items)
		random.shuffle(items)
	
		pos = 0
		splitted_items = []
		while pos + batch_size < len(items):
			splitted_items.append(items[pos : pos + batch_size])
			pos += batch_size
		splitted_items.append(items[pos:])

		batches = []
		for batch in splitted_items:
			# Sort batch elements by input sequence length (in descending order).
			batch = sorted(batch, key=lambda item: len(item.input_seq), reverse=True)
			
			input_lengths = []
			output_lengths = []
			for batch_item in batch:
				input_lengths.append(len(batch_item.input_tokens))
				output_lengths.append(len(batch_item.output_tokens))
			input_len_max = input_lengths[0]
			output_len_max = max(output_lengths)

			input_batch = []
			output_batch = []
			num_batch = []
			num_pos_batch = []
			num_stack_batch = []
			num_size_batch = []
			num_value_batch = []
			group_batch = []
			ans_batch = []
			for batch_item in batch:
				input_batch.append(pad_seq(batch_item.input_tokens, len(batch_item.input_tokens), input_len_max))
				output_batch.append(pad_seq(batch_item.output_tokens, len(batch_item.output_tokens), output_len_max))
				num_batch.append(len(batch_item.nums))
				num_pos_batch.append(batch_item.num_pos)
				num_stack_batch.append(batch_item.num_stack)
				num_size_batch.append(len(batch_item.num_pos))
				num_value_batch.append(batch_item.nums)
				group_batch.append(batch_item.group_nums)
				ans_batch.append(batch_item.answers)

			batches.append(
				MATHBatch(
					input_batch,
					input_lengths,
					output_batch,
					output_lengths,
					num_batch,
					num_stack_batch,
					num_pos_batch,
					num_size_batch,
					group_batch,
					GraphBuilder.get_single_batch_graph(
						input_batch, input_lengths, group_batch, num_value_batch, num_pos_batch
					),
					ans_batch
				)
			)

		return batches

		# 	# input_lengths.append(input_length)
		# 	# output_lengths.append(output_length)
		# 	# input_len_max = input_length[0]
		# 	# output_len_max = max(output_length)
		# 	# input_batch = []
		# 	# output_batch = []
		# 	# num_batch = []
		# 	# num_stack_batch = []
		# 	# num_pos_batch = []
		# 	# num_size_batch = []
		# 	# group_batch = []
		# 	# num_value_batch = []
		# 	# ans_batch = []
		# 	for batch_item in batch:
		# 		if ans_flag:
		# 			i, li, j, lj, num, num_pos, ans, num_stack, group = batch_item
		# 			ans_batch.append(ans)
		# 		else:
		# 			i, li, j, lj, num, num_pos, num_stack, group = batch_item
		# 		num_batch.append(len(num))
		# 		input_batch.append(self.pad_seq(i, li, input_len_max))
		# 		output_batch.append(self.pad_seq(j, lj, output_len_max))
		# 		num_stack_batch.append(num_stack)
		# 		num_pos_batch.append(num_pos)
		# 		num_size_batch.append(len(num_pos))
		# 		num_value_batch.append(num)
		# 		group_batch.append(group)

		# 	item = [
		# 		input_batch,
		# 		input_length,
		# 		output_batch,
		# 		output_length,
		# 		num_batch,
		# 		num_stack_batch,
		# 		num_pos_batch,
		# 		num_size_batch,
		# 		# num_value_batch,
		# 		# GraphBuilder.get_single_batch_graph(
		# 		# 	input_batch, input_length, group_batch, num_value_batch, num_pos_batch
		# 		# )
		# 	]
		# 	if ans_flag:
		# 		item.append(ans_batch)
		# 	batches.append(tuple(item))


		# input_lengths = []
		# output_lengths = []
		# nums_batches = []
		# batches = []
		# batched_pairs = []
		# input_batches = []
		# output_batches = []
		# num_stack_batches = []  # save the num stack which
		# num_pos_batches = []
		# num_size_batches = []
		# group_batches = []
		# graph_batches = []
		# num_value_batches = []

		# ans_batches = []
		# ans_flag = False if len(pairs[0]) == 8 else True



		# for batched_pair in batched_pairs:
		# 	# Sort batch elements by input sequence length (in descending order).
		# 	batch = sorted(batched_pair, key=lambda tp: tp[1], reverse=True)
		# 	input_length = []
		# 	output_length = []
		# 	for batch_item in batch:
		# 		if ans_flag:
		# 			_, i, _, j, _, _, _, _, _ = batch_item
		# 		else:
		# 			_, i, _, j, _, _, _, _ = batch_item
		# 		input_length.append(i)
		# 		output_length.append(j)
		# 	input_lengths.append(input_length)
		# 	output_lengths.append(output_length)
		# 	input_len_max = input_length[0]
		# 	output_len_max = max(output_length)
		# 	input_batch = []
		# 	output_batch = []
		# 	num_batch = []
		# 	num_stack_batch = []
		# 	num_pos_batch = []
		# 	num_size_batch = []
		# 	group_batch = []
		# 	num_value_batch = []
		# 	ans_batch = []
		# 	for batch_item in batch:
		# 		if ans_flag:
		# 			i, li, j, lj, num, num_pos, ans, num_stack, group = batch_item
		# 			ans_batch.append(ans)
		# 		else:
		# 			i, li, j, lj, num, num_pos, num_stack, group = batch_item
		# 		num_batch.append(len(num))
		# 		input_batch.append(self.pad_seq(i, li, input_len_max))
		# 		output_batch.append(self.pad_seq(j, lj, output_len_max))
		# 		num_stack_batch.append(num_stack)
		# 		num_pos_batch.append(num_pos)
		# 		num_size_batch.append(len(num_pos))
		# 		num_value_batch.append(num)
		# 		group_batch.append(group)

		# 	item = [
		# 		input_batch,
		# 		input_length,
		# 		output_batch,
		# 		output_length,
		# 		num_batch,
		# 		num_stack_batch,
		# 		num_pos_batch,
		# 		num_size_batch,
		# 		# num_value_batch,
		# 		# GraphBuilder.get_single_batch_graph(
		# 		# 	input_batch, input_length, group_batch, num_value_batch, num_pos_batch
		# 		# )
		# 	]
		# 	if ans_flag:
		# 		item.append(ans_batch)
		# 	batches.append(tuple(item))
				
		# return batches