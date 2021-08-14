import math
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter

#import dgl
#from dgl.nn import GraphConv

from utils.gpu_utils import is_gpu_available
from utils.sau_models import *


MAX_OUTPUT_LENGTH = 45

def copy_list(l):
    r = []
    if len(l) == 0:
        return r
    for i in l:
        if type(i) is list:
            r.append(copy_list(i))
        else:
            r.append(i)
    return r

class TreeBeam:  # the class save the beam node
	def __init__(self, score, node_stack, embedding_stack, left_childs, out):
		self.score = score
		self.embedding_stack = copy_list(embedding_stack)
		self.node_stack = copy_list(node_stack)
		self.left_childs = copy_list(left_childs)
		self.out = copy.deepcopy(out)

class TreeEmbedding:  # the class save the tree
	def __init__(self, embedding, terminal=False):
		self.embedding = embedding
		self.terminal = terminal

class EncoderWithGNN(Seq2TreeEncoder):
	def __init__(self, vocab_size, embedding_size, hidden_size):
		#self.gcn = 
		pass

class Graph2TreeModel(nn.Module):
	def __init__(self, encoder_input_size, embedding_size, hidden_size, n_layers, op_nums,
		predictor_input_size, var_nums):
		super(Graph2TreeModel, self).__init__()

		self.encoder = Seq2TreeEncoder(
			vocab_size=encoder_input_size,
			embedding_size=embedding_size,
			hidden_size=hidden_size,
			n_layers=n_layers,
		)
		self.predictor = Seq2TreePrediction(
			vocab_size=predictor_input_size,
			hidden_size=hidden_size,
			op_nums=op_nums - len(var_nums),
		)
		self.generator = Seq2TreeNodeGeneration(
			embedding_size=embedding_size,
			op_nums=op_nums,
			hidden_size=hidden_size,
		)
		self.merger = Seq2TreeSubTreeMerge(
			embedding_size=embedding_size,
			hidden_size=hidden_size,
		)
		self.semantic_alignment = Seq2TreeSemanticAlignment(
			encoder_hidden_size=hidden_size, 
			decoder_hidden_size=hidden_size, 
			hidden_size=hidden_size
		)


	def forward(self, batch, generate_num_ids, output_lang, var_nums=[], batch_first=False):

		input_batch, input_lengths = batch.input_batch, batch.input_lengths
		output_batch, output_lengths = batch.output_batch, batch.output_lengths
		nums_batch, num_size_batch = batch.num_batch, batch.num_size_batch
		nums_stack_batch, num_pos_batch = batch.num_stack_batch, batch.num_pos_batch
		ans_batch = batch.answers_batch

		# sequence mask for attention
		seq_mask = []
		max_len = max(input_lengths)
		for i in input_lengths:
			seq_mask.append([0 for _ in range(i)] + [1 for _ in range(i, max_len)])
		seq_mask = torch.ByteTensor(seq_mask)

		num_mask = []
		max_num_size = max(num_size_batch) + len(generate_num_ids) + len(var_nums)
		for i in num_size_batch:
			d = i + len(generate_num_ids) + len(var_nums)
			num_mask.append([0] * d + [1] * (max_num_size - d))
		num_mask = torch.ByteTensor(num_mask)

		unk = output_lang.word2index["UNK"]

		# Turn padded arrays into (batch_size x max_len) tensors, transpose into (max_len x batch_size)
		input_var = torch.LongTensor(input_batch).transpose(0, 1)

		target = torch.LongTensor(output_batch).transpose(0, 1)
		#batch_graph = torch.LongTensor(batch_graph)

		padding_hidden = torch.FloatTensor([0.0 for _ in range(self.predictor.hidden_size)]).unsqueeze(0)

		batch_size = len(input_lengths)

		self.encoder.train()
		self.predictor.train()
		self.generator.train()
		self.merger.train()
		self.semantic_alignment.train()

		if is_gpu_available():
			input_var = input_var.cuda()
			seq_mask = seq_mask.cuda()
			padding_hidden = padding_hidden.cuda()
			num_mask = num_mask.cuda()
			#batch_graph = batch_graph.cuda()

		encoder_outputs, problem_output = self.encoder(input_var, input_lengths) #, batch_graph)

		# Prepare input and output variables.
		node_stacks = [[TreeNode(_)] for _ in problem_output.split(1, dim=0)]

		max_target_length = max(output_lengths)

		all_node_outputs = []
		all_sa_outputs = []

		copy_num_len = [len(_) for _ in num_pos_batch]
		num_size = max(copy_num_len)
		all_nums_encoder_outputs = self.get_all_number_encoder_outputs(encoder_outputs, num_pos_batch, batch_size, num_size, self.encoder.hidden_size)

		num_start = output_lang.num_start - len(var_nums)
		embeddings_stacks = [[] for _ in range(batch_size)]
		left_childs = [None for _ in range(batch_size)]

		for t in range(max_target_length):
			num_score, op, current_embeddings, current_context, current_nums_embeddings = self.predictor(
				node_stacks, left_childs, encoder_outputs, all_nums_encoder_outputs, padding_hidden, seq_mask, num_mask
			)

			outputs = torch.cat((op, num_score), 1)
			all_node_outputs.append(outputs)

			target_t, generate_input = self.generate_tree_input(target[t].tolist(), outputs, nums_stack_batch, num_start, unk)
			target[t] = target_t
			if is_gpu_available():
				generate_input = generate_input.cuda()
			#print('nums_stack_batch ', nums_stack_batch)
			#print('current_embeddings ', current_embeddings)
			#print('generate_input ', generate_input)
			#print('target_t ', target_t)
			left_child, right_child, node_label = self.generator(current_embeddings, generate_input, current_context)
			left_childs = [] # !!!! already have such variable !!!!

			for idx, l, r, node_stack, i, o in zip(range(batch_size), left_child.split(1), right_child.split(1),
												   node_stacks, target[t].tolist(), embeddings_stacks):
				if len(node_stack) != 0:
					node = node_stack.pop()
				else:
					left_childs.append(None)
					continue

				if i < num_start:
					node_stack.append(TreeNode(r))
					node_stack.append(TreeNode(l, left_flag=True))
					o.append(TreeEmbedding(node_label[idx].unsqueeze(0), terminal=False))
				else:
					current_num = current_nums_embeddings[idx, i - num_start].unsqueeze(0)
					while len(o) > 0 and o[-1].terminal:
						sub_stree = o.pop()
						op = o.pop()
						current_num = self.merger(op.embedding, sub_stree.embedding, current_num)

						if batch_first:
							encoder_mapping, decoder_mapping = self.semantic_alignment(current_num, encoder_outputs[idx])
						else:
							temp_encoder_outputs = encoder_outputs.transpose(0, 1)
							encoder_mapping, decoder_mapping = self.semantic_alignment(current_num, temp_encoder_outputs[idx])
						all_sa_outputs.append((encoder_mapping, decoder_mapping))

					o.append(TreeEmbedding(current_num, terminal=True))
				if len(o) > 0 and o[-1].terminal:
					left_childs.append(o[-1].embedding)
				else:
					left_childs.append(None)

		all_node_outputs = torch.stack(all_node_outputs, dim=1)  # B x S x N

		target = target.transpose(0, 1).contiguous()
		if is_gpu_available():
			all_node_outputs = all_node_outputs.cuda()
			target = target.cuda()
			new_all_sa_outputs = []
			for sa_pair in all_sa_outputs:
				new_all_sa_outputs.append((sa_pair[0].cuda(), sa_pair[1].cuda()))
			all_sa_outputs = new_all_sa_outputs

		semantic_alignment_loss = nn.MSELoss()
		total_semanti_alignment_loss = 0
		sa_len = len(all_sa_outputs)
		for sa_pair in all_sa_outputs:
			total_semanti_alignment_loss += semantic_alignment_loss(sa_pair[0], sa_pair[1])
		# print(total_semanti_alognment_loss)
		total_semanti_alignment_loss = total_semanti_alignment_loss / sa_len
		# print(total_semanti_alognment_loss)

		# op_target = target < num_start
		# loss_0 = masked_cross_entropy_without_logit(all_leafs, op_target.long(), target_length)
		return masked_cross_entropy(all_node_outputs, target, output_lengths) + 0.01 * total_semanti_alignment_loss

	def generate_tree_input(self, target, decoder_output, nums_stack_batch, num_start, unk):
		# when the decoder input is copied num but the num has two pos, chose the max
		target_input = copy.deepcopy(target)

		if is_gpu_available():
			decoder_output = decoder_output.cuda() # .cpu() ???????

		for i in range(len(target)):
			if target[i] == unk:
				num_stack = nums_stack_batch[i].pop()
				max_score = -float("1e12")
				for num in num_stack:
					if decoder_output[i, num_start + num] > max_score:
						target[i] = num + num_start
						max_score = decoder_output[i, num_start + num]
			if target_input[i] >= num_start:
				target_input[i] = 0
		return torch.LongTensor(target), torch.LongTensor(target_input)

	def get_all_number_encoder_outputs(self, encoder_outputs, num_pos, batch_size, num_size, hidden_size, batch_first=False):
		indices = list()

		if batch_first:
			sen_len = encoder_outputs.size(1)
		else:
			sen_len = encoder_outputs.size(0)
		
		masked_index = []
		temp_1 = [1 for _ in range(hidden_size)]
		temp_0 = [0 for _ in range(hidden_size)]
		for b in range(batch_size):
			for i in num_pos[b]:
				indices.append(i + b * sen_len)
				masked_index.append(temp_0)
			indices += [0 for _ in range(len(num_pos[b]), num_size)]
			masked_index += [temp_1 for _ in range(len(num_pos[b]), num_size)]
		indices = torch.LongTensor(indices)
		masked_index = torch.ByteTensor(masked_index)
		masked_index = masked_index.view(batch_size, num_size, hidden_size)
		if is_gpu_available():
			indices = indices.cuda()
			masked_index = masked_index.cuda()

		if batch_first:
			all_outputs = encoder_outputs.contiguous() # B x S x H
		else:
			all_outputs = encoder_outputs.transpose(0, 1).contiguous()  # S x B x H

		all_embedding = all_outputs.view(-1, encoder_outputs.size(2))  # S x B x H -> (B x S) x H
		all_num = all_embedding.index_select(0, indices)
		all_num = all_num.view(batch_size, num_size, hidden_size)
		return all_num.masked_fill_(masked_index.bool(), 0.0)

	def evaluate_tree(self, item, generate_nums, output_lang,
					  beam_size=5, var_nums=[], beam_search=True, max_length=MAX_OUTPUT_LENGTH):
		input_batch, input_length = item.input_tokens, len(item.input_tokens)
		num_pos = item.num_pos
		# sequence mask for attention
		seq_mask = torch.ByteTensor(1, input_length).fill_(0)
		# Turn padded arrays into (batch_size x max_len) tensors, transpose into (max_len x batch_size)
		input_var = torch.LongTensor(input_batch).unsqueeze(1)

		num_mask = torch.ByteTensor(1, len(num_pos) + len(generate_nums)+ len(var_nums)).fill_(0)

		# Set to not-training mode to disable dropout
		self.encoder.eval()
		self.predictor.eval()
		self.generator.eval()
		self.merger.eval()

		padding_hidden = torch.FloatTensor([0.0 for _ in range(self.predictor.hidden_size)]).unsqueeze(0)

		batch_size = 1

		if is_gpu_available():
			input_var = input_var.cuda()
			seq_mask = seq_mask.cuda()
			padding_hidden = padding_hidden.cuda()
			num_mask = num_mask.cuda()

		# Run words through encoder
		encoder_outputs, problem_output = self.encoder(input_var, [input_length])

		# Prepare input and output variables  # # root embedding B x 1
		node_stacks = [[TreeNode(_)] for _ in problem_output.split(1, dim=0)]

		num_size = len(num_pos)
		# 提取与问题相关的数字embedding
		all_nums_encoder_outputs = self.get_all_number_encoder_outputs(encoder_outputs, [num_pos], batch_size, num_size,
																  self.encoder.hidden_size)
		num_start = output_lang.num_start - len(var_nums)
		# B x P x N
		embeddings_stacks = [[] for _ in range(batch_size)]
		left_childs = [None for _ in range(batch_size)]

		if beam_search:
			beams = [TreeBeam(0.0, node_stacks, embeddings_stacks, left_childs, [])]

			for t in range(max_length):
				current_beams = []
				while len(beams) > 0:
					b = beams.pop()
					if len(b.node_stack[0]) == 0:
						current_beams.append(b)
						continue
					# left_childs = torch.stack(b.left_childs)
					left_childs = b.left_childs

					num_score, op, current_embeddings, current_context, current_nums_embeddings = self.predictor(
						b.node_stack, left_childs, encoder_outputs, all_nums_encoder_outputs, padding_hidden,
						seq_mask, num_mask)


					out_score = nn.functional.log_softmax(torch.cat((op, num_score), dim=1), dim=1)

					# out_score = p_leaf * out_score

					topv, topi = out_score.topk(beam_size)


					for tv, ti in zip(topv.split(1, dim=1), topi.split(1, dim=1)):
						current_node_stack = copy_list(b.node_stack)
						current_left_childs = []
						current_embeddings_stacks = copy_list(b.embedding_stack)
						current_out = copy.deepcopy(b.out)
						out_token = int(ti)
						current_out.append(out_token)

						node = current_node_stack[0].pop()

						# var_num当时数字处理，SEP/;当操作符处理
						if out_token < num_start: # 非数字
							generate_input = torch.LongTensor([out_token])
							if is_gpu_available():
								generate_input = generate_input.cuda()
							left_child, right_child, node_label = self.generator(current_embeddings, generate_input, current_context)

							current_node_stack[0].append(TreeNode(right_child))
							current_node_stack[0].append(TreeNode(left_child, left_flag=True))

							current_embeddings_stacks[0].append(TreeEmbedding(node_label[0].unsqueeze(0), False))
						else:  # 数字
							current_num = current_nums_embeddings[0, out_token - num_start].unsqueeze(0)

							while len(current_embeddings_stacks[0]) > 0 and current_embeddings_stacks[0][-1].terminal:
								sub_stree = current_embeddings_stacks[0].pop()
								op = current_embeddings_stacks[0].pop()
								current_num = self.merger(op.embedding, sub_stree.embedding, current_num)
							current_embeddings_stacks[0].append(TreeEmbedding(current_num, True))
						if len(current_embeddings_stacks[0]) > 0 and current_embeddings_stacks[0][-1].terminal:
							current_left_childs.append(current_embeddings_stacks[0][-1].embedding)
						else:
							current_left_childs.append(None)
						current_beams.append(TreeBeam(b.score+float(tv), current_node_stack, current_embeddings_stacks,
													  current_left_childs, current_out))
				beams = sorted(current_beams, key=lambda x: x.score, reverse=True)
				beams = beams[:beam_size]
				flag = True
				for b in beams:
					if len(b.node_stack[0]) != 0:
						flag = False
				if flag:
					break

			return beams[0].out
		else:
			all_node_outputs = []
			for t in range(max_length):
				num_score, op, current_embeddings, current_context, current_nums_embeddings = self.predictor(
					node_stacks, left_childs, encoder_outputs, all_nums_encoder_outputs, padding_hidden,
					seq_mask, num_mask)

				out_scores = nn.functional.log_softmax(torch.cat((op, num_score), dim=1), dim=1)
				out_tokens = torch.argmax(out_scores, dim=1) # B
				all_node_outputs.append(out_tokens)
				left_childs = []
				for idx, node_stack, out_token, embeddings_stack in zip(range(batch_size), node_stacks, out_tokens, embeddings_stacks):
					# node = node_stack.pop()
					if len(node_stack) != 0:
						node = node_stack.pop()
					else:
						left_childs.append(None)
						continue
					# var_num当时数字处理，SEP/;当操作符处理
					if out_token < num_start: # 非数字
						generate_input = torch.LongTensor([out_token])
						if is_gpu_available():
							generate_input = generate_input.cuda()
						left_child, right_child, node_label = self.generator(current_embeddings, generate_input, current_context)
						node_stack.append(TreeNode(right_child))
						node_stack.append(TreeNode(left_child, left_flag=True))
						embeddings_stack.append(TreeEmbedding(node_label.unsqueeze(0), False))
					else: # 数字
						current_num = current_nums_embeddings[idx, out_token - num_start].unsqueeze(0)
						while len(embeddings_stack) > 0 and embeddings_stack[-1].terminal:
							sub_stree = embeddings_stack.pop()
							op = embeddings_stack.pop()
							current_num = self.merger(op.embedding.squeeze(0), sub_stree.embedding, current_num)
						embeddings_stack.append(TreeEmbedding(current_num, terminal=True))

					if len(embeddings_stack) > 0 and embeddings_stack[-1].terminal:
						left_childs.append(embeddings_stack[-1].embedding)
					else:
						left_childs.append(None)

			# all_leafs = torch.stack(all_leafs, dim=1)  # B x S x 2
			all_node_outputs = torch.stack(all_node_outputs, dim=1)  # B x S x N
			all_node_outputs = all_node_outputs.cpu().numpy()
			return all_node_outputs[0]



def sequence_mask(sequence_length, max_len=None):
	if max_len is None:
		max_len = sequence_length.data.max()
	batch_size = sequence_length.size(0)
	seq_range = torch.arange(0, max_len).long()
	seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_len)
	if sequence_length.is_cuda:
		seq_range_expand = seq_range_expand.cuda()
	seq_length_expand = (sequence_length.unsqueeze(1).expand_as(seq_range_expand))
	return seq_range_expand < seq_length_expand

def masked_cross_entropy(logits, target, length):
	if torch.cuda.is_available():
		length = torch.LongTensor(length).cuda()
	else:
		length = torch.LongTensor(length)
	"""
	Args:
		logits: A Variable containing a FloatTensor of size
			(batch, max_len, num_classes) which contains the
			unnormalized probability for each class.
		target: A Variable containing a LongTensor of size
			(batch, max_len) which contains the index of the true
			class for each corresponding step.
		length: A Variable containing a LongTensor of size (batch,)
			which contains the length of each data in a batch.
	Returns:
		loss: An average loss value masked by the length.
	"""

	# logits_flat: (batch * max_len, num_classes)
	logits_flat = logits.view(-1, logits.size(-1))
	# log_probs_flat: (batch * max_len, num_classes)
	log_probs_flat = nn.functional.log_softmax(logits_flat, dim=1)
	# target_flat: (batch * max_len, 1)
	target_flat = target.view(-1, 1)
	# losses_flat: (batch * max_len, 1)
	losses_flat = -torch.gather(log_probs_flat, dim=1, index=target_flat)

	# losses: (batch, max_len)
	losses = losses_flat.view(*target.size())
	# mask: (batch, max_len)
	mask = sequence_mask(sequence_length=length, max_len=target.size(1))
	losses = losses * mask.float()
	loss = losses.sum() / length.float().sum()

	return loss
