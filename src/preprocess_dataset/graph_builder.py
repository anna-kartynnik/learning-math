import numpy as np
import torch
import dgl

from utils.gpu_utils import get_available_device


class GraphBuilder(object):
	def __init__(self):
		super(GraphBuilder, self).__init__()

	@staticmethod
	def get_single_batch_graph(input_batch, input_length, group, num_value, num_pos):
		"""Returns a batch of graphs for the given batch of items."""
		batch_graph = []
		max_len = min(512, max(input_length))
		for i in range(len(input_length)):
			input_batch_t = input_batch[i]
			sentence_length = min(512, input_length[i])
			quantity_cell_list = group[i]
			num_list = num_value[i]
			id_num_list = num_pos[i]
			# We use only one graph.
			graph_attbet = GraphBuilder.get_attribute_between_graph(input_batch_t, max_len, id_num_list, sentence_length, quantity_cell_list)

			batch_graph.append(graph_attbet)

		return dgl.batch(batch_graph)

	@staticmethod
	def get_single_example_graph(input_batch, input_length, group, num_value, num_pos):
		"""Returns graph for one item."""
		batch_graph = []
		max_len = min(512, input_length)
		sentence_length = input_length
		quantity_cell_list = group
		num_list = num_value
		id_num_list = num_pos
		graph_attbet = GraphBuilder.get_attribute_between_graph(input_batch, max_len, id_num_list, sentence_length, quantity_cell_list)

		return dgl.batch([graph_attbet])

	@staticmethod
	def get_attribute_between_graph(input_tokens, max_len, id_num_list, sentence_length, quantity_cell_list):
		"""Uses all the tokens of the sequence for building the graph."""
		quantity_cell_list = [x for x in range(sentence_length)]

		word_cells = set(quantity_cell_list) - set(id_num_list)
		adj_matrix = torch.eye(sentence_length, dtype=torch.bool)
		for w_pos in word_cells:
			for q_pos in id_num_list:
				if abs(w_pos - q_pos) < 4:
					adj_matrix[w_pos, q_pos] = True
					adj_matrix[q_pos, w_pos] = True

		pos_indices = id_num_list
		for index1, pos1 in zip(pos_indices, quantity_cell_list):
			for index2, pos2 in zip(pos_indices, quantity_cell_list):
				if index1 == index2:
					adj_matrix[pos1, pos2] = True
					adj_matrix[pos2, pos1] = True

		src_ids, dst_ids = np.transpose(np.nonzero(adj_matrix))

		graph = dgl.graph((src_ids, dst_ids), num_nodes=max_len, device=get_available_device())

		return graph

