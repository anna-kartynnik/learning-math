import numpy as np
#import networkx as nx
import dgl

from utils.gpu_utils import get_available_device


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
			#graph_total = [graph_attbet]
			batch_graph.append(graph_attbet)
		#batch_graph = np.array(batch_graph)
		return dgl.batch(batch_graph) #batch_graph

	@staticmethod
	def get_attribute_between_graph(input_batch, max_len, id_num_list, sentence_length, quantity_cell_list, contain_zh_flag=True):
		# diag_ele = np.zeros(max_len)
		# for i in range(sentence_length):
		# 	diag_ele[i] = 1
		# graph = np.diag(diag_ele)

		# if not contain_zh_flag:
		# 	return graph
		# for i in id_num_list:
		# 	for j in quantity_cell_list:
		# 		if i < max_len and j < max_len and j not in id_num_list and abs(i-j) < 4:
		# 			graph[i][j] = 1
		# 			graph[j][i] = 1
		# for i in quantity_cell_list:
		# 	for j in quantity_cell_list:
		# 		if i < max_len and j < max_len:
		# 			if input_batch[i] == input_batch[j]:
		# 				graph[i][j] = 1
		# 				graph[j][i] = 1
		# return GraphBuilder.get_dgl_graph(graph)

		src_ids = []
		dst_ids = []
		for i in id_num_list:
			for j in quantity_cell_list:
				if i < max_len and j < max_len and j not in id_num_list and abs(i-j) < 4:
					src_ids.append(i)
					#src_ids.append(j)
					dst_ids.append(j)
					#dst_ids.append(i)
		for i in quantity_cell_list:
			for j in quantity_cell_list:
				if i < max_len and j < max_len:
					if input_batch[i] == input_batch[j]:
						src_ids.append(i)
						#src_ids.append(j)
						dst_ids.append(j)
						#dst_ids.append(i)
		#print(src_ids)
		#print(dst_ids)
		graph = dgl.graph((src_ids, dst_ids), num_nodes=max_len, device=get_available_device())
		graph = dgl.add_self_loop(graph)
		return graph




		# # Source nodes for edges (2, 1), (3, 2), (4, 3)
		# src_ids = torch.tensor([2, 3, 4])
		# # Destination nodes for edges (2, 1), (3, 2), (4, 3)
		# dst_ids = torch.tensor([1, 2, 3])
		# g = dgl.graph((src_ids, dst_ids))

