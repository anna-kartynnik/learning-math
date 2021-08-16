"""
Borrowed with modifications from: https://github.com/QinJinghui/SAU-Solver
"""


import torch
import torch.nn as nn


class TreeNode: # the class save the tree node
    def __init__(self, embedding, left_flag=False):
        self.embedding = embedding
        self.left_flag = left_flag


# Attention:
class Attn(nn.Module):
    def __init__(self, hidden_size, batch_first=False, bidirectional_encoder=True):
        super(Attn, self).__init__()
        self.hidden_size = hidden_size
        self.batch_first = batch_first
        self.bidirectional_encoder = bidirectional_encoder
        if self.bidirectional_encoder:
            self.attn = nn.Linear(hidden_size * 2, hidden_size)
        else:
            self.attn = nn.Linear(hidden_size, hidden_size)
        self.score = nn.Linear(hidden_size, 1, bias=False)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, hidden, encoder_outputs, seq_mask=None):
        if self.batch_first:  # B x S x H
            max_len = encoder_outputs.size(1)
            repeat_dims = [1] * hidden.dim()
            repeat_dims[1] = max_len
        else:  # S x B x H
            max_len = encoder_outputs.size(0)
            repeat_dims = [1] * hidden.dim()
            repeat_dims[0] = max_len
        # batch_first: False S x B x H
        # batch_first: True B x S x H
        hidden = hidden.repeat(*repeat_dims)  # Repeats this tensor along the specified dimensions

        # For each position of encoder outputs
        if self.batch_first:
            batch_size = encoder_outputs.size(0)
        else:
            batch_size = encoder_outputs.size(1)
        # (B x S) x (2 x H) or (S x B) x (2 x H)
        energy_in = torch.cat((hidden, encoder_outputs), 2).view(-1, 2 * self.hidden_size)
        attn_energies = self.score(torch.tanh(self.attn(energy_in)))  # (S x B) x 1 or (B x S) x 1
        attn_energies = attn_energies.squeeze(1)  # (S x B) or (B x S)
        if self.batch_first:
            attn_energies = attn_energies.view(batch_size, max_len)  # B x S
        else:
            attn_energies = attn_energies.view(max_len, batch_size).transpose(0, 1)  # B x S
        if seq_mask is not None:
            attn_energies = attn_energies.masked_fill_(seq_mask.bool(), -1e12)
        # Normalize energies to weights in range 0 to 1, resize to B x 1 x S
        attn_energies = self.softmax(attn_energies)
        return attn_energies.unsqueeze(1)

class Score(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Score, self).__init__()
        self.input_size = input_size  # goal vector size
        self.hidden_size = hidden_size
        self.attn = nn.Linear(hidden_size + input_size, hidden_size)
        self.score = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, hidden, num_embeddings, num_mask=None):
        # hidden: B x 1 x H; num_embeddings: B x O x H
        max_len = num_embeddings.size(1)
        repeat_dims = [1] * hidden.dim()
        repeat_dims[1] = max_len
        hidden = hidden.repeat(*repeat_dims)  # B x O x H
        # For each position of encoder outputs
        batch_size = num_embeddings.size(0)
        energy_in = torch.cat((hidden, num_embeddings), 2).view(-1, self.input_size + self.hidden_size)
        score = self.score(torch.tanh(self.attn(energy_in)))  # (B x O) x 1
        score = score.squeeze(1)
        score = score.view(batch_size, -1)  # B x O
        if num_mask is not None:
            score = score.masked_fill_(num_mask.bool(), -1e12)
        return score

class TreeAttn(nn.Module):
    def __init__(self, input_size, hidden_size, batch_first=False):
        super(TreeAttn, self).__init__()
        self.input_size = input_size # goal vector size
        self.hidden_size = hidden_size
        self.batch_first = batch_first
        self.attn = nn.Linear(hidden_size + input_size, hidden_size)
        self.score = nn.Linear(hidden_size, 1)

    def forward(self, hidden, encoder_outputs, seq_mask=None):
        if self.batch_first:
            # encoder_outputs: S x B x H
            max_len = encoder_outputs.size(1)
            # hidden:  B x 1 x H
            repeat_dims = [1] * hidden.dim()
            repeat_dims[1] = max_len
            batch_size = encoder_outputs.size(0)
        else:
            # encoder_outputs: S x B x H
            max_len = encoder_outputs.size(0)
            # hidden: 1 x B x H
            repeat_dims = [1] * hidden.dim()
            repeat_dims[0] = max_len
            batch_size = encoder_outputs.size(1)
        hidden = hidden.repeat(*repeat_dims)  # S x B x H or B x S x H

        energy_in = torch.cat((hidden, encoder_outputs), 2).view(-1, self.input_size + self.hidden_size) # SBx2H or BSx2H

        score_feature = torch.tanh(self.attn(energy_in))  # SBxH or BSxH
        attn_energies = self.score(score_feature)  # (S x B) x 1
        attn_energies = attn_energies.squeeze(1)
        if self.batch_first:
            attn_energies = attn_energies.view(batch_size, max_len) # B x S
        else:
            attn_energies = attn_energies.view(max_len, batch_size).transpose(0, 1)  # B x S
        if seq_mask is not None:
            attn_energies = attn_energies.masked_fill_(seq_mask.bool(), -1e12)
        attn_energies = nn.functional.softmax(attn_energies, dim=1)  # B x S

        return attn_energies.unsqueeze(1) # B x 1 x S

class Seq2TreePrediction(nn.Module):
    # a seq2tree decoder with Problem aware dynamic encoding
    def __init__(self, hidden_size, op_nums, vocab_size, dropout=0.5):
        super(Seq2TreePrediction, self).__init__()
        # Keep for reference
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.op_nums = op_nums

        self.dropout = nn.Dropout(dropout)

        self.embedding_weight = nn.Parameter(torch.randn(1, vocab_size, hidden_size))

        # for Computational symbols and Generated numbers
        self.concat_l = nn.Linear(hidden_size, hidden_size)  # left inner symbols generation
        self.concat_r = nn.Linear(hidden_size * 2, hidden_size)  # right inner symbols generation
        self.concat_lg = nn.Linear(hidden_size, hidden_size)   # left number generation
        self.concat_rg = nn.Linear(hidden_size * 2, hidden_size)  # right number generation

        self.ops = nn.Linear(hidden_size * 2, op_nums)

        self.attn = TreeAttn(hidden_size, hidden_size)
        self.score = Score(hidden_size * 2, hidden_size)

    def forward(self, node_stacks, left_children, encoder_outputs, padded_nums, padded_hidden, seq_mask, num_mask):
        current_embeddings = []
        # node_stacks: B
        # padded_hidden: B x 2H
        for st in node_stacks:
            if len(st) == 0:
                current_embeddings.append(padded_hidden)
            else:
                current_node = st[-1]
                current_embeddings.append(current_node.embedding)

        current_node_temp = []
        for l, c in zip(left_children, current_embeddings):
            if l is None:
                c = self.dropout(c)
                g = torch.tanh(self.concat_l(c))
                t = torch.sigmoid(self.concat_lg(c))
                current_node_temp.append(g * t)
            else:
                ld = self.dropout(l)
                c = self.dropout(c)
                g = torch.tanh(self.concat_r(torch.cat((ld, c), 1)))
                t = torch.sigmoid(self.concat_rg(torch.cat((ld, c), 1)))
                current_node_temp.append(g * t)

        current_node = torch.stack(current_node_temp)
        current_embeddings = self.dropout(current_node)

        current_attn = self.attn(current_embeddings.transpose(0, 1), encoder_outputs, seq_mask)
        current_context = current_attn.bmm(encoder_outputs.transpose(0, 1))  # B x 1 x N

        # the information to get the current quantity
        batch_size = current_embeddings.size(0)
        # predict the output (this node corresponding to output(number or operator)) with PADE

        repeat_dims = [1] * self.embedding_weight.dim()
        repeat_dims[0] = batch_size
        embedding_weight = self.embedding_weight.repeat(*repeat_dims)  # B x input_size x N
        embedding_weight = torch.cat((embedding_weight, padded_nums), dim=1)  # B x O x N

        leaf_input = torch.cat((current_node, current_context), 2)
        leaf_input = leaf_input.squeeze(1)
        leaf_input = self.dropout(leaf_input)

        # max pooling the embedding_weight
        embedding_weight_ = self.dropout(embedding_weight)
        num_score = self.score(leaf_input.unsqueeze(1), embedding_weight_, num_mask)

        op = self.ops(leaf_input)

        return num_score, op, current_node, current_context, embedding_weight

class Seq2TreeNodeGeneration(nn.Module):
    def __init__(self, hidden_size, op_nums, embedding_size, dropout=0.5):
        super(Seq2TreeNodeGeneration, self).__init__()
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size

        self.embeddings = nn.Embedding(op_nums, embedding_size)
        self.em_dropout = nn.Dropout(dropout)
        self.generate_l = nn.Linear(hidden_size * 2 + embedding_size, hidden_size)
        self.generate_r = nn.Linear(hidden_size * 2 + embedding_size, hidden_size)
        self.generate_lg = nn.Linear(hidden_size * 2 + embedding_size, hidden_size)
        self.generate_rg = nn.Linear(hidden_size * 2 + embedding_size, hidden_size)

    def forward(self, node_embedding, node_label, current_context):
        node_label_ = self.embeddings(node_label)
        node_label = self.em_dropout(node_label_)
        node_embedding = node_embedding.squeeze(1)
        current_context = current_context.squeeze(1)
        node_embedding = self.em_dropout(node_embedding)
        current_context = self.em_dropout(current_context)

        l_child = torch.tanh(self.generate_l(torch.cat((node_embedding, current_context, node_label), 1)))
        l_child_g = torch.sigmoid(self.generate_lg(torch.cat((node_embedding, current_context, node_label), 1)))
        r_child = torch.tanh(self.generate_r(torch.cat((node_embedding, current_context, node_label), 1)))
        r_child_g = torch.sigmoid(self.generate_rg(torch.cat((node_embedding, current_context, node_label), 1)))
        l_child = l_child * l_child_g
        r_child = r_child * r_child_g
        return l_child, r_child, node_label_
#!
class Seq2TreeSubTreeMerge(nn.Module):
    def __init__(self, hidden_size, embedding_size, dropout=0.5):
        super(Seq2TreeSubTreeMerge, self).__init__()

        self.embedding_size = embedding_size
        self.hidden_size = hidden_size

        self.em_dropout = nn.Dropout(dropout)
        self.merge = nn.Linear(hidden_size * 2 + embedding_size, hidden_size)
        self.merge_g = nn.Linear(hidden_size * 2 + embedding_size, hidden_size)

    def forward(self, node_embedding, sub_tree_1, sub_tree_2):
        sub_tree_1 = self.em_dropout(sub_tree_1)
        sub_tree_2 = self.em_dropout(sub_tree_2)
        node_embedding = self.em_dropout(node_embedding)

        sub_tree = torch.tanh(self.merge(torch.cat((node_embedding, sub_tree_1, sub_tree_2), 1)))
        sub_tree_g = torch.sigmoid(self.merge_g(torch.cat((node_embedding, sub_tree_1, sub_tree_2), 1)))
        sub_tree = sub_tree * sub_tree_g
        return sub_tree

class Seq2TreeSemanticAlignment(nn.Module):
    def __init__(self, encoder_hidden_size, decoder_hidden_size, hidden_size, batch_first=False, bidirectional_encoder=True):
        super(Seq2TreeSemanticAlignment, self).__init__()
        self.batch_first = batch_first
        self.attn = Attn(encoder_hidden_size,batch_first=batch_first,bidirectional_encoder=bidirectional_encoder)
        self.encoder_linear1 = nn.Linear(encoder_hidden_size, hidden_size)
        self.encoder_linear2 = nn.Linear(hidden_size, hidden_size)

        self.decoder_linear1 = nn.Linear(decoder_hidden_size, hidden_size)
        self.decoder_linear2 = nn.Linear(hidden_size, hidden_size)

    def forward(self,  decoder_hidden, encoder_outputs):
        if self.batch_first:
            decoder_hidden = decoder_hidden.unsqueeze(0)
            encoder_outputs = encoder_outputs.unsqueeze(0)
        else:
            decoder_hidden = decoder_hidden.unsqueeze(0)
            encoder_outputs = encoder_outputs.unsqueeze(1)
        attn_weights = self.attn(decoder_hidden, encoder_outputs, None)
        if self.batch_first:
            align_context = attn_weights.bmm(encoder_outputs) # B x 1 x H
        else:
            align_context = attn_weights.bmm(encoder_outputs.transpose(0, 1))  # B x 1 x H
            align_context = align_context.transpose(0,1)

        encoder_linear1 = torch.tanh(self.encoder_linear1(align_context))
        encoder_linear2 = self.encoder_linear2(encoder_linear1)

        decoder_linear1 = torch.tanh(self.decoder_linear1(decoder_hidden))
        decoder_linear2 = self.decoder_linear2(decoder_linear1)

        return encoder_linear2, decoder_linear2


