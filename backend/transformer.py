# transformer.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from torch import optim
from tokenizer import *
import pickle

class SelfAttention(nn.Module):
    def __init__(self, d_model):
        super(SelfAttention, self).__init__()
        self.d_model = d_model
        self.Wq = nn.Linear(d_model, d_model)
        self.Wk = nn.Linear(d_model, d_model)
        self.Wv = nn.Linear(d_model, d_model)
    
    def forward(self, x):
        # x: tensor of shape 20 x  d_model
        q = self.Wq(x)
        k = self.Wk(x)
        v = self.Wv(x)
        # q k v = 20 x d_model
        scores = torch.matmul(q, k.transpose(0, 1)) / (self.d_model ** .5)
        # Apply an upper triangular mask to prevent the model from attending to positions after the current one
        mask = torch.triu(torch.ones(scores.shape), diagonal=1)
        scores = scores.masked_fill(mask == 1, -1e9)
        attention_weights = nn.Softmax(dim=-1)(scores)
        attended_values = torch.matmul(attention_weights, v)
        return attention_weights, attended_values

# Should contain your overall Transformer implementation. You will want to use Transformer layer to implement
# a single layer of the Transformer; this Module will take the raw words as input and do all of the steps necessary
# to return distributions over the labels (0, 1, or 2).
class Transformer(nn.Module):
    def __init__(self, vocab_size, num_positions, d_model, d_internal, num_classes, num_layers, num_heads=4):
        """
        :param vocab_size: vocabulary size of the embedding layer
        :param num_positions: max sequence length that will be fed to the model; should be 20
        :param d_model: see TransformerLayer
        :param d_internal: see TransformerLayer
        :param num_classes: number of classes predicted at the output layer; should be 3
        :param num_layers: number of TransformerLayers to use; can be whatever you want
        """
        super().__init__()
        self.num_layers = num_layers
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, num_positions)
        self.transformer_layers = nn.ModuleList([TransformerLayer(d_model, d_internal) for _ in range(num_layers)])
        self.linear = nn.Linear(d_model, num_classes)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, indices):
        """
        :param indices: list of input indices
        :return: A tuple of the softmax log probabilities (should be a 20x3 matrix) and a list of the attention
        maps you use in your layers (can be variable length, but each should be a 20x20 matrix)
        """
        # input: [batch_size, seq_len]
        input = self.embedding(indices)
        input = self.positional_encoding(input)
        attn_maps = []
        for i in range(self.num_layers):
            input, attention = self.transformer_layers[i](input)
            attn_maps.append(attention)
        # output: [batch_size, seq_len, num_classes]
        output = self.linear(input)
        # output: [batch_size, seq_len, num_classes]
        output = self.softmax(output)
        return output, attn_maps


# Your implementation of the Transformer layer goes here. It should take vectors and return the same number of vectors
# of the same length, applying self-attention, the feedforward layer, etc.
class TransformerLayer(nn.Module):
    def __init__(self, d_model, d_internal):
        """
        :param d_model: The dimension of the inputs and outputs of the layer (note that the inputs and outputs
        have to be the same size for the residual connection to work)
        :param d_internal: The "internal" dimension used in the self-attention computation. Your keys and queries
        should both be of this length.
        """
        super().__init__()
        self.d_model = d_model
        self.d_internal = d_internal
        self.self_attention = SelfAttention(d_model)
        self.feedforward = nn.Sequential(
            nn.Linear(d_model, d_internal),
            nn.ReLU(),
            nn.Linear(d_internal, d_model)
        )

    def forward(self, input_vecs):
        """
        :param input_vecs: [batch_size, seq_len, d_model]
        :return: [batch_size, seq_len, d_model]
        """
        # output: [batch_size, seq_len, d_model]
        attended_weights, attended_values = self.self_attention(input_vecs)
        # output: [batch_size, seq_len, d_model]
        x = input_vecs + attended_values
        ff_output = self.feedforward(x)
        x = x + ff_output
        return x, attended_weights


# Implementation of positional encoding that you can use in your network
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, num_positions: int=20, batched=False):
        """
        :param d_model: dimensionality of the embedding layer to your model; since the position encodings are being
        added to character encodings, these need to match (and will match the dimension of the subsequent Transformer
        layer inputs/outputs)
        :param num_positions: the number of positions that need to be encoded; the maximum sequence length this
        module will see
        :param batched: True if you are using batching, False otherwise
        """
        super().__init__()
        # Dict size
        self.emb = nn.Embedding(num_positions, d_model)
        self.batched = batched

    def forward(self, x):
        """
        :param x: If using batching, should be [batch size, seq len, embedding dim]. Otherwise, [seq len, embedding dim]
        :return: a tensor of the same size with positional embeddings added in
        """
        # Second-to-last dimension will always be sequence length
        input_size = x.shape[-2]
        indices_to_embed = torch.tensor(np.asarray(range(0, input_size))).type(torch.LongTensor)
        if self.batched:
            # Use unsqueeze to form a [1, seq len, embedding dim] tensor -- broadcasting will ensure that this
            # gets added correctly across the batch
            emb_unsq = self.emb(indices_to_embed).unsqueeze(0)
            return x + emb_unsq
        else:
            return x + self.emb(indices_to_embed)


# This is a skeleton for train_classifier: you can implement this however you want
def train_classifier(train):

    # We are training on 2000 tokens. We will feed a maximum of 100 characters at a time.
    vocab_size = 2000
    num_positions = 100
    # You'll probably want to tweak these in your experimentation
    d_model = 512
    d_internal = 2048
    num_classes = 2000
    # We will be using 4 heads in the self-attention
    num_heads = 4
    num_layers = 4

    model = Transformer(vocab_size, num_positions, d_model, d_internal, num_classes, num_layers, num_heads)
    model.zero_grad()
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    tokenizer = RotatedTokenizer(["backend/data.txt"])
    indexer = tokenizer.get_indexer()
    train_text_indexed = tokenizer.get_encoding(train).ids

    train_exs_input = []
    train_exs_truth = []
    window_size = 100
    for i in range(0, len(train_text_indexed) - window_size, window_size):
        window = train_text_indexed[i:i+window_size]
        train_exs_input.append([indexer.token_to_id('<SEP>')] + window[:-1])
        train_exs_truth.append(window)
    # Convert train_exs into tensors
    train_exs_input = torch.LongTensor(train_exs_input)
    train_exs_truth = torch.LongTensor(train_exs_truth)

    num_epochs = 10
    for t in range(0, num_epochs):
        loss_this_epoch = 0.0
        random.seed(t)
        ex_idxs = [i for i in range(0, len(train_exs_input))]
        random.shuffle(ex_idxs)
        loss_fcn = nn.NLLLoss()
        for ex_idx in ex_idxs:
            log_probs, _ = model.forward(train_exs_input[ex_idx])
            loss = loss_fcn(log_probs, train_exs_truth[ex_idx])
            model.zero_grad()
            loss.backward()
            optimizer.step()
            loss_this_epoch += loss.item()
        print("Loss at epoch %i: %.4f" % (t, loss_this_epoch))
    model.eval()
    return model

with open("backend/data.txt", "r") as f:
    data = f.read()
train_classifier(data)