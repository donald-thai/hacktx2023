# models.py

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from transformer import *
import math


class LanguageModel(object):

    def get_next_token_log_probs(self, context) -> np.ndarray:
        """
        Returns a log probability distribution over the next characters given a context.
        The log should be base e

        NOTE: You should make sure you call model.eval() to determinize inference here (turns off dropout
        layers in TransformerEncoder).
        :param context: the string context that the LM conditions on
        :return: A numpy vector log P(y | context) where y ranges over the output vocabulary.
        """
        raise Exception("Only implemented in subclasses")


    def get_log_prob_sequence(self, next_chars, context) -> float:
        """
        Scores a bunch of characters following context. That is, returns
        log P(nc1, nc2, nc3, ... | context) = log P(nc1 | context) + log P(nc2 | context, nc1), ...
        The log should be base e

        NOTE: You should make sure you call model.eval() to determinize inference here (turns off dropout
        layers in TransformerEncoder).
        :param next_chars:
        :param context:
        :return: The float probability
        """
        raise Exception("Only implemented in subclasses")

class NeuralLanguageModel(LanguageModel):
    def __init__(self, vocab_size, vocab_index, window_size, model=None):
        self.model = model
        self.vocab_size = vocab_size
        self.vocab_index = vocab_index
        self.window_size = window_size

    def get_next_token_log_probs(self, context):
        if len(context) >self.window_size-1:
            context = context[-self.window_size-1:]
        if len(context) < self.window_size-1:
            context = ["<SEP>" for _ in range((self.window_size-1 - len(context)))] + context
        assert(len(context) == self.window_size-1)
        context = ["<SEP>"] + context
        # Convert the context to indices
        context = [self.vocab_index.token_to_id(c) for c in context]
        # Convert the context to tensor and add batch dimension
        context = torch.LongTensor(context)
        
        # Run the model to get predictions
        with torch.no_grad():
            self.model.eval()
            log_probs, attn_map = self.model(context)
        # Convert from log base 10 to log base e
        
        # Convert log probabilities tensor num_positions x vocab_size to numpy array vocab_size
        log_probs = log_probs.squeeze(0).numpy()
        log_probs = log_probs / np.log(math.e)
        if log_probs.ndim == 0:
            raise Exception("Log probs is a scalar")
        return log_probs[-1]


    def get_log_prob_sequence(self, next_chars, context):
        log_prob = 0.0
        next_chars = [self.vocab_index.index_of(c) for c in next_chars]
        for char in next_chars:
            char_log_probs = self.get_next_char_log_probs(context)
            # Add the log probability of the current character
            log_prob += char_log_probs[char]
            context += self.vocab_index.id_to_token(char)
            char_log_probs = self.get_next_char_log_probs(context)
        
        return log_prob
    
def train_model(train):

    # We are training on 2000 tokens. We will feed a maximum of 100 characters at a time.
    vocab_size = 2000
    num_positions = 20
    # You'll probably want to tweak these in your experimentation
    d_model = 512
    d_internal = 2048
    num_classes = 2000
    # We will be using 4 heads in the self-attention
    num_heads = 4
    num_layers = 4

    tokenizer = RotatedTokenizer(["backend/data.txt"])
    indexer = tokenizer.get_indexer()
    train_text_indexed = tokenizer.get_encoding(train).ids[:50000]

    model = Transformer(vocab_size, num_positions, d_model, d_internal, num_classes, num_layers, num_heads)
    model.zero_grad()
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    train_exs_input = []
    train_exs_truth = []
    window_size = 20
    for i in range(0, len(train_text_indexed) - window_size, window_size):
        window = train_text_indexed[i:i+window_size]
        train_exs_input.append([indexer.token_to_id('<SEP>')] + window[:-1])
        train_exs_truth.append(window)
    # Convert train_exs into tensors
    train_exs_input = torch.LongTensor(train_exs_input)
    train_exs_truth = torch.LongTensor(train_exs_truth)

    print("done preprocessing")

    num_epochs = 5
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

# with open("backend/data.txt", "r") as f:
#     train = f.read()
# model = train_model(train)
# torch.save(model.state_dict(), "backend/transformerModel.pt")