import torch
import torch.nn as nn
from tokenizer import *
import pickle
from rotate import *

with open("backend/data.txt", "r") as f:
    data = f.read()
words = data.split()
vocab = set(words)

vocab_size = len(vocab)
d_model = 512
nhead = 8
num_encoder_layers = 6
num_decoder_layers = 6
dim_feedforward = 2048
max_seq_length = 100  # Adjust as needed

epochs = 5  # Adjust as needed
tokenizer = RotatedTokenizer(["backend/data.txt"])

# model = ... # Implement transformer model here

# optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
# criterion = nn.CrossEntropyLoss()

input_string = "def():\n\treturn 0"
encoding = tokenizer.get_encoding(input_string)
print(encoding.tokens)

# Training loop

# Write the model to a file