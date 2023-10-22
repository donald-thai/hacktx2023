import torch
import torch.nn as nn
from tokenizer import *
import pickle

vocab_size = 1500  # Adjust this based on your tokenizer's vocabulary size
d_model = 512
nhead = 8
num_encoder_layers = 6
num_decoder_layers = 6
dim_feedforward = 2048
max_seq_length = 100  # Adjust as needed
pos_dropout = 0.1
trans_dropout = 0.1

epochs = 5  # Adjust as needed
tokenizer = RotatedTokenizer(["dataRotated.txt"], vocab_size)

model = ... # Implement transformer model here

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss()

with open("dataRotated.txt", "r") as f:
    data = f.read()

encoding = tokenizer.get_encoding(data)
data = encoding.ids

# Training loop

# Write the model to a file

with open('model.pkl', 'wb') as f:
  pickle.dump(model, f)
