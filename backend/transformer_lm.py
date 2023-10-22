import torch
import torch.nn as nn
from tokenizer import *
import pickle

class TransformerModel(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, max_seq_length, pos_dropout, trans_dropout):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, pos_dropout, max_seq_length)
        self.transformer = nn.Transformer(d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, trans_dropout)
        self.fc = nn.Linear(d_model, vocab_size)
        self.d_model = d_model

    def forward(self, src, tgt, src_mask=None, tgt_mask=None, src_padding_mask=None, tgt_padding_mask=None):
        src = self.embedding(src) * (self.d_model ** 0.5)
        tgt = self.embedding(tgt) * (self.d_model ** 0.5)
        src = self.pos_encoder(src)
        tgt = self.pos_encoder(tgt)
        output = self.transformer(src, tgt, src_mask, tgt_mask, None, src_padding_mask, tgt_padding_mask, None)
        return self.fc(output)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

vocab_size = 1500  # Adjust this based on your tokenizer's vocabulary size
d_model = 512
nhead = 8
num_encoder_layers = 6
num_decoder_layers = 6
dim_feedforward = 2048
max_seq_length = 100  # Adjust as needed
pos_dropout = 0.1
trans_dropout = 0.1

model = TransformerModel(vocab_size, d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, max_seq_length, pos_dropout, trans_dropout)

# Assuming you have a DataLoader `data_loader` that outputs source and target sequences
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss()

epochs = 5  # Adjust as needed
# import plain text data from the files "dataRotated{i}.txt"
tokenizer = RotatedTokenizer(["dataRotated.txt"], vocab_size)

data = ""
with open("dataRotated.txt", "r") as f:
    data = f.read()
encoding = tokenizer.get_encoding(data)
print(encoding)
data = encoding.ids

for epoch in range(epochs):
    for i in range(0, len(data) - max_seq_length, max_seq_length):
        src = data[i:i + max_seq_length]
        tgt = data[i + 1:i + max_seq_length + 1]
        optimizer.zero_grad()
        tgt_input = tgt[:-1]
        tgt_real = tgt[1:]
        output = model(torch.tensor(src).unsqueeze(1), torch.tensor(tgt_input).unsqueeze(1))
        output = output.view(-1, vocab_size)
        tgt_real = torch.tensor(tgt_real).view(-1)
        loss = criterion(output, tgt_real)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch} loss: {loss.item()}")

with open('model.pkl', 'wb') as f:
  pickle.dump(model, f)
