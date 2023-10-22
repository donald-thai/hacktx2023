from transformer_lm import NeuralLanguageModel
from tokenizer import *
import torch
from transformer import *

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
window_size = 20

tokenizer = RotatedTokenizer(["backend/data.txt"])
indexer = tokenizer.get_indexer()
# load the model from torch
transformer = Transformer(vocab_size, num_positions, d_model, d_internal, num_classes, num_layers, num_heads)
transformer.load_state_dict(torch.load("backend/transformerModel.pt"))
transformer.eval()

lm = NeuralLanguageModel(vocab_size, indexer, window_size, transformer)

next_input = indexer.encode("""class<SEP>bruh():""").tokens

for i in range(10):
    # get the next token log probabilities
    next_input = indexer.encode("".join(next_input)).tokens
    log_probs = lm.get_next_token_log_probs(next_input)

    # sort the nd array but keep the indices
    sorted_indices = np.argsort(log_probs)

    top_choice = indexer.id_to_token(sorted_indices[-1])

    next_input.append(top_choice)

print(next_input)