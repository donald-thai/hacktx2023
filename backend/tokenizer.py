from tokenizers import (
    models,
    trainers,
    Tokenizer,
    pre_tokenizers,
    decoders,
)
import os

class RotatedTokenizer():
    def __init__(self, files):
        # check if the tokenizer is already saved as a file
        # if it is, load it
        # otherwise, train the tokenizer and save it as a file
        if os.path.exists("backend/tokenizer.json"):
            self.tokenizer = Tokenizer.from_file("backend/tokenizer.json")
            return
        unk_token = "<UNK>"  # token for unknown words
        spl_tokens = [unk_token, "<SEP>"]  # special tokens
        self.tokenizer = Tokenizer(models.BPE(unk_token = unk_token))
        trainer = trainers.BpeTrainer(special_tokens = spl_tokens, vocab_size = 2000)
        self.tokenizer.train(files, trainer=trainer)
        self.tokenizer.save("backend/tokenizer.json")

    def get_indexer(self):
        return self.tokenizer

    def get_encoding(self, text):
        return self.tokenizer.encode(text)