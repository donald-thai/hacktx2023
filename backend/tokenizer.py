from tokenizers import (
    models,
    trainers,
    Tokenizer,
)

class RotatedTokenizer():
    def __init__(self, files, vocab_size=10000):
        self.tokenizer = Tokenizer(models.BPE())
        trainer = trainers.BpeTrainer(vocab_size=vocab_size, special_tokens=[""])
        self.tokenizer.train(files, trainer=trainer)

    def get_encoding(self, text):
        return self.tokenizer.encode(text)