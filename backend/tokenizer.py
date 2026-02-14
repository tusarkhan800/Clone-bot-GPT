import json
import os
from typing import List

class ByteTokenizer:
    def __init__(self):
        self.vocab = {}

    def encode(self, text: str) -> List[int]:
        return [self.vocab[char] for char in text if char in self.vocab]

    def decode(self, token_ids: List[int]) -> str:
        reverse_vocab = {v: k for k, v in self.vocab.items()}
        return ''.join(reverse_vocab[token_id] for token_id in token_ids)

    def save(self, file_path: str):
        with open(file_path, 'w') as f:
            json.dump(self.vocab, f)

    @staticmethod
    def load(file_path: str) -> 'ByteTokenizer':
        tokenizer = ByteTokenizer()
        with open(file_path, 'r') as f:
            tokenizer.vocab = json.load(f)
        return tokenizer
