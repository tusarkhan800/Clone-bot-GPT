class ByteTokenizer:
    def __init__(self):
        pass

    def tokenize(self, text):
        # Example tokenization logic
        return list(text.encode('utf-8'))

    def detokenize(self, tokens):
        # Convert tokens back to string
        return bytes(tokens).decode('utf-8')
