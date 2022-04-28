

class Params:

    def __init__(self, vocab_size, tag_size, pos_size):
        self.vocab_size = vocab_size
        self.num_classes = tag_size
        self.pos_size = pos_size
        self.hidden_dim = 128
        self.embedding_dim = 100
        self.pos_embedding_dim = 2
        self.num_layers = 2
        self.dropout = 0.3
