from torch.nn import Module
from torch import Tensor
import torch.nn as nn
import torch

class NerModel(Module):

    def __init__(self, params):
        super(NerModel, self).__init__()
        self.emb = nn.Embedding(params.vocab_size, params.embedding_dim)
        self.lstm = nn.LSTM(params.embedding_dim,
                            params.hidden_dim,
                            num_layers=params.num_layers,
                            bidirectional=True,
                            dropout=params.dropout if
                            params.num_layers > 1 else 0)
        lstm_out_dim = params.hidden_dim * 2
        self.dropout = nn.Dropout(params.dropout)
        self.decoder = nn.Linear(lstm_out_dim, params.num_classes)

    def forward(self, x) -> Tensor:
            emb = self.emb(x)
            emb = self.dropout(emb)
            out, (h, c) = self.lstm(emb)
            out = self.dropout(out)
            label_class = self.decoder(out)
            return label_class

    def tensor_tag_to_id_list(self, predictions) -> list:
        return torch.argmax(predictions, -1).tolist()

    def predict(self, x) -> list:
        self.eval()
        tensor = self.forward(x)
        return self.tensor_tag_to_id_list(tensor)
