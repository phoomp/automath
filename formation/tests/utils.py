from collections import Counter, OrderedDict

import torch
from torch import nn
from torch.utils.data import DataLoader
import torchtext

import numpy as np


class Model(nn.Module):
    def __init__(self, dataset, lstm_size, embedding_dim, lstm_num_layer, lstm_dropout=0.2):
        super().__init__()
        self.lstm_size = 128
        self.embedding_dim = embedding_dim
        self.lstm_num_layer = lstm_num_layer
        self.lstm_dropout = lstm_dropout

        n_vocab = dataset.get_vocab_len()

        self.embedding1 = nn.Embedding(num_embeddings=n_vocab,
                                       embedding_dim=self.embedding_dim
                                       )

        self.lstm = nn.LSTM(input_size=self.lstm_size,
                            hidden_size=self.lstm_size,
                            num_layers=self.lstm_num_layer,
                            dropout=self.lstm_dropout
                            )

        self.fc1 = nn.Linear(self.lstm_size, n_vocab)

    def forward(self, x, cell_state):
        x = self.embedding1(x)
        x, state = self.lstm(x, cell_state)
        x = self.fc(x)
        return x, state

    def init_state(self, seq_len):
        return (
            torch.zeros(self.lstm_num_layer, seq_len, self.lstm_size),
            torch.zeros(self.lstm_num_layer, seq_len, self.lstm_size)
        )


def is_dialogue(line) -> bool:
    words = line.split()
    if len(words) <= 2:
        return False

    for word in words:
        if word.isupper() and len(word) > 1:
            return False

    return True


class Dataset(torch.utils.data.Dataset):
    def __init__(self, ds_path):
        self.lines = []
        with open(ds_path, 'r') as txt:
            for line in txt:
                line = line.replace('"', '')
                line = line.replace('\t', '')
                line = line.replace('\n', '')
                self.lines.append(line)

        print(len(self.lines))
        print(self.lines[-1])

        # Prepare dataset
        self.s_lines = [list(s) for s in self.lines]
        self.max_seq_len = max([len(x) for x in self.s_lines])
        print(f'Max sequence length: {self.max_seq_len}')

        # Build vocabs
        v = sorted(set(''.join(self.lines)))
        self.vocab = {}
        for char in v:
            self.vocab[char] = len(self.vocab)

        print(f'Character list: {self.vocab}')

    def __len__(self):
        return len(self.s_lines)

    def get_tokenized_tensor(self, line):
        tensor = torch.zeros(len(line) - 1, 1, len(self.vocab))

        for i, char in enumerate(line[:-1]):
            idx = self.vocab[char]
            tensor[i, 0, idx] = 1

        return tensor

    def get_tokenized_label(self, x):
        tensor = torch.zeros()

    def __getitem__(self, idx):
        selected = self.s_lines[idx]
        feature = torch.Tensor(self.get_tokenized_tensor(selected))
        labels = torch.Tensor(self.get_tokenized_label(selected))

        return feature, labels
