from collections import Counter, OrderedDict

import torch
from torch import nn
from torch.utils.data import DataLoader
import torchtext

import numpy as np


class Model(nn.Module):
    def __init__(self, dataset, lstm_size=128, embedding_dim=128, lstm_num_layer=5, lstm_dropout=0.2):
        super().__init__()
        self.dataset = dataset
        self.lstm_size = lstm_size
        self.embedding_dim = embedding_dim
        self.lstm_num_layer = lstm_num_layer
        self.lstm_dropout = lstm_dropout

        n_vocab = self.dataset.get_vocab_len()

        self.embedding1 = nn.Embedding(num_embeddings=n_vocab,
                                       embedding_dim=self.embedding_dim
                                       )

        self.lstm = nn.LSTM(input_size=self.embedding_dim,
                            hidden_size=self.lstm_size,
                            num_layers=self.lstm_num_layer,
                            dropout=self.lstm_dropout, batch_first=True
                            )

        self.fc1 = nn.Linear(self.lstm_size, n_vocab)

    def forward(self, x, hidden):
        x = self.embedding1(x)
        # print(f'Before error: {x.size()}, {hidden[0].size()}')
        x, hidden = self.lstm(x, hidden)
        x = self.fc1(x)
        return x, hidden

    def init_state(self, device, batch_size):
        return (
            torch.randn(self.lstm_num_layer, batch_size, self.lstm_size).to(device),
            torch.randn(self.lstm_num_layer, batch_size, self.lstm_size).to(device),
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

        # print(len(self.lines))
        # print(self.lines[-1])

        # Prepare dataset
        self.s_lines = [list(s) for s in self.lines]
        self.max_seq_len = max([len(x) for x in self.s_lines])
        print(f'Max sequence length: {self.max_seq_len}')

        # Build vocabs
        v = sorted(set(''.join(self.lines)))
        self.vocab = {'EOS': 0}
        for char in v:
            self.vocab[char] = len(self.vocab)

        print(f'Character list: {self.vocab}')

    def __len__(self):
        return len(self.s_lines)

    def get_tokenized_tensor_onehot(self, line):
        tensor = torch.zeros(self.max_seq_len, len(self.vocab))

        for i, char in enumerate(line):
            idx = self.vocab[char]
            tensor[i, idx] = 1

        return tensor

    def get_tokenized_tensor(self, line):
        new_list = []

        for i, char in enumerate(line):
            new_list.append(self.vocab[char])

        while len(new_list) < self.max_seq_len:
            new_list.append(0)

        return torch.Tensor(new_list).float()

    def get_tokenized_label(self, line):
        new_line = []
        for i, char in enumerate(line[1:]):
            new_line.append(self.vocab[char])

        new_line.append(self.vocab['EOS'])
        while len(new_line) < self.max_seq_len:
            new_line.append(0)
        # print(new_line)
        return torch.Tensor(new_line).float()

    def get_vocab_len(self):
        return len(self.vocab)

    def __getitem__(self, idx):
        selected = self.s_lines[idx]
        feature = self.get_tokenized_tensor(selected)
        label = self.get_tokenized_label(selected)

        return feature, label
