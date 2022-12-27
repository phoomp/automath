import torch
from torch import nn


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
                if is_dialogue(line):
                    self.lines.append(line)
                else:
                    continue

        print(len(self.lines))
        print(self.lines[-1])

    def __len__(self):
        pass

    def __getitem__(self, idx):
        if not torch.is_tensor(idx):
            idx = torch.Tensor(idx)

