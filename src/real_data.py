import torch
from torch.utils.data import Dataset


class RealTPPDataset(Dataset):
    def __init__(self, sequences):
        self.sequences = sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx]


def collate_pad(batch):
    #padding for wrong length tensors
    lengths = [seq.shape[0] for seq in batch]
    max_len = max(lengths)

    B = len(batch)
    deltas_padded = torch.zeros(B, max_len, dtype=torch.float32)
    mask = torch.zeros(B, max_len, dtype=torch.float32)

    for i, seq in enumerate(batch):
        L = seq.shape[0]
        deltas_padded[i, :L] = seq
        mask[i, :L] = 1.0

    return deltas_padded, mask