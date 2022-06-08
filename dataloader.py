from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import torch

class MyDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y
        
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, index):
        x = torch.tensor(self.x[index])
        y = torch.tensor(self.y[index])
        return x, y

def collate_fn(batch):
    # separate xs and ys
    xs, ys = zip(*batch)
    # get lengths
    lengths = [len(x) for x in xs]
    # pad sequences
    x_padded = pad_sequence(xs, batch_first=True, padding_value=pad_tok_id)
    y_padded = pad_sequence(ys, batch_first=True, padding_value=pad_tag_id)
    # return padded
    return x_padded, y_padded, lengths