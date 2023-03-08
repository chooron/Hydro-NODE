from typing import List, Union

import numpy as np
import torch
from torch.utils.data import Dataset

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


# pretrain dataset
class PretrainDataset(Dataset):
    def __init__(self, df, input_cols, target_cols):
        self.input_arr = df[input_cols].values
        self.target_arr = df[target_cols].values

    def __len__(self):
        return len(self.input_arr)

    def __getitem__(self, idx):
        x = torch.from_numpy(self.input_arr[idx, :].astype(np.float32)).to(device)
        y = torch.from_numpy(self.target_arr[idx, :].astype(np.float32)).to(device)
        return x, y, idx


# train dataset
class TrainDataset(Dataset):
    TYPE = "Single Step Train Dateset"

    def __init__(self, df, input_cols, target_cols):
        self.input_arr = df[input_cols].values
        self.target_arr = df[target_cols].values
        self.sample_idx = df.index.values

    def __len__(self):
        return len(self.input_arr)

    def __getitem__(self, idx):
        x = torch.from_numpy(self.input_arr[idx, :].astype(np.float32)).to(device)
        y = torch.from_numpy(self.target_arr[idx, :].astype(np.float32)).to(device)
        t = torch.from_numpy(np.array(self.sample_idx[idx]).astype(np.float32)).to(device)
        return x, y, t
