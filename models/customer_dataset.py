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
        self.time_idx = df.index.values

    def __len__(self):
        return len(self.input_arr)

    def __getitem__(self, idx):
        x = torch.from_numpy(self.input_arr[idx, :].astype(np.float32)).to(device)
        y = torch.from_numpy(self.target_arr[idx, :].astype(np.float32)).to(device)
        t = torch.from_numpy(np.array(self.time_idx[idx]).astype(np.float32)).to(device)
        return x, y, t


class BatchTrainDataset(Dataset):
    TYPE = "Batch Train Dateset"

    def __init__(self, df, input_cols, target_cols, time_len=200):
        self.input_arr = df[input_cols].values
        self.target_arr = df[target_cols].values
        self.time_idx = df.index.values
        self.time_len = time_len
        self.input_arr_list, self.target_arr_list, self.time_idx_list = self.prepare_dataset(self.time_len)

    def prepare_dataset(self, time_len):
        input_arr_list = []
        target_arr_list = []
        time_idx_list = []
        idx = 0
        while idx < len(self.input_arr):
            end_idx = idx + time_len if idx + time_len < len(self.input_arr) else -1
            input_arr_list.append(self.input_arr[idx:end_idx, :])
            target_arr_list.append(self.target_arr[idx:end_idx])
            time_idx_list.append(self.time_idx[idx:end_idx])
            idx = idx + time_len
        return input_arr_list, target_arr_list, time_idx_list

    def __len__(self):
        return len(self.input_arr_list)

    def __getitem__(self, idx):
        x = torch.from_numpy(self.input_arr_list[idx].astype(np.float32)).to(device)
        y = torch.from_numpy(self.target_arr_list[idx].astype(np.float32)).to(device)
        t = torch.from_numpy(self.time_idx_list[idx].astype(np.float32)).to(device)
        return x, y, t
