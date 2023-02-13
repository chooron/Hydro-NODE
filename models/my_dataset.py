from typing import List, Union

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset, random_split


class TrainDataset(Dataset):
    def __init__(self, feature=None, target=None, device=torch.device('cpu')):
        self.feature = feature
        self.target = target
        self.device = device

    def __len__(self):
        return len(self.feature)

    def __getitem__(self, i):
        if i >= len(self):
            return ValueError()
        x = torch.from_numpy(self.feature[i, :].astype(np.float32)).to(self.device)
        y = torch.from_numpy(self.target[i, :].astype(np.float32)).to(self.device)
        return x, y

    @classmethod
    def set_data(cls, feature, target, dataset):
        # required before training
        return cls(feature=feature, target=target, device=dataset.device)


class TorchDataModule:
    def __init__(self, data: pd.DataFrame, time_idx: Union[str, None], feature_cols: List[str], target_cols: List[str],
                 feature_scaler=StandardScaler(), target_scaler=StandardScaler(), datamodule_type='train',
                 batch_size: int = 64, shuffle: bool = True, dataset=None,
                 device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'), num_workers: int = 0):
        # load exp config
        self.data = data
        self.time_idx = time_idx
        self.time_series = data.pop(time_idx) if time_idx is not None else None
        self.feature_cols = feature_cols
        self.target_cols = target_cols if isinstance(target_cols, List) else [target_cols]
        self.feature_scaler = feature_scaler
        self.target_scaler = target_scaler
        self.datamodule_type = datamodule_type
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.dataset = dataset
        self.device = device
        self.num_workers = num_workers
        self.scaled_feature, self.scaled_target = None, None
        self.normalize()

    def normalize(self):
        if self.datamodule_type == 'train':
            if self.feature_scaler is not None:
                self.scaled_feature = self.feature_scaler.fit_transform(self.data[self.feature_cols])
            else:
                self.scaled_feature = self.data[self.feature_cols].values
            if self.target_scaler is not None:
                self.scaled_target = self.target_scaler.fit_transform(self.data[self.target_cols])
            else:
                self.scaled_target = self.data[self.target_cols].values
        elif self.datamodule_type in ['test', 'val']:
            if self.feature_scaler is not None:
                self.scaled_feature = self.feature_scaler.transform(self.data[self.feature_cols])
            else:
                self.scaled_feature = self.data[self.feature_cols].values
            if self.target_scaler is not None:
                self.scaled_target = self.target_scaler.transform(self.data[self.target_cols])
            else:
                self.scaled_target = self.data[self.target_cols].values
        else:
            raise NotImplementedError

    def get_attributes(self):
        kwargs = {'time_idx': self.time_idx, 'feature_cols': self.feature_cols, 'target_cols': self.target_cols,
                  'feature_scaler': self.feature_scaler, 'target_scaler': self.target_scaler}
        return kwargs

    @classmethod
    def from_datamodule(cls, data, datamodule, datamodule_type='test'):
        kwargs = datamodule.get_attributes()
        kwargs['datamodule_type'] = datamodule_type
        return cls(data, **kwargs)

    def get_sample(self):

        if self.datamodule_type == 'train':
            if self.shuffle:
                dataset = self.dataset.set_data(self.scaled_feature, self.scaled_target, self.dataset)
                train_dataset, val_dataset = random_split(
                    dataset=dataset,
                    lengths=[int(len(dataset) * 0.8), len(dataset) - int(len(dataset) * 0.8)],
                    generator=torch.Generator().manual_seed(42))
            else:
                train_dataset = self.dataset.set_data(self.scaled_feature[:int(len(self.scaled_feature) * 0.8), :],
                                                      self.scaled_target[:int(len(self.scaled_feature) * 0.8), :],
                                                      self.dataset)
                val_dataset = self.dataset.set_data(self.scaled_feature[:int(len(self.scaled_feature) * 0.8), :],
                                                    self.scaled_target[:int(len(self.scaled_feature) * 0.8), :],
                                                    self.dataset)
            return DataLoader(train_dataset, shuffle=False, batch_size=self.batch_size, num_workers=self.num_workers), \
                DataLoader(val_dataset, shuffle=False, batch_size=self.batch_size, num_workers=self.num_workers)
        elif self.datamodule_type == 'test':
            dataset = self.dataset.set_data(self.scaled_feature, self.scaled_target, self.dataset)
            return DataLoader(dataset, shuffle=False, batch_size=self.batch_size, num_workers=self.num_workers)
