from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import torch
import os
import tqdm
from copy import copy, deepcopy

from sklearn.model_selection import train_test_split

from dataset.dataset_utils import wavelet_spectrogram, stft_spectogram
from dataset.Processor import Processor


class VolvoDataset(Dataset):
    def __init__(self, data_path = "", variants_path="", verbose=True, test=False):
        self.data_path = data_path
        self.variants_path = variants_path
        self.val = False
        self.test = test
        self.groups = []

        #load df in memory
        self.volvo_df = pd.read_csv(self.data_path)
        self.variants = pd.read_csv(self.variants_path)
  
        self.randomize_length = True

    def set_randomize(self, randomize):
        self.randomize_length = randomize

    def split_train_validation(self, train_ratio=0.8):
        all_indexes = list(range(len(self.df_list)))
        X_train, X_test, _, _ = train_test_split(all_indexes, all_indexes, train_size=train_ratio)
        validation_dataset = copy(self)

        self.keep_indexes(X_train)
        validation_dataset.keep_indexes(X_test)
        validation_dataset.val = True

        return self, validation_dataset

    def keep_indexes(self, idx):
        self.df_list = [self.df_list[i] for i in idx]

    def get_processor(self):
        self.processor = Processor()
        self.volvo_df = self.processor.preprocess(self.volvo_df)
        self.processor.fit(self.volvo_df)        
        self.df_list = self.processor.group_by_chassis(self.volvo_df, skip_if_less_than=10, split_in_len=10)
    

        return self.processor

    def set_processor(self, processor):
        self.processor = processor
        self.volvo_df = self.processor.preprocess(self.volvo_df)
        self.df_list = self.processor.group_by_chassis(self.volvo_df)

    def get_n_features(self):
        assert self.volvo_df is not None
        features, variant, labels = self[0]
        return features.shape[-1]
        
    def get_len_variants(self):
        assert self.volvo_df is not None
        features, variant, labels = self[0]
        return len(variant)

    def __len__(self):
        return len(self.df_list)

    def __getitem__(self, idx):
        """_summary_

        Args:
            idx (int): idx over df_lists of chassis df

        Returns:
            tuple: time_series, one_hot labels for each point in time series
        """
        assert idx < len(self), f"Got {idx=} when {len(self)=}"
        # retrieve the idx-th group
        ts = self.df_list[idx].sort_values(by=["Timesteps"], ascending=True)
        chassis_id = ts['ChassisId_encoded'].iloc[0]
        _, time_series = self.processor.split_heading_features(ts)

        time_series = self.processor.extract_features(time_series)
        time_series = time_series.values

        # point_wise labels
        if not self.test:
            #train data with labels 
            timestep_labels = ts["risk_level"]
            labels = self.processor.risk_encoder.transform(timestep_labels.values.reshape(-1, 1)).todense()
            if np.isnan(labels).any():
                print(np.isnan(labels).any())
                print(labels)
        elif self.test:
            #test data without risk_level as key in dataframe
            labels = np.empty((len(time_series),3))
            labels.fill(np.nan)

        chassis_vector = self.variants[self.variants['ChassisId_encoded'] == chassis_id].drop(['ChassisId_encoded'], axis=1).values[0]

        random_start = 0
        random_end = len(time_series)
        if  not self.test:
            
            if self.randomize_length and len(labels) > 5:
                random_len = np.random.randint(5, 15)#len(time_series))
                # if self.val: random_len = 10

                remainder = len(time_series) - random_len
                random_start = np.random.randint(0, remainder) if remainder > 0 else 0
                random_end = random_start + random_len
            
            # means = np.mean(time_series, axis=0)
            # std = np.std(time_series, axis=0)
            # noise = np.random.normal(0, 0.2*std, size=time_series.shape)
            # time_series += noise

        # labels_binary = labels.copy()
        # labels_binary[:, 1] = labels_binary[:, 1] + labels_binary[:, 2]
        # labels_binary = labels_binary[:, :2]
        # time_series = np.hstack([time_series, labels_binary])

        return torch.Tensor(time_series)[random_start: random_end], torch.Tensor(chassis_vector), torch.Tensor(labels)[random_start: random_end]
    
    @staticmethod
    def padding_collate_fn(batch):
        data, variants, labels = zip(*batch)
        # get shapes
        n_features = data[0].shape[1]
        n_labels = labels[0].shape[1]
        # compute max len
        max_len = max([d.shape[0] for d in data])
        # allign data with respect to max sequence len
        data_alligned = torch.zeros((len(batch), max_len, n_features))
        labels_allinged = torch.zeros((len(batch), max_len, n_labels))
        # 0 where we , FLO is happier this way
        mask = torch.zeros((len(batch), max_len))
        # fill with meaningfull data
        for i, d in enumerate(data):
            data_alligned[i, :d.shape[0], :] = d
            labels_allinged[i, :labels[i].shape[0], :] = labels[i]
            # set 1 where meaningfull values
            mask[i,:d.shape[0]] = 1

        # trust me bro
        variants = torch.stack(variants).unsqueeze(1).repeat([1, max_len, 1])

        return data_alligned, variants, labels_allinged, mask