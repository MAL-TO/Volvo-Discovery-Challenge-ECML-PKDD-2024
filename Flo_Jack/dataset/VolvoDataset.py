from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import torch
import os
import tqdm

from dataset.dataset_utils import wavelet_spectrogram

class VolvoDataset(Dataset):
    
    def __init__(self, data_path = "", variants_path="", verbose=True, test=False, columns_to_keep=None, scaler=None):
        self.data_path = data_path
        self.variants_path = variants_path
        self.test = test
        self.groups = []
        self.scaler = scaler
        #load df in memory
        self.volvo_df = pd.read_csv(self.data_path)
        self.variants = pd.read_csv(self.variants_path)

        #fit one hot encoder on labels
        if not self.test:
            print("--- Train Dataset ---")
            self.risk_encoder = OneHotEncoder(
                categories=[['Low', 'Medium', 'High']]
                )
            self.risk_encoder.fit(self.volvo_df["risk_level"].values.reshape(-1, 1))
            self.header_columns = ["Timesteps", "ChassisId_encoded", "gen", "risk_level"]
            #preprocess df
            print("preprocessing ... ")
            self.volvo_df = self.__preprocess__(verbose)
            #save dataframe structure to apply on unseen data
            self.kept_columns = self.volvo_df.columns
            
        elif self.test:
            print("--- Test Dataset ---")
            self.header_columns = ["Timesteps", "ChassisId_encoded", "gen"] 
            if columns_to_keep is not None:
                column_to_drop = [ x for x in self.volvo_df.columns if x not in columns_to_keep]
                self.volvo_df.drop(column_to_drop, axis=1, inplace=True)
            
        self.df_list = self.__group_by_chassis__(verbose)
        #get statistics
        self.n_groups = len(self.df_list)
        self.groups_len = [len(df) for df in self.df_list]
        # self.test_contiguity()
        self.randomize_length = True

    def set_randomize(self, randomize):
        self.randomize_length = randomize

    def keep_indexes(self, indexes):
        # TODO
        pass

    def __preprocess__(self, verbose = False):
        """
        Preprocess the volvo df by removing NaN columns, static columns and correlated features
        """
        assert self.volvo_df is not None
        
        if verbose:
            print("Dropping useless columns")
        columns_to_drop = ["f__51", "f__52", "f__65", "f__117", "f__119", "f__123",  "f__133", \
                            "af2__5",  "af2__6",  "af2__13",  "af2__18",  "af2__19",  "af2__20",  "af2__22", "f__22", \
                            'af1__5', 'af1__10', 'af1__22', 'af1__26', 'af1__30', 'f__10', 'f__15', 'f__26', 'f__36', 'f__90', 'f__91', 'f__98', 'f__104', 'f__120', 'f__122', 'f__127', 'f__129', 'f__140', 'f__152', 'f__153', 'f__154', 'f__158', 'f__169', 'f__170', 'f__172', 'f__173', 'f__174', 'f__175', 'f__176', 'f__177', 'f__178', 'f__188', 'f__198', 'f__199', 'f__203', 'f__207', 'f__214', 'f__215', 'f__219', 'f__228', 'f__231', 'f__236']

        self.volvo_df.drop(columns_to_drop, axis=1, inplace=True)
        return self.volvo_df
    
    def get_schema(self):
        return self.kept_columns
        
    def __group_by_chassis__(self, verbose = True):
        assert self.volvo_df is not None
        
        #each chassis has now a df with its multivariate time series
        self.df_list = []
        groups = self.volvo_df.groupby("ChassisId_encoded")
        for name, group_df in tqdm.tqdm(groups, desc="Group and feature extraction"):
            if not self.test and (len(group_df) < 5):# or len(group_df['risk_level'].value_counts()) < 2):
                continue
            
            group_headings = group_df[self.header_columns]
            group_features = group_df.drop(self.header_columns, axis = 1)

            diffs = group_features.diff(axis=1).fillna(0)
            diffs.columns = [x + "_diff" for x in group_features.columns]

            wavelet_df = wavelet_spectrogram(group_features, 4)

            group_df = pd.concat([group_headings.reset_index(drop=True), 
                                  group_features.reset_index(drop=True), 
                                  diffs.reset_index(drop=True),
                                  wavelet_df.reset_index(drop=True)
                                  ], axis=1)

            self.df_list.append(group_df)

        # print("Scaling features...", end='')
        # if self.scaler == None:
        #     if not self.test:
        #         all_dataset = pd.concat(self.df_list, axis=0, ignore_index=True).drop(self.header_columns, axis=1)
        #         self.scaler = MinMaxScaler()
        #         self.scaler.fit(all_dataset)
        # for group in self.df_list:
        #     group_features = group.drop(self.header_columns, axis=1)
        #     group_features_scaled = self.scaler.transform(group_features)
        #     group_features_scaled_df = pd.DataFrame(group_features_scaled, columns=group_features.columns)
        #     group = pd.concat([group[self.header_columns], group_features_scaled_df], axis=0, ignore_index=True)
        # print('done')
        
        print(f"Num timeseries: {len(self.df_list)}")
        print(f"Num features: {len(self.df_list[0].columns)}")
        return self.df_list
    
    def get_n_features(self):
        assert self.volvo_df is not None
        features, variant, labels = self[0]
        return features.shape[-1]
        
    def get_len_variants():
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
        time_series = ts.drop(self.header_columns, axis = 1).values

        # point_wise labels
        if not self.test:
            #train data with labels 
            timestep_labels = ts["risk_level"]
            labels = self.risk_encoder.transform(timestep_labels.values.reshape(-1, 1)).todense()
        elif self.test:
            #test data without risk_level as key in dataframe
            labels = np.empty((len(time_series),3))
            labels.fill(np.nan)

        chassis_vector = self.variants[self.variants['ChassisId_encoded'] == chassis_id].drop(['ChassisId_encoded'], axis=1).values[0]

        random_start = 0
        random_end = len(time_series)
        if not self.test and self.randomize_length and len(labels) > 5:
            random_len = np.random.randint(5, len(time_series))
            remainder = len(time_series) - random_len
            random_start = np.random.randint(0, remainder)
            random_end = random_start + random_len

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