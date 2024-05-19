from sklearn.preprocessing import OneHotEncoder
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import torch
import os

class VolvoDataset(Dataset):
    
    def __init__(self, data_path = "", verbose=True):
        self.data_path = data_path
        self.groups = []
        #load df in memory
        self.volvo_df = pd.read_csv(self.data_path)
        #fit one hot encoder on labels
        self.risk_encoder = OneHotEncoder()
        self.risk_encoder.fit(self.volvo_df["risk_level"].values.reshape(-1, 1))
        #preprocess df   
        self.volvo_df = self.__preprocess__(verbose)
        self.df_list = self.__group_by_chassis__(verbose)
        #get statistics
        self.n_groups = len(self.df_list)
        self.groups_len = [len(df) for df in self.df_list]

    def __preprocess__(self, verbose = False):
        """
        Preprocess the volvo df by removing NaN columns, static columns and correlated features
        """
        assert self.volvo_df is not None
        
        if verbose:
            print("Dropping all NaN column")
        self.volvo_df.dropna(axis = 1, inplace=True)
        if verbose:
            print("Dropping all static columns")
        columns_to_drop = self.volvo_df.loc[:, self.volvo_df.apply(pd.Series.nunique) == 1].columns
        columns_to_drop = [ x for x in columns_to_drop if x not in ["Timesteps", "ChassisId_encoded", "gen", "risk_level"]]
        self.volvo_df.drop(columns_to_drop, axis=1, inplace=True)
        return self.volvo_df
        
    def __group_by_chassis__(self, verbose = True):
        assert self.volvo_df is not None
        
        if verbose:
            print("Grouping by Chassis id")
        #each chassis has now a df with its multivariate time series
        self.df_list = [t[1] for t in self.volvo_df.groupby("ChassisId_encoded")]
        return self.df_list
        
    def __len__(self):
        return len(self.df_list)

    def __getitem__(self, idx):
        """_summary_

        Args:
            idx (int): idx over df_lists of chassis df

        Returns:
            tuple: time_series, one_hot labels for each point in time series
        """
        assert idx < len(self)
        # retrieve the idx-th group
        ts = self.df_list[idx].sort_values(by=["Timesteps"], ascending=True)
        # retrieve all usefull infromation from that df
        chassis = ts["ChassisId_encoded"].iloc[0]
        # generate multivariate timesereies (n_timesteps, 289) 289 atm with simple preprocess
        time_series = ts.drop(["Timesteps", "ChassisId_encoded", "gen", "risk_level"], axis = 1).values
        # point_wise labels
        timestep_labels = ts["risk_level"]
        return time_series , self.risk_encoder.transform(timestep_labels.values.reshape(-1, 1)).todense()
    
    @staticmethod
    def padding_collate_fn(batch):
        data, labels = zip(*batch)
        # get shapes
        n_features = data[0].shape[1]
        n_labels = labels[0].shape[1]
        # compute max len
        max_len = max([d.shape[0] for d in data])
        # allign data with respect to max sequence len
        data_alligned = np.zeros((len(batch), max_len, n_features))
        labels_allinged = np.zeros((len(batch), max_len, n_labels))
        # 0 where we , FLO is happier this way
        mask = np.zeros((len(batch), max_len))
        # fill with meaningfull data
        for i, d in enumerate(data):
            data_alligned[i, :d.shape[0], :] = d
            labels_allinged[i, :labels[i].shape[0], :] = labels[i]
            # set 1 where meaningfull values
            mask[i,:d.shape[0]] = 1
        return data_alligned, labels_allinged, mask