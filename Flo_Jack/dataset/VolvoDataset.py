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


class VolvoDatasetPart1(Dataset):
    def __init__(self, data_path = "", variants_path="", verbose=True, test=False):
        self.data_path = data_path
        self.variants_path = variants_path
        self.val = False
        self.test = test
        self.groups = []

        #load df in memory
        self.volvo_df = pd.read_csv(self.data_path)
        self.variants = pd.read_csv(self.variants_path)

        # if not self.test:
            # self.volvo_df = self.volvo_df.loc[self.volvo_df['risk_level']!="Low"] DROPPA TUTTO CIO' CHE E' LOW 
  
        self.randomize_length = True

    def set_randomize(self, randomize):
        self.randomize_length = randomize

    def get_weights(self):
        return np.unique(self.y_stratify, return_counts=True)[1]/len(self.y_stratify)

    def split_train_validation(self, train_ratio=0.8):
        all_indexes = list(range(len(self.df_list)))
        self.y_stratify = []
        for i, group_df in enumerate(self.df_list):
            if len(group_df['risk_level'].value_counts()) > 1:
                self.y_stratify.append(1)
                # remove all the low risk_level
                self.df_list[i] = group_df[group_df['risk_level'] != 'Low']
            else:
                self.y_stratify.append(0)

        X_train, X_test, _, _ = train_test_split(all_indexes, all_indexes, train_size=train_ratio, stratify=self.y_stratify)
        validation_dataset = deepcopy(self)

        self.keep_indexes(X_train)
        validation_dataset.keep_indexes(X_test)

        # PARTE 1 + 2 
        # prende ogni time series e prende una window che salta di param every, prima di fare questo droppa tutte le low se la time series = 1, altrimenti tieni tutte le low 
        # check se ci sono solo medium o solo high (quindi no salto)
        self.df_list = self.processor.split_in_len(
            self.df_list, 10, every=4
        )

        validation_dataset.df_list = validation_dataset.processor.split_in_len(
            validation_dataset.df_list, split_in_len=10, every=4
        )
        validation_dataset.val = True

        # new_df_list = []
        # for i in tqdm.tqdm(range(len(self.df_list))):
        #     if len(self.df_list[i].value_counts()) > 1:
        #         new_df_list.append(self.df_list[i])

        # self.df_list = new_df_list


        return self, validation_dataset

    def keep_indexes(self, idx):
        self.df_list = [self.df_list[i] for i in idx]

    def get_processor(self):
        self.processor = Processor()
        self.volvo_df = self.processor.preprocess(self.volvo_df)
        self.processor.fit(self.volvo_df)        
        self.df_list = self.processor.group_by_chassis(self.volvo_df, skip_if_less_than=10, split_in_len=10)

        # PARTE 1 
        # dopo che hai fatto il group by chassis, dagli una label per dire se è sempre buono oppure no per poi trainare il classificatore binario 

        return self.processor
    
    # ESPERIMENTO 1: classificatore che divide buono da cattivo, se ho cattivo prima metà medium seconda high, se buona tutto low 

    def set_processor(self, processor):
        self.processor = processor
        self.volvo_df = self.processor.preprocess(self.volvo_df)
        self.df_list = self.processor.group_by_chassis(self.volvo_df)

    def get_n_features(self):
        assert self.volvo_df is not None
        features, variant, labels = self[0]
        return features.shape[-1]
    
    def get_n_classes(self):
        return np.unique(self.y_stratify).shape[0]
        
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
            # SE IN FASE 1: UNICA LABEL 0 SE TUTTO LOW, 1 SE TUTTO MEDIUM/HIGH 
            # IN FASE 2 AVRO' SOLO MEDIUM E HIGH E VOGLIO 1 DOVE AVVIENE IL SALTO E 0 ALTRIMENTI  
            timestep_labels = ts["risk_level"].values
            label = 0 if (np.unique(timestep_labels) == "Low").all() else 1
        elif self.test:
            label = -1

        chassis_vector = self.variants[self.variants['ChassisId_encoded'] == chassis_id].drop(['ChassisId_encoded'], axis=1).values[0] # keep the 'static' features

        # questo prende le random len nelle time series, probabilmente da togliere dopo che nella seconda fase io tengo solo medium e high e ricavo da lì le time series in modo sistematico 
        random_start = 0
        random_end = len(time_series)
        # if  not self.test:
            
        #     if self.randomize_length and len(labels) > 5:
        #         random_len = np.random.randint(5, 15)#len(time_series))
        #         if self.val: random_len = 10

        #         remainder = len(time_series) - random_len
        #         random_start = np.random.randint(0, remainder) if remainder > 0 else 0
        #         random_end = random_start + random_len
            
        #     if not self.val:
        #         std = np.std(time_series, axis=0)
        #         noise = np.random.normal(0, 0.2*std, size=time_series.shape)
        #         time_series += noise

        # labels_binary = labels.copy()
        # labels_binary[:, 1] = labels_binary[:, 1] + labels_binary[:, 2]
        # labels_binary = labels_binary[:, :2]
        # labels = labels_binary
        # time_series = np.hstack([time_series, labels_binary])

        # time_series = time_series + self.getPositionEncoding(len(time_series), len(time_series[0]))
        time_series = np.hstack([time_series, self.getPositionEncoding(len(time_series), 50)])

        #time_series = np.hstack([time_series, ts['Timesteps'].to_numpy().reshape(-1,1)]) DA CAPIRE SE HA SENSO MANTENERLO 
        return torch.Tensor(time_series)[random_start: random_end], torch.Tensor(chassis_vector), label
    
    def getPositionEncoding(self, seq_len, d, n=1000):
        P = np.zeros((seq_len, d))
        for k in range(seq_len):
            for i in np.arange(int(d/2)):
                denominator = np.power(n, 2*i/d)
                P[k, 2*i] = np.sin(k/denominator)
                P[k, 2*i+1] = np.cos(k/denominator)
        return P

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


# create children class of VolvoDataset

class VolvoDatasetPart2(VolvoDatasetPart1):
    def __init__(self, data_path = "", variants_path="", verbose=True, test=False):
        super().__init__(data_path, variants_path, verbose, test)

    def split_train_validation(self, train_ratio=0.8):
        self.df_list = [group_df for group_df in self.df_list if len(group_df['risk_level'].value_counts()) > 1]
        all_indexes = list(range(len(self.df_list)))

        for i, group_df in enumerate(self.df_list):
            self.df_list[i] = group_df[group_df['risk_level'] != 'Low']
            

        X_train, X_test, _, _ = train_test_split(all_indexes, all_indexes, train_size=train_ratio)
        validation_dataset = deepcopy(self)

        self.keep_indexes(X_train)
        validation_dataset.keep_indexes(X_test)

        # PARTE 2 
        # prende ogni time series e prende una window che salta di param every, prima di fare questo droppa tutte le low se la time series = 1, altrimenti tieni tutte le low 
        # check se ci sono solo medium o solo high (quindi no salto)
        self.df_list = self.processor.split_in_len(
            self.df_list, 10, every=4
        )

        validation_dataset.df_list = validation_dataset.processor.split_in_len(
            validation_dataset.df_list, split_in_len=10, every=4
        )
        validation_dataset.val = True

        # new_df_list = []
        # for i in tqdm.tqdm(range(len(self.df_list))):
        #     if len(self.df_list[i].value_counts()) > 1:
        #         new_df_list.append(self.df_list[i])

        # self.df_list = new_df_list

        # keep only the windows that have at least 2 different labels
        print(f"Train: {len(self.df_list)} Validation: {len(validation_dataset.df_list)}")

        new_df_list = []
        for i in range(len(self.df_list)):
            if len(self.df_list[i]['risk_level'].value_counts()) > 1:
                new_df_list.append(self.df_list[i])
        self.df_list = new_df_list

        new_df_list = []
        for i in range(len(validation_dataset.df_list)):
            if len(validation_dataset.df_list[i]['risk_level'].value_counts()) > 1:
                new_df_list.append(validation_dataset.df_list[i])
        validation_dataset.df_list = new_df_list

        print(f"Train: {len(self.df_list)} Validation: {len(validation_dataset.df_list)}")

        return self, validation_dataset
    
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
            # IN FASE 2 AVRO' SOLO MEDIUM E HIGH E VOGLIO 1 DOVE AVVIENE IL SALTO E 0 ALTRIMENTI  
            timestep_labels = ts["risk_level"].values
            print(timestep_labels)

            # create a vector that has only 0 and 1 where the jump happens
            label = np.zeros(len(timestep_labels))
            first_high = np.where(timestep_labels == "High")[0][0]
            label[first_high] = 1
        elif self.test:
            label = np.zeros(len(time_series))

        chassis_vector = self.variants[self.variants['ChassisId_encoded'] == chassis_id].drop(['ChassisId_encoded'], axis=1).values[0] # keep the 'static' features

        # questo prende le random len nelle time series, probabilmente da togliere dopo che nella seconda fase io tengo solo medium e high e ricavo da lì le time series in modo sistematico 
        random_start = 0
        random_end = len(time_series)
        # if  not self.test:
            
        #     if self.randomize_length and len(labels) > 5:
        #         random_len = np.random.randint(5, 15)#len(time_series))
        #         if self.val: random_len = 10

        #         remainder = len(time_series) - random_len
        #         random_start = np.random.randint(0, remainder) if remainder > 0 else 0
        #         random_end = random_start + random_len
            
        #     if not self.val:
        #         std = np.std(time_series, axis=0)
        #         noise = np.random.normal(0, 0.2*std, size=time_series.shape)
        #         time_series += noise

        # labels_binary = labels.copy()
        # labels_binary[:, 1] = labels_binary[:, 1] + labels_binary[:, 2]
        # labels_binary = labels_binary[:, :2]
        # labels = labels_binary
        # time_series = np.hstack([time_series, labels_binary])

        # time_series = time_series + self.getPositionEncoding(len(time_series), len(time_series[0]))
        time_series = np.hstack([time_series, self.getPositionEncoding(len(time_series), 50)])

        #time_series = np.hstack([time_series, ts['Timesteps'].to_numpy().reshape(-1,1)]) DA CAPIRE SE HA SENSO MANTENERLO 
        return torch.Tensor(time_series)[random_start: random_end], torch.Tensor(chassis_vector), torch.Tensor(label)

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

        # if not self.test:
            # self.volvo_df = self.volvo_df.loc[self.volvo_df['risk_level']!="Low"] DROPPA TUTTO CIO' CHE E' LOW 
  
        self.randomize_length = True

    def set_randomize(self, randomize):
        self.randomize_length = randomize

    def split_train_validation(self, train_ratio=0.8):
        all_indexes = list(range(len(self.df_list)))
        y_stratify = []
        for i, group_df in enumerate(self.df_list):
            if len(group_df['risk_level'].value_counts()) > 1:
                y_stratify.append(1)
            else:
                y_stratify.append(0)


        X_train, X_test, _, _ = train_test_split(all_indexes, all_indexes, train_size=train_ratio, stratify=y_stratify)
        validation_dataset = deepcopy(self)

        self.keep_indexes(X_train)
        validation_dataset.keep_indexes(X_test)

        # PARTE 1 + 2 
        # prende ogni time series e prende una window che salta di param every, prima di fare questo droppa tutte le low se la time series = 1, altrimenti tieni tutte le low 
        # check se ci sono solo medium o solo high (quindi no salto)
        # self.df_list = self.processor.split_in_len(
        #     self.df_list, 10, every=4
        # )

        # validation_dataset.df_list = validation_dataset.processor.split_in_len(
        #     validation_dataset.df_list, split_in_len=10, every=4
        # )
        validation_dataset.val = True

        # new_df_list = []
        # for i in tqdm.tqdm(range(len(self.df_list))):
        #     if len(self.df_list[i].value_counts()) > 1:
        #         new_df_list.append(self.df_list[i])

        # self.df_list = new_df_list


        return self, validation_dataset

    def keep_indexes(self, idx):
        self.df_list = [self.df_list[i] for i in idx]

    def get_processor(self):
        self.processor = Processor()
        self.volvo_df = self.processor.preprocess(self.volvo_df)
        self.processor.fit(self.volvo_df)        
        self.df_list = self.processor.group_by_chassis(self.volvo_df, skip_if_less_than=10, split_in_len=10)

        # PARTE 1 
        # dopo che hai fatto il group by chassis, dagli una label per dire se è sempre buono oppure no per poi trainare il classificatore binario 

        return self.processor
    
    # ESPERIMENTO 1: classificatore che divide buono da cattivo, se ho cattivo prima metà medium seconda high, se buona tutto low 

    def set_processor(self, processor):
        self.processor = processor
        self.volvo_df = self.processor.preprocess(self.volvo_df)
        self.df_list = self.processor.group_by_chassis(self.volvo_df)

    def get_n_features(self):
        assert self.volvo_df is not None
        features, variant, labels = self[0]
        return features.shape[-1]
    
    def get_n_classes(self):
        assert self.volvo_df is not None
        features, variant, labels = self[0]
        return labels.shape[-1]
        
        
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
            # SE IN FASE 1: UNICA LABEL 0 SE TUTTO LOW, 1 SE TUTTO MEDIUM/HIGH 
            # IN FASE 2 AVRO' SOLO MEDIUM E HIGH E VOGLIO 1 DOVE AVVIENE IL SALTO E 0 ALTRIMENTI  
            timestep_labels = ts["risk_level"]
            labels = self.processor.risk_encoder.transform(timestep_labels.values.reshape(-1, 1)).todense()
            if np.isnan(labels).any():
                print(np.isnan(labels).any())
                print(labels)
        elif self.test:
            #test data without risk_level as key in dataframe
            labels = np.empty((len(time_series),3))
            labels.fill(np.nan)

        chassis_vector = self.variants[self.variants['ChassisId_encoded'] == chassis_id].drop(['ChassisId_encoded'], axis=1).values[0] # keep the 'static' features

        # questo prende le random len nelle time series, probabilmente da togliere dopo che nella seconda fase io tengo solo medium e high e ricavo da lì le time series in modo sistematico 
        random_start = 0
        random_end = len(time_series)
        # if  not self.test:
            
        #     if self.randomize_length and len(labels) > 5:
        #         random_len = np.random.randint(5, 15)#len(time_series))
        #         if self.val: random_len = 10

        #         remainder = len(time_series) - random_len
        #         random_start = np.random.randint(0, remainder) if remainder > 0 else 0
        #         random_end = random_start + random_len
            
        #     if not self.val:
        #         std = np.std(time_series, axis=0)
        #         noise = np.random.normal(0, 0.2*std, size=time_series.shape)
        #         time_series += noise

        # labels_binary = labels.copy()
        # labels_binary[:, 1] = labels_binary[:, 1] + labels_binary[:, 2]
        # labels_binary = labels_binary[:, :2]
        # labels = labels_binary
        # time_series = np.hstack([time_series, labels_binary])

        # time_series = time_series + self.getPositionEncoding(len(time_series), len(time_series[0]))
        time_series = np.hstack([time_series, self.getPositionEncoding(len(time_series), 50)])
        #time_series = np.hstack([time_series, ts['Timesteps'].to_numpy().reshape(-1,1)]) DA CAPIRE SE HA SENSO MANTENERLO 
        return torch.Tensor(time_series)[random_start: random_end], torch.Tensor(chassis_vector), torch.Tensor(labels)[random_start: random_end]
    
    def getPositionEncoding(self, seq_len, d, n=1000):
        P = np.zeros((seq_len, d))
        for k in range(seq_len):
            for i in np.arange(int(d/2)):
                denominator = np.power(n, 2*i/d)
                P[k, 2*i] = np.sin(k/denominator)
                P[k, 2*i+1] = np.cos(k/denominator)
        return P

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