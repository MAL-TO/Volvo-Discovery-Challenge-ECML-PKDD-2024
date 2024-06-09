import numpy as np
import pandas as pd
import tqdm
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, StandardScaler
from dataset.dataset_utils import wavelet_spectrogram, stft_spectogram, classify_columns

class Processor:
    def __init__(self, scale=False):
        self.header_columns = ["Timesteps", "ChassisId_encoded", "gen", "risk_level"]
        self.columns_to_drop = ["f__51", "f__52", "f__65", "f__117", "f__119", "f__123",  "f__133", \
                                "af2__5",  "af2__6",  "af2__13",  "af2__18",  "af2__19",  "af2__20",  "af2__22", "f__22", \
                                'af1__5', 'af1__10', 'af1__22', 'af1__26', 'af1__30', 'f__10', 'f__15', 'f__26', 'f__36', 'f__90', 'f__91', 'f__98', 'f__104', 'f__120', 'f__122', 'f__127', 'f__129', 'f__140', 'f__152', 'f__153', 'f__154', 'f__158', 'f__169', 'f__170', 'f__172', 'f__173', 'f__174', 'f__175', 'f__176', 'f__177', 'f__178', 'f__188', 'f__198', 'f__199', 'f__203', 'f__207', 'f__214', 'f__215', 'f__219', 'f__228', 'f__231', 'f__236']
        self.risk_encoder = OneHotEncoder(
                categories=[['Low', 'Medium', 'High']]
                # categories=[['Low', 'High']]
                # categories=[['Medium', 'High']]
        )

    def fit(self, df, feature_classification_ratio=0.1):
        self.risk_encoder.fit(df["risk_level"].values.reshape(-1, 1))

        df_features = df.drop(self.header_columns, axis=1)
        self.feature_classification = classify_columns(df_features, ratio=feature_classification_ratio)

        self.num_features = np.array([1 if x=="numerical" else 0 for x in self.feature_classification]).astype(np.bool_)
        self.cat_features = np.logical_not(self.num_features)

    def preprocess(self, df):
        df = df.drop(self.columns_to_drop, axis=1, inplace=False)
        return df

    def group_by_chassis(self, df, skip_if_less_than=-1, split_in_len=-1):
        df_list = []
        groups = df.groupby("ChassisId_encoded")
        for name, group_df in tqdm.tqdm(groups):    
            if len(group_df) < skip_if_less_than: continue  

            df_list.append(group_df)

        return df_list
    
    def split_in_len(self, orig_df_list, split_in_len=10, every=2):
        
        df_list = []
        for group_df in orig_df_list:
            if split_in_len <= 0:
                df_list.append(group_df)
            elif split_in_len > 0 and len(group_df) < split_in_len:
                df_list.append(group_df)
            else:
                for i in range(0, len(group_df)-split_in_len, every):
                    start = len(group_df) - (i+split_in_len)
                    end = len(group_df) - (i)
                    if start < 0:
                        break
                    df_list.append(group_df.iloc[start:end])

                    # print(group_df.iloc[start:end])

        return df_list
    

    def extract_features(self, df):
        # df_headings, df_features = self.split_heading_features(df)
        df_features = df
        df_numerical_features = df_features.iloc[:, self.num_features]
        df_categorical_features = df_features.iloc[:, self.cat_features]

        # create first derivative
        diffs = df_numerical_features.diff(axis=1).fillna(0)
        diffs.columns = [x + "_diff" for x in df_numerical_features.columns]

        wavelet_df = wavelet_spectrogram(df_numerical_features, 3, wavelet="coif1")
        # wavelet_db8_df = wavelet_spectrogram(df_numerical_features, 3, wavelet="db4")
        # stft_df = stft_spectogram(df_numerical_features, window=5)

        enhanced_df = pd.concat([
                                df_numerical_features.reset_index(drop=True), 
                                df_categorical_features.reset_index(drop=True),
                                diffs.reset_index(drop=True),
                                wavelet_df.reset_index(drop=True),
                                # wavelet_db8_df.reset_index(drop=True),
                                # stft_df.reset_index(drop=True),
                                ], axis=1)
        
        return enhanced_df


    def split_heading_features(self, df):
        try:
            headings = df[self.header_columns]
        except Exception as e:
            headings = df[[x for x in self.header_columns if x != 'risk_level']]
        features = df.drop(self.header_columns, axis = 1, errors='ignore')
        return headings, features
    
    def __scaler(self):
        pass
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
        