# This file contains various dataset classes for PyTorch

import os
import torch
import torchaudio
from torch.utils.data import Dataset
import numpy as np
import pandas as pd

class MockDataSet(Dataset):

    def __init__(self, size):
        self.x_tensor = torch.randint(low=1, high=5, size=(size, 4)).type(torch.FloatTensor)
        self.y_tensor = torch.sum(self.x_tensor, dim=1).reshape(size, 1)

    def __getitem__(self, idx):
        x = self.x_tensor[idx]
        y = self.y_tensor[idx]
        return x, y

    def __len__(self):
        return self.x_tensor.shape[0]


class SpeechQualityDataset_mos(Dataset):
    """
        Main dataset class that loads the training dataset and contains the len and getitem functions required by Pytorch.
    """

    def __init__(self, df, data_dir, device, num_layers):
        self.df         = df
        self.data_dir   = data_dir
        self.dev        = device
        self.num_layers = num_layers

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        row = self.df.iloc[[idx]]
        path = [str(x) for x in row['filepath_deg']][0]
        wav_file_path = os.path.join(self.data_dir, path)
        mos = [float(x) for x in row['mos']][0]
        features = self.extract_features_xlsr53(wav_file_path)
        x_tensor = torch.stack(features, dim=0)
        x_tensor = torch.squeeze(x_tensor)
        x_tensor = torch.flatten(x_tensor[:,:200,:])
        x = [x_tensor, row.values.flatten().tolist()]
        y_mos = self.df['mos'].iloc[idx].reshape(-1).astype('float32') 
        y_noi = self.df['noi'].iloc[idx].reshape(-1).astype('float32')
        y_dis = self.df['dis'].iloc[idx].reshape(-1).astype('float32')         
        y_col = self.df['col'].iloc[idx].reshape(-1).astype('float32')                
        y_loud = self.df['loud'].iloc[idx].reshape(-1).astype('float32')                
        y = np.concatenate((y_mos, y_noi, y_dis, y_col, y_loud), axis=0)
        return x, mos, y

    def extract_features_xlsr53(self, path):
        bundle                  = torchaudio.pipelines.WAV2VEC2_XLSR53
        model                   = bundle.get_model().to(self.dev)
        waveform, sample_rate   = torchaudio.load(path)
        waveform                = waveform.to(self.dev)
        if sample_rate != bundle.sample_rate:
            waveform            = torchaudio.functional.resample(waveform, sample_rate, bundle.sample_rate)
        with torch.inference_mode():
            features, _         = model.extract_features(waveform, num_layers=self.num_layers)
        return features
    
#################################################################################
# Simple Dataset class
#################################################################################

class SqaDataset(Dataset):
    def __init__(self, data_dir, df, sample_rate):
        super().__init__()
        self.data_dir       = data_dir
        self.df             = df
        self.sample_rate    = sample_rate

    def __getitem__(self, idx):
        file_path = os.path.join(self.data_dir, self.df['filepath_deg'].iloc[idx])
        waveform, sample_rate = torchaudio.load(file_path)
        if sample_rate != self.sample_rate:
            waveform            = torchaudio.functional.resample(waveform, sample_rate, self.sample_rate)
        

        print(f'\n\nsize rate of wav: {waveform.shape}\n\n')
        x = waveform






        y_mos   = self.df['mos'].iloc[idx].reshape(-1).astype('float32') 
        y_noi   = self.df['noi'].iloc[idx].reshape(-1).astype('float32')
        y_dis   = self.df['dis'].iloc[idx].reshape(-1).astype('float32')         
        y_col   = self.df['col'].iloc[idx].reshape(-1).astype('float32')                
        y_loud  = self.df['loud'].iloc[idx].reshape(-1).astype('float32')                
        y       = np.concatenate((y_mos, y_noi, y_dis, y_col, y_loud), axis=0)
        
        return x, y

    def __len__(self):
        return len(self.df)