# Library of classes and functions used.
import os
import sys
import pandas as pd
import numpy as np
import torch
import torchaudio
import torch.nn as nn
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

#--------------------------- dataset class ----------------------------------------
class DimDataset(Dataset):

    def __init__(self, df, data_dir, sample_rate, max_length):
        super().__init__()
        self.df = df
        self.data_dir = data_dir
        self.sample_rate = sample_rate
        self.max_length = max_length

    def __getitem__(self, idx):
        file_path = os.path.join(self.data_dir, self.df['filepath_deg'].iloc[idx])
        waveform, sample_rate = torchaudio.load(file_path)
        if sample_rate != self.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sample_rate, self.sample_rate)
        if waveform.shape[1] > self.max_length:
            waveform = waveform[:,:self.max_length]
        y_mos = self.df['mos'].iloc[idx].reshape(-1).astype('float32') 
        y_noi = self.df['noi'].iloc[idx].reshape(-1).astype('float32')
        y_dis = self.df['dis'].iloc[idx].reshape(-1).astype('float32')         
        y_col = self.df['col'].iloc[idx].reshape(-1).astype('float32')                
        y_loud = self.df['loud'].iloc[idx].reshape(-1).astype('float32')                
        y = np.concatenate((y_mos, y_noi, y_dis, y_col, y_loud), axis=0)
        return waveform, y, idx

    def __len__(self):
        return len(self.df)
    

class SQAdataset(Dataset):

    def __init__(self, df, device, dir):
        super().__init__()

        self.df = df
        self.device = get_device(device)
        self.dir = dir
        self.bundle = torchaudio.pipelines.WAV2VEC2_XLSR53
        self.model = self.bundle.get_model().to(self.device)

    def __getitem__(self, idx):
        row = self.df.iloc[[idx]]
        path = [str(x) for x in row['filepath_deg']][0]
        wav_file_path = os.path.join(self.dir, path)
        #mos = [float(x) for x in row['mos']][0]
        features = self.extract_features_xlsr53(wav_file_path)
        x_tensor = torch.stack(features, dim=0)
        x_tensor = torch.squeeze(x_tensor)
        x_tensor = torch.flatten(x_tensor)
        x = [x_tensor, row.values.flatten().tolist()]
        y_mos = self.df['mos'].iloc[idx].reshape(-1).astype('float32') 
        y_noi = self.df['noi'].iloc[idx].reshape(-1).astype('float32')
        y_dis = self.df['dis'].iloc[idx].reshape(-1).astype('float32')         
        y_col = self.df['col'].iloc[idx].reshape(-1).astype('float32')                
        y_loud = self.df['loud'].iloc[idx].reshape(-1).astype('float32')                
        y = np.concatenate((y_mos, y_noi, y_dis, y_col, y_loud), axis=0)
        return x, y, idx

    def extract_features_xlsr53(self, path):
        waveform, sample_rate = torchaudio.load(path)
        waveform = waveform.to(self.device)
        if sample_rate != self.bundle.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sample_rate, self.bundle.sample_rate)
        with torch.inference_mode():
            features, _ = self.model.extract_features(waveform, num_layers=self.num_layers)
        return features
    
    def __len__(self):
        return len(self.df)
    

#--------------------------- model class ----------------------------------------
class ModelDim(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.bundle = torchaudio.pipelines.WAV2VEC2_XLSR53
        self.feature_extractor = self.bundle.get_model()


    def forward(self, x_padded, x_lens):

        for n, (p, l) in enumerate(zip(x_padded, x_lens)):
            print(n, len(p), l)

        with torch.inference_mode():
            features, _ = self.feature_extractor.extract_features(x_padded, num_layers=7)
        y_pred = torch.stack(features, dim=0)
        y_pred = torch.squeeze(y_pred)

        print(f'length of y_pred: {len(y_pred)}')
        return y_pred



#--------------------------- functions ----------------------------------------

def get_device(device):   
        if torch.cuda.is_available() and device == "cuda":
            print(f'\nDevice is cuda.')
            return torch.device("cuda")
        else:
            print(f'\nDevice is cpu.')
            return torch.device("cpu")
    

def get_set(data_dir, csv_file, db_dict, sample_rate, max_length):
    # load dataframes
    dfile = pd.read_csv(os.path.join(data_dir, csv_file))
    if not set(db_dict).issubset(dfile.db.unique().tolist()):
        missing_datasets = set(db_dict).difference(dfile.db.unique().tolist())
        raise ValueError('Not all dbs found in csv:', missing_datasets)
    df = dfile[dfile.db.isin(db_dict)].reset_index()
    # instantiate Pytorch data class 
    return df, DimDataset(df, data_dir, sample_rate, max_length)


def load_data(device, dir, csv, tr_db, val_db, sr, max_len):
    print(f'\nLoading training set...')
    dfile = pd.read_csv(os.path.join(dir, csv))
    if not set(tr_db + val_db).issubset(dfile.db.unique().tolist()):
        missing_datasets = set(tr_db + val_db).difference(dfile.db.unique().tolist())
        raise ValueError('Not all dbs found in csv:', missing_datasets)
    tr_df = dfile[dfile.db.isin(tr_db)].reset_index()
    val_df = dfile[dfile.db.isin(val_db)].reset_index()

    print(f'\n\nLoading datasets - created dfs.\n\nTrain df is \n{tr_df}\Val df is \n{val_df}')
    
    tr_set = SQAdataset(tr_df, device, dir) # dataset with feature extraction performed in getitem()
    val_set = SQAdataset(val_df, device, dir)

    return  tr_df, tr_set, val_df, val_set


def pad_collate(batch):
    (xx, yy, idx) = zip(*batch)
    xx = [x.squeeze(0) for x in xx]
    x_lens = [len(x) for x in xx]

    #y_lens = [len(y) for y in yy]
    xx_pad = pad_sequence(xx, batch_first=True, padding_value=0)
    return xx_pad, x_lens, yy, idx