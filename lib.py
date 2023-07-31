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

    def __init__(self, df):
        super().__init__()

        self.df = df

    def __getitem__(self, index):
        return self.df(index)
    

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

def get_set(data_dir, csv_file, db_dict, sample_rate, max_length):
    # load dataframes
    dfile = pd.read_csv(os.path.join(data_dir, csv_file))
    if not set(db_dict).issubset(dfile.db.unique().tolist()):
        missing_datasets = set(db_dict).difference(dfile.db.unique().tolist())
        raise ValueError('Not all dbs found in csv:', missing_datasets)
    df = dfile[dfile.db.isin(db_dict)].reset_index()
    # instantiate Pytorch data class 
    return df, DimDataset(df, data_dir, sample_rate, max_length)


def load_data(dir, csv, tr_db, val_db, sr, max_len, feature_extractor):
    print(f'\nLoading training set...')
    dfile = pd.read_csv(os.path.join(dir, csv))
    if not set(tr_db + val_db).issubset(dfile.db.unique().tolist()):
        missing_datasets = set(tr_db + val_db).difference(dfile.db.unique().tolist())
        raise ValueError('Not all dbs found in csv:', missing_datasets)
    tr_df = dfile[dfile.db.isin(tr_db)].reset_index()
    val_df = dfile[dfile.db.isin(val_db)].reset_index()

    print(f'\n\nLoading datasets - created dfs.\n\nTrain df is \n{tr_df}\Val df is \n{val_df}')
    
    tr_set = SQAdataset(tr_df) # dataset with feature extraction performed in getitem()
    val_set = SQAdataset(val_df)

    return  tr_df, tr_set, val_df, val_set


def pad_collate(batch):
    (xx, yy, idx) = zip(*batch)
    xx = [x.squeeze(0) for x in xx]
    x_lens = [len(x) for x in xx]

    #y_lens = [len(y) for y in yy]
    xx_pad = pad_sequence(xx, batch_first=True, padding_value=0)
    return xx_pad, x_lens, yy, idx