# Library of classes and functions used.
import os
import sys
import pandas as pd
import numpy as np
import torchaudio
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
    

#--------------------------- model class ----------------------------------------



#--------------------------- functions ----------------------------------------

def load_data(data_dir, csv_file, db_dict, sample_rate, max_length):
    # load dataframes
    dfile = pd.read_csv(os.path.join(data_dir, csv_file))
    if not set(db_dict).issubset(dfile.db.unique().tolist()):
        missing_datasets = set(db_dict).difference(dfile.db.unique().tolist())
        raise ValueError('Not all dbs found in csv:', missing_datasets)
    df = dfile[dfile.db.isin(db_dict)].reset_index()
    # instantiate Pytorch data class 
    return df, DimDataset(df, data_dir, sample_rate, max_length)

def pad_collate(batch):
    (xx, yy, idx) = zip(*batch)
    xx = [x.squeeze(0) for x in xx]
    #x_lens = [len(x) for x in xx]
    #y_lens = [len(y) for y in yy]
    xx_pad = pad_sequence(xx, batch_first=True, padding_value=0)
    return xx_pad, yy, idx