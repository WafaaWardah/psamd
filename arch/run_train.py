# Main training script, Wafaa Wardah, TU-Berlin, June 2023

import argparse
import yaml
import datetime
import torch
import os
import csv
import datasets
import architectures
import pandas as pd
import sys
import torchaudio
from torch.utils.data import DataLoader


###################################################################################################
# main class that runs the training and validation loop
###################################################################################################

class Run(): 

    def __init__(self, args):
        self.device         = self._get_device()
        self.start_time     = datetime.datetime.today()
        self.name           = args['name']
        self.data_dir       = args['data_dir']
        self.output_dir     = args['output_dir']
        self.csv_file       = args['csv_file']
        self.batch_size     = args['batch_size']
        self.num_epochs     = args['num_epochs']
        self.csv_db_train   = args['csv_db_train']
        self.csv_db_val     = args['csv_db_val']
        self.num_layers     = args['num_layers']
        self.sample_rate    = args['sample_rate']

        print(f"\n\nInitiating run < {self.name} >\n\n")

        self._make_run_files()
        self._get_datasets()
    
        # Load model
        self.model = architectures.NiSQA_Dim()

        #print(f'\nModel:\n{self.model}')
        self._save_log('\nmodel', self.model.feature_extractor, newline=True)

        self._train()
        

    def _get_device(self):
        if torch.cuda.is_available():
            return torch.device("cuda")
        else:
            return torch.device("cpu")
        

    def _make_run_files(self):
        new_dir = os.path.join(self.output_dir, self.name + self.start_time.strftime("_%Y%m%d_%H%M%S"))
        if not os.path.exists(new_dir): os.mkdir(new_dir)
        self.csv_file_path = os.path.join(new_dir, 'results.csv')
        self.txt_log_path = os.path.join(new_dir, 'log.txt')
        print(f'-> Creating results csv file {self.csv_file_path}, and log txt file {self.txt_log_path}')
        with open(self.csv_file_path, 'w', newline='', encoding='utf-8') as csv_file:
            header = ['epoch', 'patience', 'tr_loss', 'val_loss']
            writer = csv.writer(csv_file)
            writer.writerow(header)

        self._save_log('name', self.name)
        self._save_log('start_time', self.start_time.strftime("%d.%m.%Y %H:%M:%S"))
        self._save_log('device', self.device)
        self._save_log('batch_size', self.batch_size)
        self._save_log('num_epochs', self.num_epochs)

        

    def _get_datasets(self):

        csv_file_path = os.path.join(self.data_dir, self.csv_file)
        dfile = pd.read_csv(csv_file_path)

        if not set(self.csv_db_train + self.csv_db_val).issubset(dfile.db.unique().tolist()):
            missing_datasets = set(self.csv_db_train + self.csv_db_val).difference(dfile.db.unique().tolist())
            raise ValueError('Not all dbs found in csv:', missing_datasets)
        
        train_df = dfile[dfile.db.isin(self.csv_db_train)].reset_index()
        val_df = dfile[dfile.db.isin(self.csv_db_val)].reset_index()
        
        self.trainset = datasets.SqaDataset(self.data_dir, train_df, self.sample_rate)
        self.valset = datasets.SqaDataset(self.data_dir, val_df, self.sample_rate)

    def _save_log(self, heading, info, newline=False):
        if newline: info = '\n' + str(info) 
        else: info = str(info)
        with open(self.txt_log_path, 'a') as log_file:
            log_file.write('\n' + str(heading) + ': ' + info)

    
    def _train(self):
        self.trainloader = DataLoader(self.trainset, shuffle=True, batch_size=self.batch_size)
        self.valloader = DataLoader(self.valset, shuffle=False, batch_size=self.batch_size)

        counter = 0
        counter_check = 5 

        for i, x in enumerate(self.trainloader):
            counter += 1
            print(f'i={i} x={x}')

            if counter == counter_check: break



###################################################################################################
# main
###################################################################################################

parser = argparse.ArgumentParser()
parser.add_argument('--yaml', required=True, type=str, help='YAML file with config')
args = parser.parse_args()
args = vars(args)

if __name__ == '__main__':
    with open(args['yaml'], "r") as ymlfile:
        args_yaml = yaml.load(ymlfile, Loader=yaml.FullLoader)
    args = {**args_yaml, **args}

    train = Run(args)

######################### END #####################################################################
