# Main script for training a model

import argparse
import yaml
import datetime
import torch
import os
import csv
import pandas as pd
import sys
import torchaudio
import lib
from torch.utils.data import DataLoader

parser = argparse.ArgumentParser()
parser.add_argument('--yaml', required=True, type=str, help='YAML file with config')
args = parser.parse_args()
args = vars(args)

if __name__ == "__main__":

    with open(args['yaml'], "r") as ymlfile:
        args_yaml = yaml.load(ymlfile, Loader=yaml.FullLoader)
    args = {**args_yaml, **args}

    print(f'\n\n-----------------------------\n\tModel Training\n-----------------------------')

    # load data
    tr_df, trainset = lib.load_data(args['data_dir'], args['csv_file'], args['csv_db_train'], 
                             args['sample_rate'], args['max_length'])
    val_df, valset = lib.load_data(args['data_dir'], args['csv_file'], args['csv_db_val'], 
                           args['sample_rate'], args['max_length'])

    trainloader = DataLoader(trainset, batch_size=args['batch_size'], shuffle=False, collate_fn=lib.pad_collate)
    valloader = DataLoader(valset, batch_size=args['batch_size'], shuffle=False, collate_fn=lib.pad_collate)

    # load model
    

    # train loop for each loop

    for epoch in range(args['max_epochs']):

        # train loop
        for xx_pad, yy, idx in trainloader:
            print(f'indices {idx} batch_x shape is {xx_pad.shape}, batch_y shape is {len(yy)}')


        # validation loop

        # early stop check

        # save model

        

        # save model