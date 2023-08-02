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
from torch.optim import Adam

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
    tr_df, trainset, val_df, valset = lib.load_data(args['device'], args['data_dir'], args['csv_file'], 
                                    args['csv_db_train'], args['csv_db_train'], 
                                    args['sample_rate'], args['max_length'])

    x, y, idx = trainset.__getitem__(0)

    print(f'idx: {idx}, x: {len(x)}, y: {len(y)}')
    
    trainloader = DataLoader(trainset, batch_size=args['batch_size'], shuffle=False) #, collate_fn=lib.pad_collate

    for b_num, xx, yy in enumerate(trainloader):
        print(f'batch number {b_num}, len of xx {len(xx)}, len of yy {len(yy)}')

    sys.exit()
    valloader = DataLoader(valset, batch_size=args['batch_size'], shuffle=False, collate_fn=lib.pad_collate)

    # load model
    model = lib.ModelDim()


    criterion = torch.nn.MSELoss(reduction='none')
    optimizer = Adam(model.parameters(), lr=0.00001)

    # train loop for each loop

    for epoch in range(args['max_epochs']):

        # train loop
        for b, (xx_pad, yy, idx) in enumerate(trainloader):
            print(f'\n\nbatch {b} indices {idx} batch_x shape is {xx_pad.shape}, batch_y shape is {len(yy)}\n')
            outputs = model(xx_pad)

            outputs2 = model(xx_pa)

            print(len(outputs))
            print(len(outputs2))


            for i, j in zip(outputs, outputs2):

                print(f'i shape {i.shape}, j shape {j.shape}')

        
        sys.exit()

        # validation loop

        # early stop check

        # save model

        

        # save model