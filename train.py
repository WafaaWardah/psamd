# Main tscript for training a model

import argparse
import yaml
import datetime
import torch
import os
import csv
import pandas as pd
import sys
import torchaudio
from torch.utils.data import DataLoader

parser = argparse.ArgumentParser()
parser.add_argument('--yaml', required=True, type=str, help='YAML file with config')
args = parser.parse_args()
args = vars(args)

if __name__ == "__main__":

    print(f'\n\n-----------------------------\n\tModel Training\n-----------------------------')

    # load data

    # load model

    # train loop for each loop

        # train loop

        # validation loop

        # early stop check

        # save model

        

        # save model