# A simple PyTorch Machine Learning script (Template) by Wafaa Wardah, TU-Berlin 2022.

import csv
import os
import pandas as pd
from tqdm import tqdm
import yaml
import torch
import argparse
import datetime
from torch.optim import Adam
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import datasets
import architectures


class Run(): # <--------------------------------- main class that runs the training, validation and test loop

    def __init__(self, args):

        self.name = args['name']
        self.batch_size = args['batch_size']
        self.num_epochs = args['num_epochs']
        self.early_stop_patience = args['early_stop_patience']
        self.data_dir = args['data_dir']
        self.csv_file = args['csv_file'] 
        self.csv_db_train = args['csv_db_train'] # list
        self.csv_db_val = args['csv_db_val'] # list
        self.num_layers = args['num_layers']
        self.device = self._get_device()
        self.start_time = datetime.datetime.today()
        print(f'\n-> Initiating run <{self.name}> at time {self.start_time}')

        # Start logging experiment
        self.exp_dir, self.result_file, self.log_file = self._make_run_files()

        # Load datasets
        '''
        self.trainset = datasets.MockDataSet(8000)
        self.valset = datasets.MockDataSet(2000)
        '''
        self.trainset = self.load_datasets(self.csv_db_train)
        self.valset = self.load_datasets(self.csv_db_val)

        # Log dataset info
        with open(self.log_file, 'a') as f: f.write('\n' + '-'*115)
        self._log_dataset_info('Training', self.trainset)
        self._log_dataset_info('Validation', self.valset)

        # Create dataloader
        self.trainloader = DataLoader(dataset=self.trainset, shuffle=True, batch_size=self.batch_size)
        self.valloader = DataLoader(dataset=self.valset, shuffle=False, batch_size=self.batch_size)

        # Initialize model (weights)
        in_features = self.trainset.__getitem__(0)[0][0].shape[0]  # number of features per sample
        out_features = 1  # number of output (output dimension)
        self.model = architectures.Linear1(in_features, out_features)
        with open(self.log_file, 'a') as f: f.write('\n' + '-'*115)
        with open(self.log_file, 'a') as f: f.write('\nModel Architecture:\n' + str(self.model) + '\n' + '-'*115)
        # Initialize criterion for calculating the error in prediction (loss function)
        self.criterion = torch.nn.MSELoss(reduction='none')
        # Initialize optimizer
        self.optimizer = Adam(self.model.parameters(), lr=0.00001)

        # Training loop
        self._train_epochs()

        # Test
        #self._run_test()

        with open(self.log_file, 'a') as f:
            f.write('\n' + '-' * 115)
            f.write('\nTime to train: ' +
                str((self.end_time - self.start_time).total_seconds()) + ' seconds')
            f.write('\n' + '-' * 115)

    def load_datasets(self, csv_db):
        csv_file_path = os.path.join(self.data_dir, self.csv_file)
        dfile = pd.read_csv(csv_file_path)
        if not set(csv_db).issubset(dfile.db.unique().tolist()):
            missing_datasets = set(csv_db).difference(dfile.db.unique().tolist())
            raise ValueError('Not all dbs found in csv:', missing_datasets)
        df = dfile[dfile.db.isin(csv_db)].reset_index()
        dataset = datasets.SpeechQualityDataset_mos(df, self.data_dir, self.device, self.num_layers)
        return dataset


    def _get_device(self):
        if torch.cuda.is_available():
            return torch.device("cuda")
        else:
            return torch.device("cpu")

    def _make_run_files(self):
        cwd = os.getcwd()
        new_dir = os.path.join(cwd, self.name + '_' + self.start_time.strftime("%Y%m%d_%H%M%S"))
        if not os.path.exists(new_dir): os.mkdir(new_dir)
        csv_file_path = os.path.join(new_dir, 'results.csv')
        txt_log_path = os.path.join(new_dir, 'log.txt')
        print(f'-> Creating results csv file {csv_file_path},\n\tand log txt file {txt_log_path}')
        with open(csv_file_path, 'w', newline='', encoding='utf-8') as csv_file:
            header = ['epoch', 'patience', 'tr_loss', 'val_loss']
            writer = csv.writer(csv_file)
            writer.writerow(header)
        with open(txt_log_path, 'w') as log_file:
            log_file.write(self.name + ' ' + self.start_time.strftime("%d.%m.%Y %H:%M:%S"))
            log_file.write('\nDevice: ' + str(self.device) + '\nBatch size: ' + str(self.batch_size) +
                           '\nNumber of epochs: ' + str(self.num_epochs))
        return new_dir, csv_file_path, txt_log_path

    def _print_dataset_info(self, ds):
        ds_len = ds.__len__()
        ds_sample_x = ds.__getitem__(0)[0]
        ds_sample_y = ds.__getitem__(0)[1]
        print(f'\t\tdataset length:{ds_len} samples'
              f'\t\tsample x: {ds_sample_x.shape}'
              f'\t\tsample y: a float')

    def _log_dataset_info(self, setname, ds):
        ds_len = ds.__len__()
        ds_sample_x = ds.__getitem__(0)[0][0]
        ds_sample_y = ds.__getitem__(0)[1]
        with open(self.log_file, 'a') as f:
            f.write('\n' + setname +
                    '\t\tdataset length:' + str(ds_len) + ' samples' +
                    '\t\tsample x: ' + str(ds_sample_x.shape) +
                    '\t\tsample y: a float')

    def _train_epochs(self):  # <----------------------------------- main training & validation loop
        print('-> Training')
        tr_epoch_loss_list = []
        val_epoch_loss_list = []

        # For early stopping
        running_loss = 100.0
        running_best_epoch = 0

        for epoch in tqdm(range(self.num_epochs)):
            tr_loss = self._training()
            tr_epoch_loss_list.append(tr_loss)  # <---------------------------Training loop
            current_loss = self._validation()            # <-------------------------Validation loop
            val_epoch_loss_list.append(current_loss)
            self.model_path = os.path.join(self.exp_dir, 'saved_model_' + str(epoch) + '.pth')
            torch.save(self.model.state_dict(), self.model_path)

            self._save_results(epoch, self.early_stop_patience, tr_loss, current_loss)

            # Early stop check
            if current_loss < running_loss:
                running_best_epoch = epoch
                running_loss = current_loss
                self.early_stop_patience = args['early_stop_patience']
            else:
                self.early_stop_patience -= 1

            if (self.early_stop_patience == 0) or (epoch == self.num_epochs - 1):
                self.best_model = 'saved_model_' + str(running_best_epoch) + '.pth'
                with open(self.log_file, 'a') as f:
                    f.write('\n' + '-'*115)
                    f.write('\nBest model is saved_model_' + str(running_best_epoch) + '.pth')
                    f.write(', and training continued til epoch ' + str(epoch))
                break

        self.end_time = datetime.datetime.today()
        print('   Done')
        print(f'-> Plotting loss')
        self._plot_losses(tr_epoch_loss_list, val_epoch_loss_list)
        self._log_loss(mode='Training', lst=tr_epoch_loss_list, ep=running_best_epoch)
        self._log_loss(mode='Validation', lst=val_epoch_loss_list, ep=running_best_epoch) # ------------------------------

    def _log_loss(self, mode, lst, ep):
        with open(self.log_file, 'a') as logfile:
            row = '\nBest epoch ' + str(ep) + ' - ' + mode + ' loss: \t' + str(lst[ep])
            logfile.write(row)

    def _plot_losses(self, train_loss_list, val_loss_list):
        plt.plot(train_loss_list, label='Training loss')
        plt.plot(val_loss_list, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Loss per Epoch')
        plt.legend()
        plt.savefig(os.path.join(self.exp_dir, 'epoch_losses.png'))

    def _training(self):  # <---------------------------------------------------- main training loop
        tr_batch_loss_list = []

        #dev_counter = 0


        for iteration, (X, Y) in enumerate(self.trainloader):
            #dev_counter += 1
            inputs = X[0].to(self.device)
            #targets = torch.tensor(Y).to(self.device)
            #targets = targets.float()
            targets = Y.to(self.device)
            targets = torch.unsqueeze(targets, dim=-1)
            inputs, targets = inputs.float(), targets.float()
            self.model.to(self.device)
            self.model.zero_grad()
            self.model.train()
            outputs = self.model(inputs)
            loss_each = self.criterion(outputs, targets)
            loss_all = torch.mean(loss_each)
            loss_all.backward()
            self.optimizer.step()
            # print(f'batch ie iteration {iteration}\tloss {loss_all.item()}')
            tr_batch_loss_list.append(round(loss_all.item(), 3))

            #if dev_counter == 8: break

        tr_epoch_avg_loss = sum(tr_batch_loss_list) / len(tr_batch_loss_list)
        return round(tr_epoch_avg_loss, 3)  # ------------------------------------------------------

    def _validation(self):
        self.model.to(self.device)
        self.model.eval()
        return self._predict(self.valloader)


    def _predict(self, dl):

        #dev_counter = 0

        pred_batch_loss_list = []
        for iteration, (X, Y) in enumerate(dl):
            #dev_counter += 1
            inputs = X[0].to(self.device)
            targets = Y.to(self.device)
            targets = torch.unsqueeze(targets, dim=-1)
            inputs, targets = inputs.float(), targets.float()
            outputs = self.model(inputs)

            print(f'{iteration}\t{outputs}')

            loss_each = self.criterion(outputs, targets)
            loss_all = torch.mean(loss_each)

            pred_batch_loss_list.append(round(loss_all.item(), 2))
            #if dev_counter == 8: break
            #self._save_results(iteration, inputs, targets, outputs, loss_each.detach())
        pred_epoch_avg_loss = sum(pred_batch_loss_list) / len(pred_batch_loss_list)
        return round(pred_epoch_avg_loss, 2)

    def _run_test(self):
        loss = self._testing()
        print(f'Test loss: {loss}')
        with open(self.log_file, 'a') as logfile: logfile.write('\nBest model Test loss: \t' + str(loss))

    def _save_results(self, epoch, patience, tr_loss, val_loss):
        with open(self.result_file, 'a', newline='', encoding='utf-8') as f:
            row = [str(epoch), str(patience), str(tr_loss), str(val_loss)]
            writer = csv.writer(f)
            writer.writerow(row)

# -----------------------------------------------------------------------------------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument('--yaml', required=True, type=str, help='YAML file with config')
args = parser.parse_args()
args = vars(args)

if __name__ == '__main__':
    with open(args['yaml'], "r") as ymlfile:
        args_yaml = yaml.load(ymlfile, Loader=yaml.FullLoader)
    args = {**args_yaml, **args}

    run = Run(args)

