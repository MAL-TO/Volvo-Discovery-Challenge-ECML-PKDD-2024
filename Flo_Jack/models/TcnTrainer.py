import sys 
sys.path.append("../")

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

from pathlib import *
import os
import tqdm
import argparse
import json

from models.tcn import MS_TCN
from dataset.VolvoDataset import VolvoDataset
from utils.ContinuityCrossEntropyLoss import ContinuityCrossEntropyLoss
from utils.StatsComputer import StatsComputer

class TcnTrainer:
    def __init__(self, args):
        self.args = args

        ### Set seed
        np.random.seed(self.args.seed)
        torch.manual_seed(self.args.seed)

        ### Get important paths
        self.curr_dir = os.path.dirname( os.path.abspath(__file__) )
        self.home_path = Path(self.curr_dir).parent.absolute()
        self.dataset_path = self.args.data_path 
        self.variants_path = os.path.join(self.dataset_path, self.args.variants_csv)
        self.train_data_path = os.path.join(self.dataset_path, self.args.train_csv)
        self.test_data_path = os.path.join(self.dataset_path, self.args.test_csv)
        self.weights_path = self.args.tcnn_weights
        os.makedirs(self.weights_path, exist_ok=True)
        
        ### Get dataset and model type
        self.num_classes = 3

        self.train_dataset = VolvoDataset(data_path=self.train_data_path, variants_path=self.variants_path)
        self.processor = self.train_dataset.get_processor()
        self.label_encoder = self.processor.risk_encoder

        self.train_dataset, self.validation_dataset = self.train_dataset.split_train_validation()
    
        self.test_dataset = VolvoDataset(data_path=self.test_data_path, variants_path=self.variants_path, test=True)
        self.test_dataset.set_processor(self.processor) 
     
        
        n_features = self.train_dataset.get_n_features()
        
        #check if preprocess is giving some problems
        assert self.train_dataset.get_n_features() == self.test_dataset.get_n_features()
        
        self.model = MS_TCN(num_stages=0, 
                            num_input_channels=n_features, 
                            num_classes=self.num_classes)

        ### Get device
        self.device = torch.device(
                    "cuda" if (torch.cuda.is_available() and not self.args.disable_cuda) else "cpu"
                )
        self.model.to(self.device)
        print(f"Working on {self.device}")
        
        # Load weights if necessary
        if self.args.load_model != "":
            if not(self.args.load_model.endswith(".pth") or self.args.load_model.endswith(".pt")):
                raise Exception("Weights file should end with .pt or .pth")
            model_path = os.join.path(self.weights_path, self.args.load_model)
            print(f"Loading Model from {model_path}")
            self.model.load_state_dict(
                torch.load( model_path )
            )

        # Create DataLoader instances for train, validation, and test sets
        self.train_loader = DataLoader(self.train_dataset, 
                                       batch_size=self.args.batch_size, 
                                       collate_fn = VolvoDataset.padding_collate_fn,
                                       shuffle=True,
                                       num_workers=12) #pin_memory=True #consigliano
        self.val_loader = DataLoader(self.validation_dataset, 
                                     batch_size=self.args.batch_size, 
                                     collate_fn = VolvoDataset.padding_collate_fn, 
                                     shuffle=True,
                                     num_workers=12)
        self.test_loader =  DataLoader(self.test_dataset, 
                                       batch_size=self.args.batch_size,
                                       collate_fn = VolvoDataset.padding_collate_fn)
        
        # Define criterion
        print('Computing class weights...', end='')
        y = self.label_encoder.transform( self.train_dataset.volvo_df[["risk_level"]].values )
        y = np.argmax(y, axis=1).flatten()
        y = np.array(y)[0]

        weights = compute_class_weight(class_weight="balanced", classes=np.unique(y), y=y)
        weights = weights 
        self.criterion = ContinuityCrossEntropyLoss(
            weights=torch.Tensor(weights).to(self.device)
            )
        print('done')
        print('Class weights = ', weights)
        # self.criterion = ContinuityCrossEntropyLoss(weights=torch.Tensor([1,1,1]).to(self.device))


    def reset_optimizer(self, lr):
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, 
                                                    gamma=self.args.lr_scheduler_gamma, 
                                                    step_size=self.args.lr_scheduler_step)

    def train(self):
        print("=== Start training ===")
        print(f"Batch size: {self.args.batch_size}")
        # Define loss function and optimizer
        self.reset_optimizer(self.args.learning_rate)

        waiting_epochs = 0
        best_val_loss = float('inf')
        best_val_f1 = 0
        num_resets = 0
        for epoch in range(self.args.num_epochs):
            ### Run epoch
            print( "="*25, f"EPOCH {epoch}", "="*25)
            print("Learning rate: ", self.optimizer.param_groups[0]['lr'])
            epoch_loss = self.run_epoch()
            self.scheduler.step()
            print(f"Epoch [{epoch+1}/{self.args.num_epochs}], Train Loss: {epoch_loss:.4f}")

            ### Run validation
            validation_loss, validation_accuracy, validation_f1 = self.run_validation()            
            # print(f"Validation Accuracy: {validation_accuracy:.4f}")
            print(f"Validation Loss: {validation_loss:.4f} vs Best {best_val_loss:.4f}")
            print(f"Validation F1: {validation_f1} vs Best {best_val_f1}")
            
            
            ### Save model if best and early stopping
            # if validation_loss < best_val_loss:
            #     print(f"Saving new model [New best loss {validation_loss} vs Old best loss {best_val_loss}]")
            if validation_f1 > best_val_f1 or (validation_f1 == best_val_f1 and validation_loss < best_val_loss):
                print(f"Saving new model \n", 
                      f"[New best loss {validation_loss} vs Old best loss {best_val_loss}] \n ",
                      f"[New best f1   {validation_f1} vs Old best f1   {best_val_f1}]")
                best_val_loss = validation_loss
                best_val_f1 = validation_f1
                waiting_epochs = 0
                torch.save(self.model.state_dict(), os.path.join(self.weights_path, "TCN_best.pth"))
            else:
                waiting_epochs += 1
                if waiting_epochs >= self.args.patience_epochs:
                    print(f"Early stopping because ran more than {self.args.patience_epochs} without improvement")
                    break
                    # waiting_epochs = 0
                    # num_resets += 1
                    # lr = (0.8**num_resets) * self.args.learning_rate
                    # self.reset_optimizer(lr) 
                    # print("Resetting LR")

    def run_epoch(self):
        self.model.train()
        
        pbar = tqdm.tqdm(self.train_loader)
        running_loss = 0
        i = 0

        for timeseries, variants, labels, mask in pbar:
            pbar.set_description(f"Running loss: {running_loss/(i+1e-5) :.4}")
            
            timeseries, variants, labels = timeseries.to(self.device), variants.to(self.device), labels.to(self.device)

            outputs = self.model(timeseries, variants)

            self.optimizer.zero_grad()
            
            loss = 0
            for o in outputs:
                loss += self.criterion(o, labels, mask)
                if torch.isnan(loss).any():
                    print(torch.isnan(o).any())
                    print(torch.isnan(labels).any())
            loss.backward()
            self.optimizer.step()

                

            running_loss += loss.item() * mask.sum()
            i += mask.sum()

        return running_loss / (i+1e-5)

    def run_validation(self):
        self.model.eval()
        
        with torch.no_grad():
            pbar = tqdm.tqdm(self.val_loader)
            running_loss = 0
            running_acc = 0
            i = 0
            stats = StatsComputer()

            for timeseries, variants, labels, mask in pbar:
                pbar.set_description(f"Running loss: {running_loss/(i+1e-5) :.4}")
                
                timeseries, variants, labels = timeseries.to(self.device), variants.to(self.device), labels.to(self.device)

                outputs = self.model(timeseries, variants)
                

                loss = 0
                for o in outputs:
                    loss += self.criterion(o, labels, mask)
                
                outputs = outputs[-1]
                masked_outputs, masked_labels = self.criterion.get_masked_reshaped(outputs, labels, mask)        
                acc = torch.sum(torch.argmax(masked_labels, dim = 1) == torch.argmax(masked_outputs, dim = 1))
                
                running_loss += loss.item() * mask.sum()
                running_acc += acc 

                for n in range(len(outputs)): #for every timeseries
                    ts_outputs = outputs[n]
                    ts_labels = labels[n]
                    ts_mask = mask[n]

                    ts_masked_outputs = ts_outputs[ts_mask.type(torch.bool)]
                    ts_masked_labels = ts_labels[ts_mask.type(torch.bool)]
                    stats.append(outputs=torch.argmax(ts_masked_outputs, dim = 1).cpu().tolist(), 
                                labels=torch.argmax(ts_masked_labels, dim = 1).cpu().tolist())

                i += mask.sum()

            validation_loss = running_loss / i
            validation_accuracy = running_acc / i
            
            validation_f1 = stats.macro_avg_f1()
            print(stats)

        return validation_loss, validation_accuracy, validation_f1


    def test(self, save = True):
        print("=== Start Testing ===")
        print(f"Batch size: {self.args.batch_size}")
        
        ### Save results
        test_results = self.run_test()            
        test_results = self.label_encoder.inverse_transform(test_results)
        
        if save:
            res_df = pd.DataFrame(test_results, columns=["pred"])
            res_df.to_csv("prediction.csv", index=False)

    def run_test(self):
        # Test the model
        self.model.load_state_dict(
            torch.load(
                os.path.join(self.weights_path, "TCN_best.pth")
            ))
        self.model.eval()
        
        with torch.no_grad():
            pbar = tqdm.tqdm(self.test_loader)
            all_outputs = []
            
            for timeseries, variants, labels, mask in pbar:
                pbar.set_description(f"Testing: ")
                
                timeseries, variants, labels = timeseries.to(self.device), variants.to(self.device), labels.to(self.device)

                outputs = self.model(timeseries, variants)[0]

                mask = mask.reshape(-1)                
                outputs = outputs.reshape(-1, self.num_classes)
                outputs = outputs[mask.type(torch.bool)]
                
                all_outputs.extend(outputs.cpu().tolist())
        
        return all_outputs
           
            
     