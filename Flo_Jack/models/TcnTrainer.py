import sys 
sys.path.append("../")

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision.transforms import ToTensor, Normalize, Compose
from sklearn.metrics import f1_score
import pandas as pd

from pathlib import *
import os
import tqdm
import argparse
import json

from models.tcn import MyTCN
from dataset.VolvoDataset import VolvoDataset

class TcnTrainer:
    def __init__(self, args):
        self.args = args

        ### Get important paths
        self.curr_dir = os.path.dirname( os.path.abspath(__file__) )
        self.home_path = Path(self.curr_dir).parent.absolute()
        self.dataset_path = self.args.data_path 
        self.train_data_path = os.path.join(self.dataset_path, self.args.train_csv)
        self.test_data_path = os.path.join(self.dataset_path, self.args.test_csv)
        self.weights_path = self.args.tcnn_weights
        os.makedirs(self.weights_path, exist_ok=True)
        
        ### Get dataset and model type
        self.num_classes = 3

        self.train_dataset = VolvoDataset(data_path=self.train_data_path)
        self.label_encoder = self.train_dataset.risk_encoder
        kept_columns = self.train_dataset.get_schema()
        self.test_dataset = VolvoDataset(data_path=self.test_data_path, test=True, columns_to_keep=kept_columns)      
        
        #289 is n_features with naive preprocess
        n_features = self.train_dataset.get_n_features()
        #check if preprocess is giving some problems
        assert self.train_dataset.get_n_features() == self.test_dataset.get_n_features()
        
        self.model = MyTCN(num_input_channels=n_features, num_classes=self.num_classes)
        
        # Load weights if necessary
        if self.args.load_model != "":
            if not(self.args.load_model.endswith(".pth") or self.args.load_model.endswith(".pt")):
                raise Exception("Weights file should end with .pt or .pth")
            model_path = os.join.path(self.weights_path, self.args.load_model)
            print(f"Loading Model from {model_path}")
            self.model.load_state_dict(
                torch.load( model_path )
            )

        ### Get device
        self.device = torch.device(
                    "cuda" if (torch.cuda.is_available() and not self.args.disable_cuda) else "cpu"
                )
        self.model.to(self.device)
        print(f"Working on {self.device}")

        # Split dataset into train, validation, and test sets
        train_ratio = 0.8

        train_size = int(train_ratio * len(self.train_dataset))
        val_size = len(self.train_dataset) - train_size

        ### TODO: FAI STRATIFIED DIOBOIA DIO
        self.train_dataset, self.val_dataset = random_split(self.train_dataset, [train_size, val_size])

        # Create DataLoader instances for train, validation, and test sets
        self.train_loader = DataLoader(self.train_dataset, batch_size=self.args.batch_size, shuffle=True, collate_fn = VolvoDataset.padding_collate_fn)
        self.val_loader = DataLoader(self.val_dataset, batch_size=self.args.batch_size, collate_fn = VolvoDataset.padding_collate_fn)
        self.test_loader =  DataLoader(self.test_dataset, batch_size=self.args.batch_size, collate_fn = VolvoDataset.padding_collate_fn)
        
        # Define criterion
        self.criterion = nn.CrossEntropyLoss()

    def train(self):
        print("=== Start training ===")
        print(f"Batch size: {self.args.batch_size}")
        # Define loss function and optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, 
                                                    gamma=self.args.lr_scheduler_gamma, 
                                                    step_size=self.args.lr_scheduler_step)

        best_val_loss = float('inf')
        
        for epoch in range(self.args.num_epochs):
            ### Run epoch
            print( "="*25, f"EPOCH {epoch}", "="*25)
            epoch_loss = self.run_epoch()
            self.scheduler.step()
            print(f"Epoch [{epoch+1}/{self.args.num_epochs}], Train Loss: {epoch_loss:.4f}")

            ### Run validation
            validation_loss, validation_accuracy, validation_f1= self.run_validation()            
            print(f"Validation Loss: {validation_loss:.4f}")
            print(f"Validation Accuracy: {validation_accuracy:.4f}")
            print(f"Validation F1: {validation_f1}")
            
            
            ### Save model if best and early stopping
            if validation_loss < best_val_loss:
                print(f"Saving new model [New best loss {validation_loss:.4} vs Old best loss {best_val_loss:.4}]")
                best_val_loss = validation_loss
                waiting_epochs = 0
                torch.save(self.model.state_dict(), os.path.join(self.weights_path, "TCN_best.pth"))
            else:
                waiting_epochs += 1
                if waiting_epochs >= self.args.patience_epochs:
                    print(f"Early stopping because ran more than {self.args.patience_epochs} without improvement")
                    break

    def run_epoch(self):
        self.model.train()
        
        pbar = tqdm.tqdm(self.train_loader)
        running_loss = 0
        i = 0

        for data, labels, mask in pbar:
            pbar.set_description(f"Running loss: {running_loss/(i+1e-5) :.4}")
            
            timeseries = data
            timeseries, labels = timeseries.to(self.device), labels.to(self.device)

            outputs = self.model(timeseries)

            self.optimizer.zero_grad()
            outputs = outputs.reshape(-1, self.num_classes)
            labels = labels.reshape(-1, self.num_classes)
            mask = mask.reshape(-1)
            loss = self.criterion(outputs[mask.type(torch.bool)], labels[mask.type(torch.bool)])
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item() * mask.sum()
            i += mask.sum()

        return running_loss / (i+1e-5)

    def run_validation(self):
        with torch.no_grad():
            pbar = tqdm.tqdm(self.val_loader)
            running_loss = 0
            running_acc = 0
            i = 0
            all_labels = []
            all_outputs = []
            
            for data, labels, mask in pbar:
                pbar.set_description(f"Running loss: {running_loss/(i+1e-5) :.4}")
                
                timeseries = data
                timeseries, labels = timeseries.to(self.device), labels.to(self.device)

                outputs = self.model(timeseries)
                
                outputs = outputs.reshape(-1, self.num_classes)
                labels = labels.reshape(-1, self.num_classes)
                mask = mask.reshape(-1)
                
                outputs = outputs[mask.type(torch.bool)]
                labels = labels[mask.type(torch.bool)]
                
                loss = self.criterion(outputs, labels)
                acc = torch.sum(torch.argmax(labels, dim = 1) == torch.argmax(outputs, dim = 1))
                
                running_loss += loss.item() * mask.sum()
                running_acc += acc 
                
                all_labels.extend(torch.argmax(labels, dim = 1).cpu().tolist())
                all_outputs.extend(torch.argmax(outputs, dim = 1).cpu().tolist())
        
                i += mask.sum()

            validation_loss = running_loss / i
            validation_accuracy = running_acc / i
            validation_f1 = f1_score(all_labels, all_outputs, average="micro")
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
        self.model.eval()
        
        with torch.no_grad():
            pbar = tqdm.tqdm(self.test_loader)
            all_outputs = []
            
            for data, _ , mask in pbar:
                pbar.set_description(f"Test progress ")
                
                timeseries = data
                timeseries = timeseries.to(self.device)

                outputs = self.model(timeseries)
                
                outputs = outputs.reshape(-1, self.num_classes)
                mask = mask.reshape(-1)
                
                outputs = outputs[mask.type(torch.bool)]
                
                all_outputs.extend(outputs.cpu().tolist())
        
        return all_outputs
           
            
     