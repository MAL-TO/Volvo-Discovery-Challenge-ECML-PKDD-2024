
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision.transforms import ToTensor, Normalize, Compose

from pathlib import *
import os
import tqdm
import argparse
import json

from tcn_model.tcn import MyTCN
from Flo_Jack.dataset.VolvoDataset import VolvoDataset

class TcnTrainer:
    def __init__(self, args):
        self.args = args

        ### Get important paths
        self.curr_dir = os.path.dirname( os.path.abspath(__file__) )
        self.home_path = Path(self.curr_dir).parent.absolute()
        self.dataset_path = self.args.train_data_path
        self.weights_path = self.args.tcnn_weights
        os.mkdir(self.weights_path, exists_ok=True)
        
        ### Get dataset and model type
        self.num_classes = 3

        self.dataset = VolvoDataset(data_path=self.dataset_path)
        self.model = MyTCN(num_input_channels=289, num_classes=self.num_classes)
        
        # Load weights if necessary
        if self.args.load_model != "":
            if not(self.args.load_model.endswith(".pth") or self.args.load_model.endswith(".pt")):
                raise Exception("Weights file should end with .pt or .pth")
            self.model.load_state_dict(
                torch.load( os.join.path(self.weights_path, self.args.load_model) )
            )

        ### Get device
        self.device = torch.device(
                    "cuda" if (torch.cuda.is_available() and not self.args.disable_cuda) else "cpu"
                )
        self.model.to(self.device)
        print(f"Working on {self.device}")

        # Split dataset into train, validation, and test sets
        train_ratio = 0.8

        train_size = int(train_ratio * len(self.dataset))
        val_size = len(self.dataset) - train_size

        ### TODO: FAI STRATIFIED DIOBOIA DIO
        self.train_dataset, self.val_dataset = random_split(self.dataset, [train_size, val_size])

        # Create DataLoader instances for train, validation, and test sets
        self.train_loader = DataLoader(self.train_dataset, batch_size=self.args.batch_size, shuffle=True, collate_fn = VolvoDataset.padding_collate_fn)
        self.val_loader = DataLoader(self.val_dataset, batch_size=self.args.batch_size, collate_fn = VolvoDataset.padding_collate_fn)

        # Define criterion
        self.criterion = nn.CrossEntropyLoss()

    def train(self):
        print("=== Start training ===")
        print(f"Batch size: {self.args.batch_size}")
        print(f"Hist size: {self.args.hist_size}")
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
            validation_loss = self.run_validation()            
            print(f"Validation Loss: {validation_loss:.4f}")

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
            loss = self.criterion(outputs[mask.astype(bool)], labels[mask.astype(bool)])
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item() * mask.sum()
            i += mask.sum()

        return running_loss / (i+1e-5)

    def run_validation(self):
        with torch.no_grad():
            pbar = tqdm.tqdm(self.val_loader)
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
                loss = self.criterion(outputs[mask.astype(bool)], labels[mask.astype(bool)])
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item() * mask.sum()
                i += mask.sum()

            validation_loss = running_loss / i
        return validation_loss

    def run_test(self, save=True):
        # Test the model
        self.model.eval()
        total_loss = 0.0
        num_samples = 0
        with torch.no_grad():
            for images, labels in self.test_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                total_loss += loss.item() * images.size(0)
                num_samples += len(images)
            
        test_loss = total_loss / num_samples
        print(f"Test Loss: {test_loss:.4f}") 