import sys 
sys.path.append("../")

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

from torch.nn import BCELoss, L1Loss

from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

from pathlib import *
import os
import tqdm
import argparse
import json

from models.tcn import SS_TCN
from dataset.VolvoDataset import VolvoDataset, VolvoDatasetPart1, VolvoDatasetPart2
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

        self.train_dataset = VolvoDatasetPart1(data_path=self.train_data_path, variants_path=self.variants_path)
        self.processor = self.train_dataset.get_processor()
        self.label_encoder = self.processor.risk_encoder

        self.train_dataset, self.validation_dataset = self.train_dataset.split_train_validation()
    
        self.test_dataset = VolvoDataset(data_path=self.test_data_path, variants_path=self.variants_path, test=True)
        self.test_dataset.set_processor(self.processor) 
     
        
        self.num_classes = self.train_dataset.get_n_classes()
        
        #check if preprocess is giving some problems
        assert self.train_dataset.get_n_features() == self.test_dataset.get_n_features()
        
        self.model = SS_TCN(num_input_channels=self.train_dataset.get_n_features(), 
                            num_classes=self.num_classes, 
                            is_phase_1=True
                            )

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
                                       #collate_fn = VolvoDataset.padding_collate_fn,
                                       shuffle=True,
                                       num_workers=12) #pin_memory=True #consigliano
        self.val_loader = DataLoader(self.validation_dataset, 
                                     batch_size=self.args.batch_size, 
                                     #collate_fn = VolvoDataset.padding_collate_fn, 
                                     shuffle=True,
                                     num_workers=12)
        self.test_loader =  DataLoader(self.test_dataset, 
                                       batch_size=self.args.batch_size,
                                       shuffle=False)
                                       #collate_fn = VolvoDataset.padding_collate_fn)
        
        # Define criterion
        print('Computing class weights...', end='')
        # y = self.label_encoder.transform( self.train_dataset.volvo_df[["risk_level"]].values )
        # y = np.argmax(y, axis=1).flatten()
        # y = np.array(y)[0]

        # weights = compute_class_weight(class_weight="balanced", classes=np.unique(y), y=y)
        # weights = weights 
        weights = self.train_dataset.get_weights()
        self.criterion = BCELoss(weight=torch.Tensor(weights).to(self.device))
        print('done')
        print('Class weights = ', weights)
        # self.criterion = ContinuityCrossEntropyLoss(weights=torch.Tensor([1,1,1]).to(self.device))


    def reset_optimizer(self, lr):
        parameters = self.model.parameters() if self.model is not None else self.model_2.parameters()
        self.optimizer = optim.NAdam(parameters, lr=lr)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, 
                                                    gamma=self.args.lr_scheduler_gamma if self.model is not None else self.args.lr_scheduler_gamma_2, 
                                                    step_size=self.args.lr_scheduler_step if self.model is not None else self.args.lr_scheduler_step_2)

    def train(self):
        if not self.args.skip_phase_1:
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
        
        self.init_phase_2()
        self.train_phase_2()

    def init_phase_2(self):
        n_input_channels = self.model.num_input_channels
        del self.model
        self.model = None

        # clean cuda memory
        torch.cuda.empty_cache()

        self.train_dataset = VolvoDatasetPart2(data_path=self.train_data_path, variants_path=self.variants_path)
        self.processor = self.train_dataset.get_processor()
        self.label_encoder = self.processor.risk_encoder

        self.train_dataset, self.validation_dataset = self.train_dataset.split_train_validation()

        self.model_2 = SS_TCN(num_input_channels=n_input_channels, 
                            num_classes=1, 
                            is_phase_1=False
                            )
        
        # load the best weights from phase 1 except for the last layer
        # state_dict = torch.load(os.path.join(self.weights_path, "TCN_best.pth"))
        # weights_to_keep = [x for x in state_dict.keys() if "mlp" not in x]

        # new_state_dict = {}
        # for key in weights_to_keep:
        #     new_state_dict[key] = state_dict[key]

        # new_state_dict.keys()
        # self.model_2.load_state_dict(new_state_dict, strict=False)

        
        self.model_2.to(self.device)
        
        self.train_loader = DataLoader(self.train_dataset, 
                                       batch_size=32, 
                                       #collate_fn = VolvoDataset.padding_collate_fn,
                                       shuffle=True,
                                       num_workers=12, 
                                       drop_last=True) #pin_memory=True #consigliano
        self.val_loader = DataLoader(self.validation_dataset, 
                                     batch_size=self.args.batch_size, 
                                     #collate_fn = VolvoDataset.padding_collate_fn, 
                                     shuffle=True,
                                     num_workers=12)
        
        self.criterion = nn.L1Loss()

        
        

    def train_phase_2(self):
        print("=== Start Second phase training ===")
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
            epoch_loss = self.run_epoch_phase_2()
            self.scheduler.step()
            print(f"Epoch [{epoch+1}/{self.args.num_epochs}], Train Loss: {epoch_loss:.4f}")

            ### Run validation
            validation_loss, validation_accuracy = self.run_validation_phase_2()            
            # print(f"Validation Accuracy: {validation_accuracy:.4f}")
            print(f"Validation Loss: {validation_loss:.4f} vs Best {best_val_loss:.4f}")
            print(f"Validation F1: {validation_accuracy} vs Best {best_val_f1}")
            
            
            ### Save model if best and early stopping
            # if validation_loss < best_val_loss:
            #     print(f"Saving new model [New best loss {validation_loss} vs Old best loss {best_val_loss}]")
            if validation_accuracy > best_val_f1 or (validation_accuracy == best_val_f1 and validation_loss < best_val_loss):
                print(f"Saving new model \n", 
                      f"[New best loss {validation_loss} vs Old best loss {best_val_loss}] \n ",
                      f"[New best f1   {validation_accuracy} vs Old best f1   {best_val_f1}]")
                best_val_loss = validation_loss
                best_val_f1 = validation_accuracy
                waiting_epochs = 0
                torch.save(self.model_2.state_dict(), os.path.join(self.weights_path, "TCN_phase2_best.pth"))
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

        for timeseries, variants, labels in pbar:
            pbar.set_description(f"Running loss: {running_loss/(i+1e-5) :.4}")
            
            timeseries, variants, labels = timeseries.to(self.device), variants.to(self.device), labels.to(self.device)

            outputs = self.model(timeseries, variants)

            self.optimizer.zero_grad()
            
            loss = self.criterion(outputs, labels)

            # loss = 0
            # for o in outputs:
            #     loss += self.criterion(o, labels, mask)
            #     if torch.isnan(loss).any():
            #         print(torch.isnan(o).any())
            #         print(torch.isnan(labels).any())
            loss.backward()
            self.optimizer.step()

                

            running_loss += loss.item()
            i += 1

        return running_loss / (i+1e-5)

    def run_validation(self):
        self.model.eval()
        
        with torch.no_grad():
            pbar = tqdm.tqdm(self.val_loader)
            running_loss = 0
            running_acc = 0
            i = 0
            stats = StatsComputer()

            for timeseries, variants, labels in pbar:
                pbar.set_description(f"Running loss: {running_loss/(i+1e-5) :.4}")
                
                timeseries, variants, labels = timeseries.to(self.device), variants.to(self.device), labels.to(self.device)


                outputs = self.model(timeseries, variants)
                

                # loss = 0
                # for o in outputs:
                #     loss += self.criterion(o, labels, mask)

                loss = self.criterion(outputs, labels)
                
                # outputs = outputs[-1]
                acc = torch.sum(torch.argmax(labels, dim = 1) == torch.argmax(outputs, dim = 1))
                
                running_acc += acc 

                # for n in range(len(outputs)): #for every timeseries
                #     ts_outputs = outputs[n]
                #     ts_labels = labels[n]
                #     print(ts_outputs)
                #     print(ts_labels)


                stats.append(outputs=torch.argmax(outputs, dim=-1).cpu().tolist(), 
                            labels=torch.argmax(labels, dim=-1).cpu().tolist())

                i += 1

            validation_loss = running_loss / i
            validation_accuracy = running_acc / i
            
            validation_f1 = stats.macro_avg_f1()
            print(stats)

        return validation_loss, validation_accuracy, validation_f1
    
    def run_epoch_phase_2(self):
        self.model_2.train()
        
        pbar = tqdm.tqdm(self.train_loader)
        running_loss = 0
        i = 0

        for timeseries, variants, labels in pbar:
            pbar.set_description(f"Running loss: {running_loss/(i+1e-5) :.4}")
            
            timeseries, variants, labels = timeseries.to(self.device), variants.to(self.device), labels.to(self.device)

            outputs = self.model_2(timeseries, variants)
            
            outputs = outputs.squeeze().float()
            labels = labels.float()

            self.optimizer.zero_grad()
            loss = self.criterion(outputs, labels)

            # loss = 0
            # for o in outputs:
            #     loss += self.criterion(o, labels, mask)
            #     if torch.isnan(loss).any():
            #         print(torch.isnan(o).any())
            #         print(torch.isnan(labels).any())
            loss.backward()
            self.optimizer.step()

                

            running_loss += loss.item()
            i += 1

        return running_loss / (i+1e-5)

    def run_validation_phase_2(self):
        self.model_2.eval()
        
        with torch.no_grad():
            pbar = tqdm.tqdm(self.val_loader)
            running_loss = 0
            running_acc = 0
            i = 0
            stats = StatsComputer()

            for timeseries, variants, labels in pbar:
                pbar.set_description(f"Running loss: {running_loss/(i+1e-5) :.4}")
                
                timeseries, variants, labels = timeseries.to(self.device), variants.to(self.device), labels.to(self.device)

                outputs = self.model_2(timeseries, variants)
                outputs = outputs.squeeze().float()
                

                # loss = 0
                # for o in outputs:
                #     loss += self.criterion(o, labels, mask)

                loss = self.criterion(outputs, labels)
                
                
                outputs = tensor_to_vector(outputs)
                labels = tensor_to_vector(labels)

                correct = torch.eq(outputs, labels).sum().item()
                total = outputs.numel()
                acc = correct / total
                
                running_acc += acc 

                # for n in range(len(outputs)): #for every timeseries
                #     ts_outputs = outputs[n]
                #     ts_labels = labels[n]
                #     print(ts_outputs)
                #     print(ts_labels)


                stats.append(outputs=torch.argmax(outputs, dim=-1).cpu().tolist(), 
                            labels=torch.argmax(labels, dim=-1).cpu().tolist())

                i += 1

            validation_loss = running_loss / i
            validation_accuracy = running_acc / i
            
            #validation_f1 = stats.macro_avg_f1()
            #print(stats)
            print(f"Validation Accuracy: {validation_accuracy:.4f}")

        return validation_loss, validation_accuracy


    def test(self, save = True):
        print("=== Start Testing ===")
        print(f"Batch size: {self.args.batch_size}")
        
        ### Save results
        test_results = self.run_test()            
        #test_results = self.label_encoder.inverse_transform(test_results)
        test_results = self.one_hot_to_result(test_results)

        if save:
            res_df = pd.DataFrame(test_results, columns=["pred"])
            res_df.to_csv("prediction.csv", index=False)

    def one_hot_to_result(self, one_hot):
        # for each element, if it is 0, add to a list "low", i it is 1, add to a list "medium", if it is 2, add to a list "high"
        result = []

        for i in range(len(one_hot)):
            if int(one_hot[i]) == 0:
                result.append("Low")
            elif int(one_hot[i]) == 1:
                result.append("Medium")
            elif int(one_hot[i]) == 2:
                result.append("High")
            else:
                raise Exception(f"Invalid value in one_hot: {one_hot[i]}")
        
        return result


    def run_test(self):
        
        del self.model_2
        self.model_2 = None

        # clean cuda memory
        torch.cuda.empty_cache()

        self.model = SS_TCN(num_input_channels=self.train_dataset.get_n_features(),
                            num_classes=2, 
                            is_phase_1=True
                            )
        # Test the model
        self.model.load_state_dict(
            torch.load(
                os.path.join(self.weights_path, "TCN_best.pth")
            ))
        self.model.eval()
        self.model.to(self.device)
        
        with torch.no_grad():
            pbar = tqdm.tqdm(self.test_loader)
            all_outputs = []
            
            for timeseries, variants, labels in pbar:
                pbar.set_description(f"Testing: ")
                
                timeseries, variants, labels = timeseries.to(self.device), variants.to(self.device), labels.to(self.device)

                outputs = self.model(timeseries, variants)
                outputs = torch.argmax(outputs, dim=-1)
                all_outputs.extend(outputs.cpu()[:timeseries.size(0)].tolist())

        # phase 2
        del self.model
        self.model = None

        # clean cuda memory
        torch.cuda.empty_cache()

        self.model_2 = SS_TCN(num_input_channels=self.train_dataset.get_n_features(), 
                            num_classes=1, 
                            is_phase_1=False
                            )
        self.model_2.load_state_dict(torch.load(os.path.join(self.weights_path, "TCN_phase2_best.pth")))
        self.model_2.eval()
        self.model_2.to(self.device)

        with torch.no_grad():
            pbar = tqdm.tqdm(self.test_loader)
            final_result = []
            
            for i, value in enumerate(pbar):
                pbar.set_description(f"Testing: ")
                
                timeseries, variants, labels = value
                timeseries, variants, labels = timeseries.to(self.device), variants.to(self.device), labels.to(self.device)

                for j in range(timeseries.size(0)):
                    index = i*self.args.batch_size + j
                    if all_outputs[index] == 0:
                        # all outputs is "low"
                        outputs = torch.zeros(10)
                        final_result.extend(outputs.cpu().tolist())
                    else:
                        timeserie = timeseries[j].unsqueeze(0)
                        variant = variants[j].unsqueeze(0)
                        outputs = self.model_2(timeserie, variant)
                        outputs = tensor_to_vector(outputs)
                        outputs = outputs + 1
                        for output in outputs.cpu().tolist():
                            final_result.extend(output)

        print(len(final_result))
        print(final_result[:100])
        return final_result
           
            
def tensor_to_vector(tensor):
        tensor = tensor * 10
        tensor = torch.round(tensor)

        # for each number, if it is <1, set it to 1
        tensor = torch.where(tensor < 1, torch.ones_like(tensor), tensor)

        # for each number, create a tensor with 10 elements, with (1 - number) being 0 and number being 1
        list = []
        for n in tensor:
            list.append(torch.cat([torch.zeros(int(10 - n)), torch.ones(int(n))]))

        return torch.stack(list)