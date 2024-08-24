import torch
from sklearn.metrics import f1_score, precision_recall_fscore_support, classification_report, confusion_matrix
import numpy as np


class StatsComputer:
    def __init__(self):
        self.num_outputs = 0
        self.all_labels = []
        self.all_outputs = []

    def append(self, outputs, labels):
       self.num_outputs += 1
       self.all_outputs.append(outputs)
       self.all_labels.append(labels)

    def average_discontinuity(self):
        counts = []
        for i in range(self.num_outputs):
            this_output = torch.Tensor(self.all_outputs[i])
            this_labels = torch.Tensor(self.all_labels[i])
            count = torch.sum(
                torch.abs(
                    this_output[1:] - this_output[:-1]
                    )
                )
            
            labels_continuity = torch.unique(this_labels)
            allowed_discontinuity = len(labels_continuity) - 1

            counts.append(count-allowed_discontinuity)

        return torch.mean( torch.Tensor(counts) )

    def more_than_max(self):
        counts = []
        for i in range(self.num_outputs):
            this_output = torch.Tensor(self.all_outputs[i])
            this_labels = torch.Tensor(self.all_labels[i])

            values, val_counts = torch.unique(this_output, return_counts=True)
            this_counts = [0,0,0]
            for i in range(len(values)):
                this_counts[ int(values[i].item()) ] = max(0, val_counts[i].item() - 9)

            this_counts[0] = 0
            this_counts = torch.Tensor(this_counts)
            counts.append(this_counts)
        
        return torch.mean( torch.stack(counts), dim=0)

    def macro_avg_f1(self):
        # Get precision, recall, and f1-score for each class
        precision, recall, f1, true_sum = precision_recall_fscore_support(self.flatten(self.all_labels), self.flatten(self.all_outputs))
        print(f1)
        # Calculate macro average F1-score
        macro_avg_f1 = sum(f1) / len(f1)

        return round(macro_avg_f1, 2)
    

    def flatten(self, xss):
        return [x for xs in xss for x in xs]

    def __str__(self):
        flatten_all_labels = self.flatten(self.all_labels)
        flatten_all_outputs = self.flatten(self.all_outputs)
        string = ''
        string += str(classification_report(flatten_all_labels, flatten_all_outputs)) + '\n'
        string += f"Average discontinuity = {self.average_discontinuity():.4}" + '\n'
        #string += f"More than max = {self.more_than_max()}" + "\n"
        string += f"F1 score = {self.macro_avg_f1():.4}" + '\n'
        
        return string