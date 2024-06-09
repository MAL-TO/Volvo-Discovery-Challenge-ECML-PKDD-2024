import torch
import torch.nn.functional as F

class ContinuityCrossEntropyLoss:
    def __init__(self, weights, num_classes=3, continuity=True):
        self.weights = weights
        self.crossentropy = torch.nn.CrossEntropyLoss(weight=weights)
        self.num_classes = num_classes

    ### NUM LABELS EQUAL
    def compute_num_labels_loss(self, outputs, labels, mask):
        loss = torch.zeros((len(outputs),))
        for n in range(len(outputs)): #for every timeseries
            ts_outputs = outputs[n]
            ts_labels = labels[n]
            ts_mask = mask[n]

            masked_outputs = ts_outputs[ts_mask.type(torch.bool)]
            smax_output = torch.nn.functional.softmax(masked_outputs, dim=1)
            sum_smax_output = torch.sum(smax_output, dim=0)
            
            masked_labels = ts_labels[ts_mask.type(torch.bool)]
            sum_smax_labels = torch.sum(masked_labels, dim=0)
            
            difference = torch.abs(sum_smax_output-sum_smax_labels)

            loss[n] = torch.sum(difference)

        loss = torch.mean(loss)
        return loss

    ### MAX 9 MEDIUM AND HIGH
    def compute_max_labels_loss(self, outputs, labels, mask):
        loss = torch.zeros((len(outputs),))
        for n in range(len(outputs)): #for every timeseries
            ts_outputs = outputs[n]
            ts_labels = labels[n]
            ts_mask = mask[n]

            masked_outputs = ts_outputs[ts_mask.type(torch.bool)]
            smax_output = torch.nn.functional.softmax(masked_outputs, dim=1)
            sum_smax_output = torch.sum(smax_output, dim=0)
            sum_smax_output = sum_smax_output[1:]
            
            relu_sum_smax_output = torch.nn.functional.relu(sum_smax_output - 9)
            # relu_sum_smax_output = torch.nn.functional.relu(sum_smax_output - 18)

            powered_relu_sum_smax_output = torch.pow(1 + relu_sum_smax_output, 2)
            loss[n] = torch.sum(powered_relu_sum_smax_output)

        loss = torch.mean(loss)
        return loss

    def compute_continuity_loss(self, outputs, labels, mask):
        continuity_loss = torch.zeros((len(outputs),))
        for n in range(len(outputs)): #for every timeseries
            ts_outputs = outputs[n]
            ts_labels = labels[n]
            ts_mask = mask[n]

            masked_cont_outputs = ts_outputs[ts_mask.type(torch.bool)]
            smax_output_continuity = torch.nn.functional.softmax(masked_cont_outputs, dim=1)
            if torch.isnan(smax_output_continuity).any(): print("smax_output_continuity")

            fw_output_continuity = smax_output_continuity[1:] - smax_output_continuity[:-1].detach()
            # fw_output_continuity = torch.pow(fw_output_continuity, 2)
            # fw_output_continuity = torch.sum(fw_output_continuity, dim=1) / 2 ################## !!!
            if torch.isnan(fw_output_continuity).any(): print("fw_output_continuity")

            output_continuity = fw_output_continuity

            masked_cont_labels = ts_labels[ts_mask.type(torch.bool)]
            labels_continuity = masked_cont_labels[1:] - masked_cont_labels[:-1]
            if torch.isnan(labels_continuity).any(): print("labels_continuity")


            # labels_continuity = torch.argmax(masked_cont_labels, dim=1)
            # labels_continuity = torch.unique(labels_continuity)
            # labels_continuity = len(labels_continuity) - 1
            if torch.isnan(F.mse_loss(output_continuity, labels_continuity)): 
                print("F.mse_loss(output_continuity, labels_continuity)")
                print(masked_cont_outputs)
                print(masked_cont_labels)
            continuity_loss[n] = F.mse_loss(output_continuity, labels_continuity)


        continuity_loss = torch.mean(continuity_loss)
        if torch.isnan(continuity_loss).any(): print("continuity_loss")

        return continuity_loss
    
    def compute_cross_entropy(self, outputs, labels, mask):
        masked_outputs, masked_labels = self.get_masked_reshaped(outputs, labels, mask)        
        loss = self.crossentropy(masked_outputs, masked_labels)
        return loss

    def get_masked_reshaped(self, outputs, labels, mask):
        outputs = outputs.reshape(-1, self.num_classes)
        labels = labels.reshape(-1, self.num_classes)
        mask = mask.reshape(-1)
        masked_outputs, masked_labels = outputs[mask.type(torch.bool)], labels[mask.type(torch.bool)]

        return masked_outputs, masked_labels

    def __call__(self, outputs, labels, mask):
        crossentropy_loss = self.compute_cross_entropy(outputs, labels, mask)

        continuity_loss = self.compute_continuity_loss(outputs, labels, mask)
    
        # num_labels_loss = self.compute_num_labels_loss(outputs, labels, mask)
        max_loss = self.compute_max_labels_loss(outputs, labels, mask)

        total_loss = crossentropy_loss + 0.5*continuity_loss + 0.5*max_loss
        return total_loss