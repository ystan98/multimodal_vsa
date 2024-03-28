import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor
import os
from torchvision import datasets, transforms
import numpy as np
import sys
import torch.optim.lr_scheduler as lr_scheduler
import tqdm
from torch.cuda.amp import GradScaler, autocast
from collections import defaultdict
from sklearn.metrics import classification_report
from torch.nn import functional as F



'''
Customized Datasets
'''
class Model_Dataset(Dataset):
 
    def __init__(self, 
                df,
                embedding_directory = "./embedding/",
                id_column = "material_id",
                target = "target_encoding",
                return_labels = True):
        
        self.df = df
        self.ids = self.df[id_column].values
        self.return_labels = return_labels
        self.embedding_directory = embedding_directory
        
        if self.return_labels:
            self.label = torch.from_numpy(np.vstack(df[target].values).astype(np.float32))
            
    def __len__(self):
        return len(self.df)
   
    def __getitem__(self,idx):
        
        self.embeddings = torch.load(self.embedding_directory + f"{self.ids[idx]}.pt")
            
        if self.return_labels:
            return self.embeddings, self.label[idx]
        
        else:
            
            return self.embeddings
        
        
        
'''
Loss Functions
'''

class FocalLoss(nn.Module):
    '''
    FocalLoss - nn.Module 
    
    Focal loss applies a modulating term to the cross entropy loss in order to focus learning on hard misclassified examples.
    It is a dynamically scaled cross entropy loss, where the scaling factor decays to zero as confidence in the correct class increases.
    Link - https://paperswithcode.com/method/focal-loss#:~:text=Focal%20loss%20applies%20a%20modulating,in%20the%20correct%20class%20increases.
    '''
    
    def __init__(self, alpha=1, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        bce_loss = F.binary_cross_entropy(inputs,  targets)
        loss = self.alpha * (1 - torch.exp(-bce_loss)) ** self.gamma * bce_loss
        return loss
    
    
    
    
    
    
    
    
''' 
Classifiers
''' 
    

class MLP(nn.Module):
    def __init__(self, dimensions, hidden_dim, number_of_classes, multilabel_bool):
        super().__init__()
        self.dimensions = dimensions
        self.hidden_dim = hidden_dim
        self.number_of_classes = number_of_classes
        self.multilabel_bool = multilabel_bool
        
      #  MLP classification layer
        self.classifier = nn.Sequential(
            nn.Linear(self.dimensions, hidden_dim),
            nn.Dropout(0.30, inplace = True),
            nn.LeakyReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(0.30, inplace = True),
            nn.LeakyReLU(inplace=True),
            nn.Linear(hidden_dim, number_of_classes)
        )


            
    def forward(self, x):
        if self.multilabel_bool:
            x = x.squeeze(1)
            out = torch.sigmoid(self.classifier(x))
        else:
            if self.number_of_classes==2:
                x = x.squeeze(1)
                out = torch.sigmoid(self.classifier(x))   
            else:
                x = x.squeeze(1)
                out = F.softmax(self.classifier(x), dim = 1)        
        return out
    
def train(dataloader, model, criterion, optimizer, device, scaler):
    training_loss = []
    model.train()
    
    for embedding, label in dataloader:
        
        optimizer.zero_grad()
     
        
        embedding = embedding.float().to(device)
        target = label.to(device)


        # forward pass
        with autocast(enabled=False):
            predicted = model(embedding)
            #target = F.one_hot(target, num_classes=2).float()
            loss = criterion(predicted,  target)

        # backward pass
        scaler.scale(loss).backward()

        # update weights with lookahead optimizer
        scaler.step(optimizer)
        scaler.update()
 
        #removed=========
        #loss.backward()
        
        
#         optimizer.step() #removed 20 april
            
        training_loss.append(loss.detach().cpu().numpy())
            
    mean_training_loss = np.mean(training_loss)
    return mean_training_loss


def evaluate(dataloader, model, criterion, device):
    eval_loss = []
    model.eval()
    
    with torch.no_grad():
        for embedding, label in dataloader:
            embedding = embedding.float().to(device)
            target = label.to(device)
            predicted = model(embedding)
            loss = criterion(predicted, target)
            eval_loss.append(loss.detach().cpu().numpy())

    mean_training_loss = np.mean(eval_loss)
    return mean_training_loss



def train_and_evaluate(model, train_generator, val_generator, model_args, device):  
    print("starting")
    lowest_loss = sys.maxsize
    lowest_loss_training = sys.maxsize
    
    early_stopping_threshold = model_args['early_stopping_rounds']
    num_epochs = model_args['max_epochs']
    learning_rate = model_args['learning_rate']
    optimizer = model_args['optimizer']
    scaler = model_args['scaler']
    criterion = model_args['criterion'] #Insert Loss Function
    
    training_losses = []
    eval_losses = []
    epoch_elapsed = 0
    
    # Learning rate become 0.0001
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)
    
    checkpoint_name = f"checkpoint_image_470"
    progress_bar = tqdm.tqdm(range(num_epochs))  
    
    for epoch in progress_bar:
        epoch_elapsed += 1
        training_loss = 0
                                             
        average_training_loss = train(train_generator, model, criterion, optimizer, device,scaler)
        average_eval_loss = evaluate(val_generator, model, criterion, device)
        
        scheduler.step(average_eval_loss)

        training_losses.append(average_training_loss)
        eval_losses.append(average_eval_loss)

        progress_bar.set_description(f"Epoch: {epoch+1} Training Loss:{average_training_loss:.6f} Eval Loss:{average_eval_loss:.6f}")

        if average_eval_loss < lowest_loss:
            epochs_no_improve = 0
            lowest_loss = average_eval_loss
            lowest_loss_training = average_training_loss
            print(f"Found a better loss!\nEpoch: {epoch+1} Training Loss:{average_training_loss:.6f} Eval Loss:{average_eval_loss:.6f}")
            best_model = model
            checkpoint_save = { 
                'epoch': epoch,
                'model': model.state_dict(),
                #optimizer to apex-optimizer
                'optimizer': optimizer.state_dict(),
                'lr_sched': scheduler
            }
            torch.save(checkpoint_save, f'{checkpoint_name}.pth')
        else:
            progress_bar.set_description(f"Epoch: {epoch+1} - no improvement\nEpoch: {epoch+1} Training Loss:{average_training_loss:.6f} Eval Loss:{average_eval_loss:.6f}")
            epochs_no_improve += 1
            
            if epochs_no_improve >= early_stopping_threshold:
                print(f"No improvement in over {early_stopping_threshold} epochs, early break")
                break

    return best_model, criterion, training_losses, eval_losses


def inference(dataloader, model, criterion, device, validation = True):

    model.eval()
    prediction_list = []
    truth_list = []
    if validation:
        with torch.no_grad():
            for embedding, label in dataloader:
                embedding = embedding.float().to(device)
                predicted = model(embedding)
                prediction_list.append(predicted.cpu().numpy())
                truth_list.append(label)
                
        predictions = np.concatenate(prediction_list, axis=0)
        truths = np.concatenate(truth_list, axis=0)
        return predictions, truths
        
    else:
        
        with torch.no_grad():
            for embedding in dataloader:
                embedding = embedding.float().to(device)
                predicted = model(embedding)
                prediction_list.append(predicted.cpu().numpy())
        predictions = np.concatenate(prediction_list, axis=0)
    
    return predictions


def evaluate_validation(predictions, truths, one_hot_class_dict, multilabel_bool, num_of_class=0, label_to_focus = "macro avg"):
    threshold_chosen = -1
    threshold_scores_dict = defaultdict(list)

    
    if multilabel_bool:
        
        threshold = list(np.arange(0.0, 1, 0.05))
        for thres in threshold:
            
            predictions_thresh = torch.tensor(np.where(predictions >= thres, 1, 0))
            score_dict = classification_report(truths, predictions_thresh, target_names= one_hot_class_dict.keys(),  zero_division = 1,  output_dict = True)[label_to_focus]

            threshold_scores_dict["precision"].append(score_dict["precision"])
            threshold_scores_dict["recall"].append(score_dict["recall"])
            threshold_scores_dict["f1"].append(score_dict["f1-score"])
            
        # pick best threshold based on f1
        l_np = np.asarray(threshold_scores_dict["f1"]) # f1
        f1_argmax = l_np.argmax()
        threshold_chosen  = threshold[f1_argmax]
        
        prediction_thresh = torch.tensor(np.where(predictions >= threshold_chosen, 1, 0))
        class_report_dict = classification_report(truths, prediction_thresh, target_names= one_hot_class_dict.keys(), output_dict = True)
        print(classification_report(truths, prediction_thresh, target_names= one_hot_class_dict.keys()))
        
    
        return prediction_thresh.numpy(), class_report_dict, threshold_chosen
    
    else:
        if num_of_class ==2:
            #binary
            indices = np.where(truths == 1)
            indices = indices[1]
            pred = predictions[:, 1]
            
            threshold = list(np.arange(0.0, 1, 0.05))
            threshold_scores_dict = defaultdict(list)
            for thres in threshold:
                # 1 means positive
                # 0 means negative
                predictions_thresh = torch.tensor(np.where(pred >= thres, 1, 0))
                score_dict = classification_report(indices, predictions_thresh, target_names= one_hot_class_dict.keys(),  
                                                   zero_division = 1,  output_dict = True)['macro avg']
                threshold_scores_dict["precision"].append(score_dict["precision"])
                threshold_scores_dict["recall"].append(score_dict["recall"])
                threshold_scores_dict["f1"].append(score_dict["f1-score"])

            l_np = np.asarray(threshold_scores_dict["f1"]) # f1
            f1_argmax = l_np.argmax()
            threshold_chosen  = threshold[f1_argmax]

            prediction_thresh = torch.tensor(np.where(pred >= threshold_chosen, 1, 0))
            class_report_dict = classification_report(indices, prediction_thresh, target_names= one_hot_class_dict.keys(), 
                                                      output_dict = True)
            print(classification_report(indices, prediction_thresh, target_names= one_hot_class_dict.keys()))     
            return prediction_thresh.numpy(), class_report_dict, threshold_chosen
            
            
        else:
            prediction_array_argmax = torch.argmax(torch.tensor(predictions), dim = 1)
            prediction_array_res = torch.zeros_like(torch.tensor(predictions)).scatter_(1, prediction_array_argmax.unsqueeze(1), 1.)
            class_report_dict = classification_report(truths, prediction_array_res, target_names= one_hot_class_dict.keys(), 
                                                      output_dict = True)
            print(classification_report(truths, prediction_array_res, target_names= one_hot_class_dict.keys()))
        
        return prediction_array_res.numpy(), class_report_dict, threshold_chosen

    
def prepare_validation_output(df_val, val_prob, classification_report_dict, threshold_chosen, unique_labels):
    class_dict = {k:v for k,v in classification_report_dict.items() if k in unique_labels}
    population_dict = {k:v for k,v in classification_report_dict.items() if k not in unique_labels}
    results = pd.DataFrame(val_prob, columns = unique_labels)
    
    df_val_final = df_val[["material_id","label","target_encoding"]].join(results)
    
    validation_result_dict = {"class_labels": class_dict,
                          "overall": population_dict,
                          "threshold":threshold_chosen}
    
    return df_val_final, validation_result_dict
