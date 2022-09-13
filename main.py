#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  9 20:52:58 2022

@author: Team BUDDI KIRAN CHAITANYA
# BUDDI KIRAN CHAITANYA
"""

# requirements 
# torch: 1.11.0+cu113
# transformers (hugging face): 4.18.0
# numpy: 1.21.6
# pandas: 1.3.5
# matplotlib: 3.2.2
# scikit-learn: 1.0.2
# requests: 2.23.0
# tabulate: 0.8.9

import transformers
from transformers import BertModel, TFBertModel, BertTokenizer, AdamW, get_linear_schedule_with_warmup
from transformers import logging
logging.set_verbosity_error()

import numpy as np
import pandas as pd

import matplotlib
from pylab import rcParams
import matplotlib.pyplot as plt
from matplotlib import rc

import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay

import requests

import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader

from tabulate import tabulate

class stanceDataset (Dataset):
    def __init__(self, max_target_len):
        text_link = "https://alt.qcri.org/semeval2016/task6/data/uploads/semeval2016-task6-trainingdata.txt"
        data = requests.get(text_link)
        data = data.text.split('\n')

        list_of_lists=[]
        for i in range (len(data)-2):
          temp_list= data[i+1].split('\t')
          list_of_lists.append(temp_list)
        df = pd.DataFrame(list_of_lists, columns=['ID', 'target', 'tweet', 'stance']) 
        df['stance']=df.apply(lambda x: x['stance'].replace('\r',''),axis=1)

        def polarity(Stance):
          if Stance=='AGAINST':
            return 2
          elif Stance=='FAVOR':
            return 0
          else:
            return 1
  
        df['stance'] = df.stance.apply(polarity)

        df=df.sample(frac = 1)
        df.reset_index(inplace = True, drop=True)
        df.pop('ID')
        df_train, df_test = train_test_split(df, test_size=0.2)
        
        y_train = df_train['stance']
        y_test = df_test['stance']
        
        df_train.pop('stance')
        df_test.pop('stance')

        df_x_train = df_train.values.tolist()
        df_y_train = y_train.values.tolist()

        df_x_test = df_test.values.tolist()
        df_y_test = y_test.values.tolist()

        tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
        bertmodel = TFBertModel.from_pretrained("bert-base-cased", )

        # BERT TOKENIZATION
        encoded_test_targets = tokenizer  ([item[0] for item in df_x_test], padding='max_length',
                                           max_length=max_target_len,return_tensors='tf', truncation=True)
        encoded_test_tweets = tokenizer ([item[1] for item in df_x_test], return_tensors='tf', padding='max_length',
                                          max_length = 50, truncation=True)
        encoded_train_targets = tokenizer  ([item[0] for item in df_x_train], padding='max_length',
                                            max_length=max_target_len,return_tensors='tf', truncation=True)
        encoded_train_tweets = tokenizer ([item[1] for item in df_x_train], return_tensors='tf', padding='max_length',
                                          max_length = 50, truncation=True)

        # BERT EMBEDDING
        bertd_test_targets=bertmodel(encoded_test_targets)
        bertd_test_tweets=bertmodel(encoded_test_tweets)
  
        bertd_train_targets=bertmodel(encoded_train_targets)
        bertd_train_tweets=bertmodel(encoded_train_tweets)

        test_target_emb = torch.tensor(bertd_test_targets.last_hidden_state.numpy())
        test_tweet_emb = torch.tensor(bertd_test_tweets.last_hidden_state.numpy())
        test_labels = torch.as_tensor(df_y_test,  dtype=torch.long)

      
        train_target_emb = torch.tensor(bertd_train_targets.last_hidden_state.numpy())
        train_tweet_emb = torch.tensor(bertd_train_tweets.last_hidden_state.numpy())
        train_labels = torch.as_tensor(df_y_train,  dtype=torch.long)

        train_data = []
        for j in range(len(train_target_emb)):
          train_data.append([train_target_emb[j],train_tweet_emb[j]])
    
        self.x_data = train_data
        self.y_data = train_labels
        
        self.x_test_targets = test_target_emb
        self.x_test_tweets = test_tweet_emb
        self.y_test = test_labels

        self.n_samples = len(df_train)

    def __getitem__(self, index):
        return  self.x_data[index], self.y_data[index]
    def __len__(self):
        return self.n_samples


class stance(nn.Module):
    def __init__(self,emb_dim, max_target_len):
        super().__init__()
        #self.conv = nn.Conv1d(in_channels=self.target_emb.shape[1], out_channels=1, kernel_size=1, stride=1, padding=0)
        self.mh1 = nn.MultiheadAttention(embed_dim=emb_dim, num_heads=3, batch_first=True)
        self.mh2 = nn.MultiheadAttention(embed_dim=emb_dim, num_heads=3, batch_first=True)
        #self.mh3 = nn.MultiheadAttention(embed_dim=emb_dim, num_heads=8, batch_first=True)
        self.f = nn.Flatten()
        self.hl1 = nn.Linear(in_features=emb_dim*max_target_len, out_features=200)
        self.act1 = nn.ReLU()
        self.hl2 = nn.Linear(in_features=200, out_features=50)
        self.act2 = nn.ReLU()
        self.op = nn.Linear(in_features=50, out_features=3)
        #self.smax = nn.Softmax(dim=1)
        
    def forward(self, x1, x2):
        #x1 = self.conv(self.target_emb)
        x2, w = self.mh1(x1,x2,x2)
        x2, w = self.mh2(x1,x2,x2)
        #x2, w = self.mh3(x1,x2,x2)
        x2 = self.f (x2)
        x2 = self.hl1(x2)
        x2 = self.act1(x2)
        x2 = self.hl2(x2)
        x2 = self.act2(x2)
        x2 = self.op(x2)
        #x2 = self.smax(x2)
        return x2
    

    
def train_one_epoch (model, train_loader, loss_fn, optimizer, max_target_len):
    
    model.train()
    Loss = 0.0
    tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
    bertmodel = TFBertModel.from_pretrained("bert-base-cased", )
    for j, batch in enumerate(train_loader):
        data, labels = batch
        targets = data[0][0].unsqueeze(0)
        tweets = data[1][0].unsqueeze(0)
        for i in range(len(data[0])-1):
          targets = torch.cat((targets, data[0][i+1].unsqueeze(0)),0)
          tweets = torch.cat((tweets, data[1][i+1].unsqueeze(0)),0)

        optimizer.zero_grad()
        outputs = model(targets, tweets)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        Loss = Loss + loss.item()
    return Loss

def test (model, dataset):
    targets = dataset.x_test_targets
    tweets = dataset.x_test_tweets
    labels = dataset.y_test
    model.eval()
    with torch.no_grad():
        pred= model(targets, tweets)
        y_hat = np.argmax(pred, axis=1)
        test_accuracy = sum(y_hat==labels)*100/len(labels)
        print(" Test Accuracy : {:.2f} % ".format(test_accuracy))
    return test_accuracy


def get_f1_scores (model, dataset):
    targets = dataset.x_test_targets
    tweets = dataset.x_test_tweets
    labels = dataset.y_test
    model.eval()
    true_pos = np.array([0.0 for i in range (3)])
    total_correct = 0.0
    TP_plus_FP = np.array([0.0 for i in range (3)])
    f1 = np.array([0.0 for i in range (3)])
    precision = np.array([0.0 for i in range (3)])
    recall = np.array([0.0 for i in range (3)])
    with torch.no_grad():
        pred = model(targets, tweets)
        y_hat = np.argmax(pred, axis=1)
    for i in range (len(y_hat)):
        predicted = int(y_hat[i])
        TP_plus_FP[predicted] =  TP_plus_FP[predicted]+1
        if y_hat[i]== labels[i]:
            total_correct = total_correct+1
            true_pos[predicted] = true_pos[predicted]+1
    for j in range (3):
        actual_j = sum (labels==j)
        precision[j] = true_pos[j]/TP_plus_FP[j]
        recall[j] = true_pos[j]/actual_j
        f1[j] = (2*precision[j]*recall[j]) / (precision[j]+recall[j])
    total_accuracy = total_correct/len(labels)
    label = 'overall_accuracy: '+str(round(total_accuracy*100,4))
    fig,ax=plt.subplots(figsize=(12,8))
    x=[0,1,2]
    ax.plot(x,f1, label = label, color = 'green', ls = ':', 
         marker='v', markerfacecolor = 'cornflowerblue', 
         markersize = 15, fillstyle='top',  markerfacecoloralt='violet') 
    
    for i, val in enumerate (f1):
      ax.text (x[i]-0.01, f1[i]-0.02, str(round(f1[i],3)))
    plt.grid()
    plt.xlabel('Stance-Classes', fontsize = 15, labelpad = 10)
    plt.ylabel('F-1 Score', fontsize = 15)
    plt.legend(loc='upper left')
    plt.title('Class wise F-1 scores on SemEval-2016 Stance Dataset')
    plt.ylim([0.4,0.75])
    plt.xticks([0,1,2],['In Favor', 'None/No Stance', 'Against'])
    plt.show()
    return total_accuracy, f1 

def plot_progress (epoch, train_losses, test_accuracy):
    fig,ax=plt.subplots(figsize=(12,8))
    x=epoch
    tre=train_losses
    ta=[i.item() for i in test_accuracy]
    ax.plot(x,tre, color = 'salmon', ls = ':', 
         marker='v', markerfacecolor = 'tan', 
         markersize = 8, fillstyle='top',  markerfacecoloralt='forestgreen') 
    for i, val in enumerate (tre):
      ax.text (x[i]-0.01, tre[i]-0.4, str(round(tre[i],3)))
    ax2 = ax.twinx()
    ax2.plot(x,ta, color = 'cornflowerblue', ls = ':', 
         marker='o', markerfacecolor = 'olive', 
         markersize = 8, fillstyle='top',  markerfacecoloralt='pink') 
    for i, val in enumerate (ta):
      ax2.text (x[i]-0.01, ta[i]-0.6, str(round(ta[i],3))) 
    ax.grid()
    ax.set_xlabel('Epoch', fontsize = 15, labelpad = 10)
    #ax.ylabel('Testing Error', fontsize = 15)
    ax.set_ylabel('Train Error', fontsize = 15)
    ax2.set_ylabel('Test Accuracy', fontsize = 15)
    #plt.legend(loc='upper left')
    plt.title('Train Error & Testing Accuracy vs Epochs (SemEval-2016 Stance Dataset)')
    ax.set_ylim([0,20])
    ax2.set_ylim([50,70])
    plt.xticks([i+1 for i in epoch])
    plt.show()
    
def displayConfusionMatrix (model, dataset):
    targets = dataset.x_test_targets
    tweets = dataset.x_test_tweets
    y_labels = dataset.y_test
    model.eval()
    with torch.no_grad():
        pred = model(targets, tweets)
        y_hat = np.argmax(pred, axis=1)
    fig,ax=plt.subplots(figsize=(8,8))
    ConfusionMatrixDisplay.from_predictions(y_labels, y_hat, display_labels=['IN FAVOR', 'NONE', 'AGAINST'], cmap='viridis', ax=ax)
    plt.title('Confusion Matrix for Validation Set: SemEval-2016 Stance Dataset')
    plt.show()


def main():
    # loading the dataset from online resource & bert embedding, 
    # takes around 15-20 minutes on colab
    max_target_len=5
    dataset = stanceDataset(max_target_len)
    
    # model initialisation, hyperparameters
    batch_size = 128
    epochs = 15
    emb_dim=768 # embedding dimension
    train_loader = DataLoader(dataset=dataset,
                              batch_size=batch_size,
                              shuffle=True)

    model = stance(emb_dim, max_target_len)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())
    
    mydata = []
    head = ["EPOCH", "TRAIN ERROR", "TEST ACCURACY"]
    
    # start training 
    train_losses = []
    epoch=[]
    test_accuracy=[]
    for i in range(epochs):
        print("Epoch {:02d}: ".format(i+1), end=" ")
        epoch_loss = train_one_epoch (model, train_loader, loss_fn, optimizer, max_target_len)
        epoch.append(i+1)
        train_losses.append(epoch_loss)
        print(" Train Loss (CE) : {:.7f} ".format(epoch_loss), end=" ")
        ta = test(model, dataset)
        test_accuracy.append(ta)
        mydata.append(['EPOCH '+str(i+1),epoch_loss,ta.item()])
    # end training
    
    # training progress and testing evaluation results display
    print(tabulate(mydata, headers=head, tablefmt="grid"))
    
    plot_progress(epoch, train_losses, test_accuracy)
    
    # F-1 scores
    tot_acc, f1_dat = get_f1_scores (model, dataset)
    
    f1data=[]
    f1data.append([tot_acc, f1_dat[0],f1_dat[1],f1_dat[2]])
    f1head = ["OVERALL ACCURACY", "F1 (CLASS: IN FAVOR)", "F1 (CLASS: NONE)", "F1 (CLASS: AGAINST)"]
    print(tabulate(f1data, headers=f1head, tablefmt="grid"))
    
    # confusion matrix
    displayConfusionMatrix (model, dataset)

if __name__ == "__main__":
    main()

