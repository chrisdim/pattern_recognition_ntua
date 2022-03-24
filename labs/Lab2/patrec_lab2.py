"""
ECE NTUA: Flow S - Signals, Control and Robotics
Pattern Recognition - 9th Semester
Lab 2: Speech Recognition with HMMs and RNNs

Author: Christos Dimopoulos - 03117037
"""

import numpy as np
import matplotlib.pyplot as plt
import librosa
import re
import os
import pandas as pd
from pomegranate import *
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

from parser import *
from lstm import *
from plot_confusion_matrix import *

import warnings
warnings.filterwarnings("ignore")

from torch.utils.data import Dataset, DataLoader
import torch
from torch import nn
from torch import optim

# Step 9: Read FSDD audiofiles and preprocess the data
X_train, X_test, y_train, y_test, spk_train, spk_test = parser('recordings', n_mfcc=6)

# Stratified Split of Train set to Train-Validation (80/20)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, stratify=y_train)

# Normalize Data
scale_fn = make_scale_fn(X_train)
X_train = scale_fn(X_train)
X_val = scale_fn(X_val)
X_test = scale_fn(X_test)

# Step 10: Define function that trains HMM models using the Forward-Backward Algorithm
def forward_backward(X, n_states, n_mixtures, max_iter, gmm= True, threshold = 1e-9):
    dists = []
    for i in range(n_states):
        if gmm and n_mixtures>1:
            a = GeneralMixtureModel.from_samples(MultivariateGaussianDistribution, n_mixtures, np.float_(np.vstack(X)))
        else:
            a = MultivariateGaussianDistribution.from_samples(np.float_(np.vstack(X)))
        dists.append(a)

    # Define Matrix of initial probabilities
    starts = np.zeros(n_states)
    starts[0] = 1

    # Define Matrix of Final probabilities
    ends = np.zeros(n_states)
    ends[-1] = 1

    # Define Transition Matrix
    trans_mat = np.zeros((n_states,n_states))
    for i in range(n_states):
        for j in range(n_states):
            if i == j or j == i+1:
                trans_mat[i,j] = 0.5  

    # Define the GMM-HMM
    model = HiddenMarkovModel.from_matrix(trans_mat, dists, starts, ends, state_names=['s{}'.format(i) for i in range(n_states)])

    # Fit the model till convergence or until max iterations 
    model.fit(X, max_iterations=max_iter, algorithm = 'baum-welch', stop_threshold=threshold)

    return model

# Step 11: Train one GMM-HMM for each of the 10 digits

# Sort samples per digit
X_train_per_digit = []
for digit in range(10):
    indices = []
    # find samples of certain digit label
    for i in range(len(X_train)):
        if y_train[i] == digit:
            indices.append(i)
    digit_samples = np.take(X_train, indices, axis = 0) # collect samples

    # X_train_per_digit is a list of length 10 that contains numpy arrays of different size
    X_train_per_digit.append(digit_samples) # num_sequences x seq_length x feature_dimension

# Function that thrains one GMM-HMM for each digit
def fit_HMM(X, n_states, n_mixtures, max_iter, gmm= True, threshold = 1e-9):
    models = []
    for i in range(10):
        models.append(forward_backward(X[i], n_states, n_mixtures, max_iter, gmm, threshold))
    return models

# Step 12: Testing of Digit Recognition

# Define Functions predict and score that use GMM-HMMS models to find predictions with the Viterbi Algrithm
def predict_HMM(models, X, y, classes = 10):
    
    y_pred = []
    
    for sample in range(len(X)):
        log_likelihoods = []
        for model in range(classes):
            # Calculate log-likelihood  with Viterbi
            logp,_ = models[model].viterbi(X[sample])
            log_likelihoods.append(logp)
        y_pred.append(np.argmax(np.array(log_likelihoods))) # Maximum log-Likelihood
    return y_pred

# Returns Accuracy score and Confusion matrix
def score_HMM(models, X, y, classes = 10):
    y_pred = predict_HMM(models, X, y, classes)
    acc = accuracy_score(y, y_pred)
    cm = confusion_matrix(y, y_pred)
    return acc, cm

#################### HYPERPARAMETER TUNING ####################

# Hyperparameter Values
n_states = [i for i in range(1,5)]
n_mixtures = [i for i in range(1,6)]
max_iter = [5,10,15,20,30]


# Train on train set and evaluate on validation set to find best hyperparameters
accs = []
hyperparameters = []

for nstates in n_states:
    for nmixtures in n_mixtures:
        for maxiter in max_iter:
            clear_output(wait=True)
            models = fit_HMM(X_train_per_digit, nstates, nmixtures, maxiter, True, 1e-9) # train HMMs
            acc, cm = score_HMM(models, X_val, y_val) # score on validation set
            
            print('Number of States:', nstates,'- Number of Gaussian Mixtures: ', nmixtures, '- Max Iterations:',maxiter,': ACCURACY:',acc)
            hyperparameters.append({'n_states':nstates, 'n_mixtures': nmixtures, 'max_iters':maxiter})
            accs.append(acc)

###################################################################
best_acc = np.max(accs)
best_hyperparam = hyperparameters[np.argmax(accs)]
print('Best Accuracy Score on Validation Set:', best_acc)
print('Best Hyperparameters:', best_hyperparam)

# Define the GMM-HMMs with the best Hyperparameters and evaluate on test set
best_models = fit_HMM(X_train_per_digit, best_hyperparam['n_states'], best_hyperparam['n_mixtures'], best_hyperparam['max_iters'], True, 1e-9) # train HMMs
val_acc, val_cm = score_HMM(best_models, X_val, y_val)
test_acc, test_cm = score_HMM(best_models, X_test, y_test)
print('Best GMM-HMMs Performance:')
print('Validation Set Accuracy:',val_acc)
print('Test Set Accuracy:',test_acc)

# Step 13: Confusion Matrices
plot_confusion_matrix(val_cm, classes = [i for i in range(10)],
                          normalize=False,
                          title='Confusion matrix - Validation Set',
                          cmap=plt.cm.Blues)
plot_confusion_matrix(test_cm, classes = [i for i in range(10)],
                          normalize=False,
                          title='Confusion matrix - Test Set',
                          cmap=plt.cm.Blues)


##############################################################################################
# Step 14: Recurrent Neural Networks - LSTMs

# Define Datasets
train_set = FrameLevelDataset(X_train, y_train)
val_set = FrameLevelDataset(X_val, y_val)
test_set = FrameLevelDataset(X_test, y_test)

BATCH_SIZE = 64

# Define Dataloaders
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=True)

# Function that Trains a Neural Network
def train_NN(dataloader, model, criterion, optimizer, packed = False):
    total_loss = 0.0
    model.train() # switch to train mode
    
    for idx, batch in enumerate(dataloader, 1):
        
        # Unpack Batch
        (inputs, labels, lengths) = batch
        
        # Zero Gradients
        optimizer.zero_grad()
        
        if packed: # allign labels with sorted input
            # Forward Pass
            y_preds, indices = model(inputs, lengths)

            # Compute Loss
            loss = criterion(y_preds, labels[indices])
        else:
            # Forward Pass
            y_preds = model(inputs, lengths)

            # Compute Loss
            loss = criterion(y_preds, labels)

        # Back-Propagate Loss
        loss.backward()

        # Update Loss
        optimizer.step()

        # Collect Loss
        total_loss += loss.data.item()

    return total_loss/idx

# Function that Evaluates a Neural Network
def eval_NN(dataloader, model, criterion, packed = False):
    
    total_loss = 0.0
    model.eval() # switch to evaluation mode
    
    y_gold = []
    y_pred = []
    
    with torch.no_grad(): # don't keep gradients
        for idx, batch in enumerate(dataloader, 1):
            
            (inputs, labels, lengths) = batch
            
            
            if packed: # allign labels with sorted input
                # Forward Pass
                y_preds, indices = model(inputs, lengths)

                # Compute Loss
                loss = criterion(y_preds, labels[indices])
            else:
                # Forward Pass
                y_preds = model(inputs, lengths)

                # Compute Loss
                loss = criterion(y_preds, labels)
            
            # Prediction: argmax of aposterioris
            prediction = torch.argmax(y_preds, dim=1)
            
            # Collect Loss and labels
            total_loss += loss.data.item()
            
            y_pred.append(prediction.numpy())
            if packed:
                y_gold.append(labels[indices].numpy())
            else:
                y_gold.append(labels.numpy())

    return total_loss / idx, (y_gold, y_pred)

# Define a simple LSTM Network at first
n_features = X_train[0].shape[1]
LSTM_model = BasicLSTM(input_dim = n_features, hidden_size = 64, output_dim = 10, num_layers = 1, bidirectional=False, dropout = 0)

# Define Hyperparameters
EPOCHS = 100
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(LSTM_model.parameters(), lr=0.001)

# Begin Training of LSTM and evaluation on validation set
train_losses = []
val_losses = []

for epoch in range(EPOCHS):
    # fit model
    train_loss = train_NN(train_loader, LSTM_model, criterion, optimizer)
    
    print('EPOCH',str(epoch),':')
    print('Training Loss:',str(train_loss))    
    train_losses.append(train_loss)
    
    # evaluate on validation set - comment out if unwanted
    val_loss, (y_gold, y_pred) = eval_NN(val_loader, LSTM_model, criterion)
    print('Validation Loss:',str(val_loss))
    val_losses.append(val_loss)

# Function that plots train and validation losses for given epochs
def plot_losses(epochs, train_loss, val_loss, title = 'Losses per Epoch'):
    n = np.arange(epochs)
    plt.plot(n, train_loss, label = "Train Loss")
    plt.plot(n, val_loss, label = "Validation Loss")
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title(title)
    plt.show()

plot_losses(EPOCHS, train_losses, val_losses, title = 'Losses per Epoch - LSTM Model')

# Add Dropout and L2-Regularization to our LSTM Model
LSTM_model2 = BasicLSTM(input_dim = n_features, hidden_size = 64, 
                       output_dim = 10, num_layers = 1, bidirectional=False, dropout = 0.5) # 50% dropout prob

# L2-Regularization Optimizer 
optimizer = torch.optim.Adam(LSTM_model2.parameters(), lr=0.001, weight_decay=1e-5)

# Begin Training of LSTM and evaluation on validation set
train_losses2 = []
val_losses2 = []

for epoch in range(EPOCHS):
    # fit model
    train_loss = train_NN(train_loader, LSTM_model2, criterion, optimizer)
    
    print('EPOCH',str(epoch),':')
    print('Training Loss:',str(train_loss))    
    train_losses2.append(train_loss)
    
    # evaluate on validation set - comment out if unwanted
    val_loss, (y_gold, y_pred) = eval_NN(val_loader, LSTM_model2, criterion)
    print('Validation Loss:',str(val_loss))
    val_losses2.append(val_loss)

plot_losses(EPOCHS, train_losses2, val_losses2, title = 'Losses per Epoch - L2 Regularization & Dropout')

# Train Model with Early Stopping
LSTM_model3 = BasicLSTM(input_dim = n_features, hidden_size = 64, 
                       output_dim = 10, num_layers = 1, bidirectional=False, dropout = 0.5) # 50% dropout prob

# L2-Regularization Optimizer 
optimizer = torch.optim.Adam(LSTM_model3.parameters(), lr=0.001, weight_decay=1e-5)

# Begin Training of LSTM and evaluation on validation set
train_losses3 = []
val_losses3 = []

counter = 0
max_increases = 5
best_val_loss = 9999999

for epoch in range(EPOCHS):
    # fit model
    train_loss = train_NN(train_loader, LSTM_model3, criterion, optimizer)
    
    print('EPOCH',str(epoch),':')
    print('Training Loss:',str(train_loss))    
    train_losses3.append(train_loss)
    
    # evaluate on validation set - comment out if unwanted
    val_loss, (y_gold, y_pred) = eval_NN(val_loader, LSTM_model3, criterion)
    print('Validation Loss:',str(val_loss))
    val_losses3.append(val_loss)
    
    # Apply Early Stopping Techniques
    if val_loss < best_val_loss:
        torch.save(LSTM_model3, "./early_model") # checkpoint
        best_val_loss = val_loss
        counter = 0 # reset counter
    else:
        counter += 1
    
    if counter == max_increases: # 10 times in a row no loss improvement - break
        print('Early Stopping')
        break

plot_losses(len(train_losses3), train_losses3, val_losses3, title = 'Losses per Epoch - Early Stopping')

# Use a Bidirectional LSTM with Early Stopping, Dropout and Regularization
bi_LSTM = BasicLSTM(input_dim = n_features, hidden_size = 64, 
                       output_dim = 10, num_layers = 1, bidirectional=True, dropout = 0.5) # 50% dropout prob

# L2-Regularization Optimizer 
optimizer = torch.optim.Adam(bi_LSTM.parameters(), lr=0.001, weight_decay=1e-5)

# Begin Training of LSTM and evaluation on validation set
train_losses4 = []
val_losses4 = []

counter = 0
max_increases = 5
best_val_loss = 9999999

for epoch in range(EPOCHS):
    # fit model
    train_loss = train_NN(train_loader, bi_LSTM, criterion, optimizer)
    
    print('EPOCH',str(epoch),':')
    print('Training Loss:',str(train_loss))    
    train_losses4.append(train_loss)
    
    # evaluate on validation set - comment out if unwanted
    val_loss, (y_gold, y_pred) = eval_NN(val_loader, bi_LSTM, criterion)
    print('Validation Loss:',str(val_loss))
    val_losses4.append(val_loss)
    
    # Apply Early Stopping Techniques
    if val_loss < best_val_loss:
        torch.save(bi_LSTM, "./early_bimodel") # checkpoint
        best_val_loss = val_loss
        counter = 0 # reset counter
    else:
        counter += 1
    
    if counter == max_increases: # 10 times in a row no loss improvement - break
        print('Early Stopping')
        break

plot_losses(len(train_losses4), train_losses4, val_losses4, title = 'Losses per Epoch - Bidirectional LSTM')

# Print Accuracy and Confusion Matrix of Best Model
best_model = torch.load('early_bimodel')

# evaluate on val and test sets
val_loss, (val_gold, val_pred) = eval_NN(val_loader, best_model, criterion)
test_loss, (test_gold, test_pred) = eval_NN(test_loader, best_model, criterion)

# Accuracies and Confusion Matrices
acc_val = accuracy_score(np.concatenate(val_gold), np.concatenate(val_pred))
acc_test = accuracy_score(np.concatenate(test_gold), np.concatenate(test_pred))
cm_val = confusion_matrix(np.concatenate(val_gold), np.concatenate(val_pred))
cm_test = confusion_matrix(np.concatenate(test_gold), np.concatenate(test_pred))

print('Best Model:')
print('Validation Set Accuracy:', acc_val)
print('Test Set Accuracy:', acc_test)

plot_confusion_matrix(cm_val, classes = [i for i in range(10)],
                          normalize=False,
                          title='Confusion matrix - Validation Set',
                          cmap=plt.cm.Blues)
plot_confusion_matrix(cm_test, classes = [i for i in range(10)],
                          normalize=False,
                          title='Confusion matrix - Test Set',
                          cmap=plt.cm.Blues)

##############################################################
# (Bonus) Use pack_padded_sequence for faster computations
n_features = X_train[0].shape[1]
EPOCHS = 100
criterion = nn.CrossEntropyLoss()

bi_LSTM_packed = BasicLSTM_packed(input_dim = n_features, hidden_size = 64, 
                       output_dim = 10, num_layers = 1, bidirectional=True, dropout = 0.5) # 50% dropout prob

# L2-Regularization Optimizer 
optimizer = torch.optim.Adam(bi_LSTM_packed.parameters(), lr=0.001, weight_decay=1e-5)

# Begin Training of LSTM and evaluation on validation set
train_losses5 = []
val_losses5 = []

counter = 0
max_increases = 5
best_val_loss = 9999999

for epoch in range(EPOCHS):
    # fit model
    train_loss = train_NN(train_loader, bi_LSTM_packed, criterion, optimizer, packed=True)
    
    print('EPOCH',str(epoch),':')
    print('Training Loss:',str(train_loss))    
    train_losses5.append(train_loss)
    
    # evaluate on validation set - comment out if unwanted
    val_loss, (y_gold, y_pred) = eval_NN(val_loader, bi_LSTM_packed, criterion, packed=True)
    print('Validation Loss:',str(val_loss))
    val_losses5.append(val_loss)
    
    # Apply Early Stopping Techniques
    if val_loss < best_val_loss:
        torch.save(bi_LSTM_packed, "./early_bimodel_packed") # checkpoint
        best_val_loss = val_loss
        counter = 0 # reset counter
    else:
        counter += 1
    
    if counter == max_increases: # 10 times in a row no loss improvement - break
        print('Early Stopping')
        break

plot_losses(len(train_losses5), train_losses5, val_losses5, title = 'Losses per Epoch - Packed Bidirectional LSTM')
