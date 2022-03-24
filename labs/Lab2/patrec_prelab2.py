#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ECE NTUA: Flow S - Signals, Control and Robotics
Pattern Recognition - 9th Semester
Prelab 2: Speech Recognition with HMMs and RNNs

Author: Christos Dimopoulos - 03117037
"""

import numpy as np
import matplotlib.pyplot as plt
import librosa
import re
import os
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
"""
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
from torch import optim
"""
from lib import * # Lab1 Classifiers
plt.close('all')

n1 = 3
n2 = 7

# Step 2: Create data parser to read .wav files
n_files = 133
digits_directory = './pr_lab2_2020-21_data/digits'
# Dictionary to convert string digits in their numerical format
str2num = {'one':1, 'two':2, 'three':3, 'four':4, 'five':5, 'six':6,
          'seven':7, 'eight':8, 'nine':9}

# Simple parser that split wav files as regular expression
# e.g. one7.wav --> 'one' + '7'
def split_digitname(s):
    return re.split(r'(\d+)', s)[:2]

# Main data parser method
def data_parser(directory):
    wav = []
    speaker = np.zeros(n_files, dtype=int)
    digit = np.zeros(n_files, dtype=int)
    
    for i, file in enumerate(os.listdir(directory)):
        filename = os.fsdecode(file)
        
        # Read wav file
        wav_sample,_ = librosa.load(os.path.join(directory, filename), sr=16000)
        wav.append(wav_sample)
        
        # Read Speaker and Digit
        decoded_name = split_digitname(filename.split('.')[0])
        digit[i] = str2num[decoded_name[0]]
        speaker[i] = int(decoded_name[1])
        
    return wav, speaker, digit

wav, speaker, digit = data_parser(digits_directory)

print('Print Data Sample Number 10:')
print('Raw wav: ', wav[9],'\nSpeaker:',speaker[9], '\nDigit:',digit[9])

# Step 3: MFCC and delta, delta-delta extraction
n_mfcc = 13
window_len = 0.025 # sec
step = 0.01 # sec
Fs = 16000 # Sampling Frequency (Hz)

# Convert to points
n_fft = int(window_len*Fs)  
hop_length = int(step*Fs)

mfccs = []
deltas = []
delta_deltas = []

for i in range(n_files):
    mfcc = librosa.feature.mfcc(wav[i], sr=Fs, hop_length=hop_length, n_fft=n_fft, n_mfcc=n_mfcc)
    mfccs.append(mfcc)
    deltas.append(librosa.feature.delta(mfcc))
    delta_deltas.append(librosa.feature.delta(mfcc, order=2))

# Step 4: Create Histograms of 1st and 2nd MFCCs for Digits 3 and 7

# Extract 1st and 2nd MFCCs of digits 3 and 7
mfcc1_d3 = np.concatenate([mfccs[i][0] for i in range(n_files) if digit[i] == n1])
mfcc2_d3 = np.concatenate([mfccs[i][1] for i in range(n_files) if digit[i] == n1])

# Extract 1st and 2nd mfcc of digit 9
mfcc1_d7 = np.concatenate([mfccs[i][0] for i in range(n_files) if digit[i] == n2])
mfcc2_d7 = np.concatenate([mfccs[i][1] for i in range(n_files) if digit[i] == n2])

# Plot histograms
fig = plt.figure()
ax1 = fig.add_subplot(2, 2, 1)
plt.hist(mfcc1_d3, bins=30, label='1st MFCC - Digit 3')
ax2 = fig.add_subplot(2, 2, 2)
plt.hist(mfcc2_d3, bins=30, label='2nd MFCC - Digit 3')
ax3 = fig.add_subplot(2, 2, 3)
plt.hist(mfcc1_d7, bins=30, label = '1st MFCC - Digit 7')
ax4 = fig.add_subplot(2, 2, 4)
plt.hist(mfcc2_d7, bins=30, label = '2nd MFCC - Digit 7')
ax1.legend()
ax2.legend()
ax3.legend()
ax4.legend()
plt.show()

# Compare MFSCs with MFCCs wrt correlation

# Function that returns 2 indexes of same digit n but different speakers
def same_digit_different_speakers(digit, n = 3):
    digit_idx = np.where(digit==n)[0]
    spk1_d = digit_idx[0]
    
    for i in range(len(digit_idx)):
        if (digit_idx[0]!=digit_idx[i]): # different speakers
            spk2_d = digit_idx[i]
            break
    return spk1_d, spk2_d

spk1_d3, spk2_d3 = same_digit_different_speakers(digit, n = n1)
spk1_d7, spk2_d7 = same_digit_different_speakers(digit, n = n2)

# Extract MFSCs
mfscs_spk1_d3 = librosa.feature.melspectrogram(wav[spk1_d3], sr=Fs, hop_length=hop_length, n_fft=n_fft, n_mels=13)
mfscs_spk2_d3 = librosa.feature.melspectrogram(wav[spk2_d3], sr=Fs, hop_length=hop_length, n_fft=n_fft, n_mels=13)
mfscs_spk1_d7 = librosa.feature.melspectrogram(wav[spk1_d7], sr=Fs, hop_length=hop_length, n_fft=n_fft, n_mels=13)
mfscs_spk2_d7 = librosa.feature.melspectrogram(wav[spk2_d7], sr=Fs, hop_length=hop_length, n_fft=n_fft, n_mels=13)

# Convert to Dataframes and plot correlation
fig = plt.figure()

counter = 1
for mfsc in [mfscs_spk1_d3, mfscs_spk2_d3, mfscs_spk1_d7, mfscs_spk2_d7]:
    fig.add_subplot(2, 2, counter)
    mfsc = pd.DataFrame.from_records(mfsc.T)
    plt.imshow(mfsc.corr())
    counter +=1

plt.show()

# Repeat for MFCCs
fig = plt.figure()

counter = 1
for mfsc in [mfccs[spk1_d3],mfccs[spk2_d3],mfccs[spk1_d7],mfccs[spk1_d7]]:
    fig.add_subplot(2, 2, counter)
    mfsc = pd.DataFrame.from_records(mfsc.T)
    plt.imshow(mfsc.corr())
    counter +=1

plt.show()

# Step 5: Create Feature Vectors and Scatter Plot

# X Vector: (mean MFCC) - (std MFCC) - (mean Delta) - (std Delta) - (Mean Delta-Delta) - (Std Delta-Delta)
X = np.zeros((n_files, 6*13))

for i in range(n_files):
    mfcc_mean = np.mean(mfccs[i], axis=1)
    mfcc_std = np.std(mfccs[i], axis=1)
    delta_mean = np.mean(deltas[i], axis=1)
    delta_std = np.std(deltas[i], axis=1)
    deltadelta_mean = np.mean(delta_deltas[i], axis=1)
    deltadelta_std = np.std(delta_deltas[i], axis=1)
    X[i,:] = np.concatenate(np.vstack((mfcc_mean, mfcc_std, delta_mean, delta_std, deltadelta_mean, deltadelta_std)))

# Similar to plt_clf of python notebooks
def scatter_2D(X, y, labels):
    fig, ax = plt.subplots()
    X0, X1 = X[:, 0], X[:, 1]
    
    one = ax.scatter(
        X0[y == 1], X1[y == 1],
        c='red', label=labels[0], 
        s=50, alpha=0.9, edgecolors='k')
    
    two = ax.scatter(
        X0[y == 2], X1[y == 2],
        c='blue', label=labels[1], 
        s=50, alpha=0.9, edgecolors='k')
    
    three = ax.scatter(
        X0[y == 3], X1[y == 3],
        c='green', label=labels[2],
        s=50, alpha=0.9, edgecolors='k')
    
    four = ax.scatter(
        X0[y == 4], X1[y == 4],
        c='yellow', label=labels[3], 
        s=50, alpha=0.9, edgecolors='k')
    
    five = ax.scatter(
        X0[y == 5], X1[y == 5],
        c='magenta', label=labels[4], 
        s=50, alpha=0.9, edgecolors='k')
    
    six = ax.scatter(
        X0[y == 6], X1[y == 6],
        c='black', label=labels[5],
        s=50, alpha=0.9, edgecolors='k')
    
    seven = ax.scatter(
        X0[y == 7], X1[y == 7],
        c='white', label=labels[6], 
        s=50, alpha=0.9, edgecolors='k')
    
    eight = ax.scatter(
        X0[y == 8], X1[y == 8],
        c='gray', label=labels[7], 
        s=50, alpha=0.9, edgecolors='k')
    
    nine = ax.scatter(
        X0[y == 9], X1[y == 9],
        c='brown', label=labels[8],
        s=50, alpha=0.9, edgecolors='k')

    ax.set_xticks(())
    ax.set_yticks(())
    ax.legend()
    plt.show()

scatter_2D(X, digit, [i for i in range(1,10)])

# Step 6: PCA and 3D Scatter
pca_2D = PCA(n_components=2)
X_2D = pca_2D.fit_transform(X)
scatter_2D(X_2D, digit, [i for i in range(1, 10)])

def scatter_3D(X, y, labels):
    fig, ax = plt.subplots()
    ax = fig.add_subplot(111, projection='3d')
    # title for the plots
    # Set-up grid for plotting.
    X0, X1, X2 = X[:, 0], X[:, 1], X[:,2]
    
    one = ax.scatter(
        X0[y == 1], X1[y == 1], X2[y == 1],
        c='red', label=labels[0],
        s=50, alpha=0.9, edgecolors='k')

    two = ax.scatter(
        X0[y == 2], X1[y == 2], X2[y == 2],
        c='blue', label=labels[1], 
        s=50, alpha=0.9, edgecolors='k')
    
    three = ax.scatter(
        X0[y == 3], X1[y == 3], X2[y == 3],
        c='green', label=labels[2],
        s=50, alpha=0.9, edgecolors='k')
    
    four = ax.scatter(
        X0[y == 4], X1[y == 4], X2[y == 4],
        c='yellow', label=labels[3], 
        s=50, alpha=0.9, edgecolors='k')
    
    five = ax.scatter(
        X0[y == 5], X1[y == 5], X2[y == 5],
        c='magenta', label=labels[4], 
        s=50, alpha=0.9, edgecolors='k')
    
    six = ax.scatter(
        X0[y == 6], X1[y == 6], X2[y == 6],
        c='black', label=labels[5],
        s=50, alpha=0.9, edgecolors='k')
    
    seven = ax.scatter(
        X0[y == 7], X1[y == 7], X2[y == 7],
        c='white', label=labels[6], 
        s=50, alpha=0.9, edgecolors='k')
    
    eight = ax.scatter(
        X0[y == 8], X1[y == 8], X2[y == 8],
        c='gray', label=labels[7], 
        s=50, alpha=0.9, edgecolors='k')
    
    nine = ax.scatter(
        X0[y == 9], X1[y == 9], X2[y == 9],
        c='brown', label=labels[8],
        s=50, alpha=0.9, edgecolors='k')
    
    ax.set_xticks(())
    ax.set_yticks(())
    ax.legend()
    plt.show()

pca_3D = PCA(n_components=3)
X_3D = pca_3D.fit_transform(X)
scatter_3D(X_3D, digit, [i for i in range(1,10)])

print('Variance Ratio after 2D PCA:', pca_2D.explained_variance_ratio_)
print('Variance Ratio after 3D PCA:', pca_3D.explained_variance_ratio_)

# Step 7: Train-Test Split and Model Evaluation
X_train, X_test, y_train, y_test = train_test_split(X, digit, test_size=0.3)

# Normalize Data
X_train = preprocessing.normalize(X_train)
X_test = preprocessing.normalize(X_test)

# Custom NB Classifier from Lab1
myNB1 = CustomNBClassifier(use_unit_variance=False)
myNB2 = CustomNBClassifier(use_unit_variance=True) # Unit Variance - Euclidean Distance Clf

# Experiment on other sklearn classifiers as well
NB_sklrn = GaussianNB()
SVM_linear = SVC(kernel='linear')
SVM_poly = SVC(kernel='poly')
SVM_rbf = SVC(kernel='rbf')
KNN3= KNeighborsClassifier(n_neighbors=3)
KNN5 = KNeighborsClassifier(n_neighbors=5)
LR = LogisticRegression()

# Train on train set and evaluate on test set
print()
scores = []
for clf in [myNB1, myNB2, NB_sklrn, SVM_linear, SVM_poly, SVM_rbf, KNN3, KNN5, LR]:
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_train)
    acc = accuracy_score(y_train, y_pred)
    scores.append(acc)
    print(clf,'- Accuracy:',acc)


# BONUS: Add zero-crossing rates
zero_crossing_rates = []

for i in range(n_files):
    zc = librosa.feature.zero_crossing_rate(wav[i], hop_length=hop_length, frame_length=n_fft)
    zero_crossing_rates.append(zc[0][:50]) #keep first 50 to have the same dimension

Xnew = np.zeros((n_files, 50+6*13))

for i in range(n_files):
    Xnew[i,:] = np.concatenate((X[i,:], zero_crossing_rates[i]))

##############################################################################
print('\n================================================================')    
print('Experiment with Zero-Crossing Rates in Feature Vector')
X_train, X_test, y_train, y_test = train_test_split(Xnew, digit, test_size=0.3)

# Normalize Data
X_train = preprocessing.normalize(X_train)
X_test = preprocessing.normalize(X_test)

# Custom NB Classifier from Lab1
myNB1 = CustomNBClassifier(use_unit_variance=False)
myNB2 = CustomNBClassifier(use_unit_variance=True) # Unit Variance - Euclidean Distance Clf

# Experiment on other sklearn classifiers as well
NB_sklrn = GaussianNB()
SVM_linear = SVC(kernel='linear')
SVM_poly = SVC(kernel='poly')
SVM_rbf = SVC(kernel='rbf')
KNN3= KNeighborsClassifier(n_neighbors=3)
KNN5 = KNeighborsClassifier(n_neighbors=5)
LR = LogisticRegression()

# Train on train set and evaluate on test set
print()
scores = []
for clf in [myNB1, myNB2, NB_sklrn, SVM_linear, SVM_poly, SVM_rbf, KNN3, KNN5, LR]:
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_train)
    acc = accuracy_score(y_train, y_pred)
    scores.append(acc)
    print(clf,'- Accuracy:',acc)

"""
# Step 8: Pytorch RNNs and LSTMs

f = 40 # sine frequency in Hz
n_samples = 1000 # total number of sines created
n_features = 10  # total number of points per sine
dx = 0.001 # distance between points

sines = np.zeros((n_samples, n_features))
cosines = np.zeros(sines.shape)

for i in range(n_samples):
    # Amplitude is Random Variable Uniformally Distributed in [1,10]
    A = np.random.uniform(1,10)
    # Time is Random Variable Uniformally Distributed in [0,T]
    ts = np.random.uniform(0, 1/f)
    tf = ts + n_features*dx
    t = np.linspace(ts, tf, n_features)
    
    sines[i] = A*np.sin(2*np.pi*f*t)
    cosines[i] = A*np.cos(2*np.pi*f*t)
    
# Plot first 8 generated sines and cosines
fig = plt.figure()

for i in range(0,8):
    fig.add_subplot(2,4,i+1)
    plt.plot(np.arange(10), sines[i][:], '-o')
    plt.plot(np.arange(10), cosines[i][:], '-o')

plt.show()

# Create RNN Class
class RNN(nn.Module):
    def __init__(self, input_size=1, n_layers = 1, hidden_size=150, output_size=1, activation = 'relu'):
        super().__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.rnn = nn.RNN(input_size, hidden_size, nonlinearity = activation, num_layers = n_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, input):
        outs = []
        batch_size = input.size(0)
        # Hidden State at time t = 0
        ht = torch.zeros(self.n_layers,batch_size, self.hidden_size, dtype=torch.double)
        for i, input_t in enumerate(input.chunk(input.size(1), dim=1)):
            output, ht = self.rnn(input_t.unsqueeze(-1), ht)
            output = self.linear(ht)
            outs.append(output[0])
        outs = torch.stack(outs, 1).squeeze(2)
        return outs

# Create Class of LSTM
class LSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=150, output_size=1):
        super().__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTMCell(input_size, hidden_size)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, input):
        outs = []
        batch_size = input.size(0)
        # Hidden State at time t = 0
        ht = torch.zeros(batch_size, self.hidden_size, dtype=torch.double)
        # Cell State at time t = 0
        ct = torch.zeros(batch_size, self.hidden_size, dtype=torch.double)

        for i, input_t in enumerate(input.chunk(input.size(1), dim=1)):
            ht, ct = self.lstm(input_t, (ht, ct))
            output = self.linear(ht)
            outs.append(output)
        outs = torch.stack(outs, 1).squeeze(2)
        return outs

# Prepare our Train and Test Sets
X_train, X_test, y_train, y_test = train_test_split(sines, cosines, test_size=0.3)

# Convert to torch tensors
X_train = torch.from_numpy(X_train)
y_train = torch.from_numpy(y_train)
X_test = torch.from_numpy(X_test)
y_test = torch.from_numpy(y_test)

def model_train(EPOCHS, model, lr, criterion, X_train, y_train, X_test, y_test):
    train_losses = []
    val_losses = []
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for i in range(EPOCHS+1):
        # Train Model
        optimizer.zero_grad()
        y_pred = model(X_train.double())
        train_loss = criterion(y_pred, y_train)
        train_loss.backward()
        optimizer.step()
        
        # Evaluate validation set
        with torch.no_grad():
            val_pred = model(X_test.double())
            val_loss = criterion(val_pred, y_test)
        
        # Print losses every 50 epochs
        if i%50 == 0:
            print('Epoch',i)
            print('- Train Loss:', train_loss.item())
            print('- Validation Loss:', val_loss.item())
        
        # Collect Losses
        train_losses.append(train_loss.item())
        val_losses.append(val_loss.item())
    
    return train_losses, val_losses
    
# Train RNN and LSTM Model on Given Data
RNN_model = RNN(input_size=1, hidden_size=150, output_size=1, activation = 'relu').double()
LSTM_model = LSTM(input_size=1, hidden_size=150, output_size=1).double()
criterion = nn.MSELoss()
EPOCHS = 1000
lr = 0.001

############################ Train RNN Model ##################################

print('=========== RNN Model =============')
train_losses, val_losses = model_train(EPOCHS, RNN_model, lr, criterion, X_train, y_train, X_test, y_test)

# Plot Losses
plt.figure()
plt.plot(np.arange(EPOCHS+1), train_losses, label='Training Loss', color='b')
plt.plot(np.arange(EPOCHS+1), val_losses, label='Validation Loss', color='r')
plt.title('RNN Model - Losses per Epoch')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


# Show first 12 Predicted Cosines and compare with ground truth
fig = plt.figure()
for i in range(12):
    fig.add_subplot(4, 3, i+1)
    plt.axis('off')
    with torch.no_grad():
        y_pred = RNN_model(X_test[i].view(1,-1))
    plt.plot(np.arange(10), y_pred[0], '-o')
    plt.plot(np.arange(10), y_test[i])
plt.show()

############################ Train LSTM Model ##################################

print('=========== LSTM Model =============')
train_losses, val_losses = model_train(EPOCHS, LSTM_model, lr, criterion, X_train, y_train, X_test, y_test)

# Plot Losses
plt.figure()
plt.plot(np.arange(EPOCHS+1), train_losses, label='Training Loss', color='b')
plt.plot(np.arange(EPOCHS+1), val_losses, label='Validation Loss', color='r')
plt.title('LSTM Model - Losses per Epoch')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


# Show first 12 Predicted Cosines and compare with ground truth
fig = plt.figure()
for i in range(12):
    fig.add_subplot(4, 3, i+1)
    plt.axis('off')
    with torch.no_grad():
        y_pred = LSTM_model(X_test[i].view(1,-1))
    plt.plot(np.arange(10), y_pred[0], '-o')
    plt.plot(np.arange(10), y_test[i])
plt.show()
"""