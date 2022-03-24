import os
import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class FrameLevelDataset(Dataset):
    def __init__(self, feats, labels):
        """
            feats: Python list of numpy arrays that contain the sequence features.
                   Each element of this list is a numpy array of shape seq_length x feature_dimension
            labels: Python list that contains the label for each sequence (each label must be an integer)
        """
        self.lengths =  [np.shape(sample)[0] for sample in feats]

        self.feats = self.zero_pad_and_stack(feats)
        if isinstance(labels, (list, tuple)):
            self.labels = np.array(labels).astype('int64')

    def zero_pad_and_stack(self, x):
        """
            This function performs zero padding on a list of features and forms them into a numpy 3D array
            returns
                padded: a 3D numpy array of shape num_sequences x max_sequence_length x feature_dimension
        """
        padded = []
        # --------------- Insert your code here ---------------- #
        max_length = np.max(self.lengths)
        for sample in x:
            padded.append(np.pad(sample, ((0,max_length-len(sample)),(0,0))))
        
        return np.array(padded)

    def __getitem__(self, item):
        return self.feats[item], self.labels[item], self.lengths[item]

    def __len__(self):
        return len(self.feats)


class BasicLSTM(nn.Module):
    def __init__(self, input_dim, hidden_size, output_dim, num_layers, bidirectional=False, dropout = 0):
        super(BasicLSTM, self).__init__()
        self.bidirectional = bidirectional
        self.feature_size = hidden_size * 2 if self.bidirectional else hidden_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # --------------- Insert your code here ---------------- #
        # Initialize the LSTM, Dropout, Output layers
        self.lstm = nn.LSTM(input_dim, hidden_size, num_layers, bidirectional=self.bidirectional, batch_first=True, dropout=dropout)
        self.linear = nn.Linear(self.feature_size, output_dim)

    def forward(self, x, lengths):
        """ 
            x : 3D numpy array of dimension N x L x D
                N: batch index
                L: sequence index
                D: feature index

            lengths: N x 1
         """
        
        # --------------- Insert your code here ---------------- #
        
        # You must have all of the outputs of the LSTM, but you need only the last one (that does not exceed the sequence length)
        # To get it use the last_timestep method
        # Then pass it through the remaining network
        
        if self.bidirectional:
            num_layers = 2*self.num_layers
        else:
            num_layers = self.num_layers
        
        h0 = torch.zeros(num_layers, x.size(0), self.hidden_size)
        c0 = torch.zeros(num_layers, x.size(0), self.hidden_size)
        
        # Forward through LSTM
        hn, _ = self.lstm(x, (h0, c0))
        # Final Linear Layer - pass last timestep
        last_outputs = self.linear(self.last_timestep(hn, lengths, self.bidirectional))

        return last_outputs

    def last_timestep(self, outputs, lengths, bidirectional=False):
        """
            Returns the last output of the LSTM taking into account the zero padding
        """
        if bidirectional:
            forward, backward = self.split_directions(outputs)
            last_forward = self.last_by_index(forward, lengths)
            last_backward = backward[:, 0, :]
            # Concatenate and return - maybe add more functionalities like average
            return torch.cat((last_forward, last_backward), dim=-1)

        else:
            return self.last_by_index(outputs, lengths)

    @staticmethod
    def split_directions(outputs):
        direction_size = int(outputs.size(-1) / 2)
        forward = outputs[:, :, :direction_size]
        backward = outputs[:, :, direction_size:]
        return forward, backward

    @staticmethod
    def last_by_index(outputs, lengths):
        # Index of the last output for each sequence.
        idx = (lengths - 1).view(-1, 1).expand(outputs.size(0),
                                               outputs.size(2)).unsqueeze(1)
        return outputs.gather(1, idx).squeeze()

class BasicLSTM_packed(nn.Module):
    def __init__(self, input_dim, hidden_size, output_dim, num_layers, bidirectional=False, dropout = 0):
        super(BasicLSTM_packed, self).__init__()
        self.bidirectional = bidirectional
        self.feature_size = hidden_size * 2 if self.bidirectional else hidden_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # --------------- Insert your code here ---------------- #
        # Initialize the LSTM, Dropout, Output layers
        self.lstm = nn.LSTM(input_dim, hidden_size, num_layers, bidirectional=self.bidirectional, batch_first=True, dropout=dropout)
        self.linear = nn.Linear(self.feature_size, output_dim)

    def forward(self, x, lengths):
        """ 
            x : 3D numpy array of dimension N x L x D
                N: batch index
                L: sequence index
                D: feature index

            lengths: N x 1
         """
        
        # --------------- Insert your code here ---------------- #
        
        # You must have all of the outputs of the LSTM, but you need only the last one (that does not exceed the sequence length)
        # To get it use the last_timestep method
        # Then pass it through the remaining network
        
        if self.bidirectional:
            num_layers = 2*self.num_layers
        else:
            num_layers = self.num_layers

        h0 = torch.zeros(num_layers, x.size(0), self.hidden_size)
        c0 = torch.zeros(num_layers, x.size(0), self.hidden_size)
        
        # Sort Inputs by decreasing length
        lengths, indices = lengths.sort(dim = 0, descending = True)
        x = x[indices] 

        packed_x = pack_padded_sequence(x, list(lengths.data), batch_first=True, enforce_sorted = True)
        
        # Forward through LSTM
        hn, _ = self.lstm(packed_x, (h0, c0))
        hn = pad_packed_sequence(hn, batch_first=True)[0]

        # Final Linear Layer - pass last timestep
        last_outputs = self.linear(self.last_timestep(hn, lengths, self.bidirectional))

        return last_outputs, indices # also return indices to allign data with labels

    def last_timestep(self, outputs, lengths, bidirectional=False):
        """
            Returns the last output of the LSTM taking into account the zero padding
        """
        if bidirectional:
            forward, backward = self.split_directions(outputs)
            last_forward = self.last_by_index(forward, lengths)
            last_backward = backward[:, 0, :]
            # Concatenate and return - maybe add more functionalities like average
            return torch.cat((last_forward, last_backward), dim=-1)

        else:
            return self.last_by_index(outputs, lengths)

    @staticmethod
    def split_directions(outputs):
        direction_size = int(outputs.size(-1) / 2)
        forward = outputs[:, :, :direction_size]
        backward = outputs[:, :, direction_size:]
        return forward, backward

    @staticmethod
    def last_by_index(outputs, lengths):
        # Index of the last output for each sequence.
        idx = (lengths - 1).view(-1, 1).expand(outputs.size(0),
                                               outputs.size(2)).unsqueeze(1)
        return outputs.gather(1, idx).squeeze()
