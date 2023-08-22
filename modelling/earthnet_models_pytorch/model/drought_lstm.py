"""ConvLSTM_LSTM
"""

from typing import Optional, Union, List

import argparse
import ast
import torch.nn as nn
import torch 
import sys
from earthnet_models_pytorch.utils import str2bool

class LSTM_oneshot(nn.Module):

    def __init__(self, hparams):
        """
        Initialize LSTM cell.
        Parameters
        ----------
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        num_layers: int
            Size of the convolutional kernel.
        """
        super().__init__()

        self.hparams = hparams
        self.input_dim = self.hparams.input_dim
        self.hidden_dim = self.hparams.hidden_dim
        self.output_dim = self.hparams.output_dim
        self.num_layers = self.hparams.num_layers
        self.target_length = self.hparams.target_length #number of timesteps to predict

        # LSTM cell
        self.lstm_cell = nn.LSTM(self.input_dim, self.hidden_dim, self.num_layers, batch_first=True)

        # Define the output layer 
        self.fc = nn.Linear(self.hidden_dim, self.output_dim)
    
   
    def forward(self, x):
        
        # Turn 5-D into 3-D tensor (remove height, width of pixel timeseries)
        x = torch.squeeze(x, (3,4))

        # Initialize the hidden state and cell state with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim, device=x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim, device=x.device)

        # Store the predictions in a list
        predictions  = torch.zeros(x.size(0), self.target_length, self.output_dim)

        # Iterate over the prediction steps
        if self.target_length>1:
            for i in range(self.target_length):
                # Use the hidden state from the previous timestep as the initial hidden state
                # for the next timestep
                if i == 0:
                    self.hidden = (h0, c0)
                else:
                    self.hidden = (self.hidden[0].detach(), self.hidden[1].detach())
                    # the hidden state and cell state are detached from their previous values, which is required to avoid backpropagation through time.

                # Forward pass through the LSTM layer
                lstm_out, self.hidden = self.lstm_cell(x, self.hidden)

                # Get the prediction and add it to the list
                y_pred = self.fc(lstm_out[:, -1, :])
                predictions[:,i,:] = y_pred
        else:
            lstm_out, self.hidden = self.lstm_cell(x, (h0,c0))
            # Get the prediction and add it to the list
            y_pred = self.fc(lstm_out[:, -1, :])
            predictions[:,0,:] = y_pred

        return predictions


    def init_hidden(self, batch_size, height, width):
        return (torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device),
                torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device))



    @staticmethod
    def add_model_specific_args(parent_parser: Optional[Union[argparse.ArgumentParser,list]] = None):
        if parent_parser is None:
            parent_parser = []
        elif not isinstance(parent_parser, list):
            parent_parser = [parent_parser]
        
        parser = argparse.ArgumentParser(parents = parent_parser, add_help = False)

        parser.add_argument("--setting", type = str, default = "en21x")

        parser.add_argument("--input_dim", type = int, default = 32)
        parser.add_argument("--hidden_dim", type = int, default = 128)
        parser.add_argument("--num_layers", type = int, default = 4)
        parser.add_argument("--output_dim", type = int, default = 1)
        parser.add_argument("--target_length", type = int, default = 2)

        return parser

