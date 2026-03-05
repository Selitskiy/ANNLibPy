import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ChannelReplicate(nn.Module):
    def __init__(self, out_channels):
        super(ChannelReplicate, self).__init__()

        self.out_channels = out_channels

    def forward(self, input):
        with torch.no_grad():
            # input: [B, E]
            x = input.unsqueeze(1)
            # x: [B, 1, E]
            x = x.repeat(1, self.out_channels, 1)  # Duplicate along channel dim

        return x

class PerChannelLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super(PerChannelLinear, self).__init__()
        self.fc = nn.Linear(in_features, out_features)

    def forward(self, x):
        # x shape: (batch_size, channels, features)
        return self.fc(x)  # Linear applied to last dimension
        

class ChannelClone(nn.Module):
    def __init__(self, out_channels):
        super(ChannelClone, self).__init__()
        self.out_channels = out_channels

    def forward(self, x):
        x = x.unsqueeze(1)
        # x: [B, 1, X]
        x = x.repeat(1, self.out_channels, 1)  # Duplicate along channel dim
        return x
    

class channelsLinear(nn.Module):
    def __init__(self, channels, in_size, out_size):
        super(channelsLinear, self).__init__()

        self.in_size = in_size
        self.channels = channels
        self.out_size = out_size
            
        # Initialize weight coefficients - glorot
        self.W = nn.Parameter(torch.empty(self.channels, self.in_size, self.out_size))
        self.W0 = nn.Parameter(torch.zeros(self.channels, self.out_size))

        nn.init.xavier_uniform_(self.W)


    def forward(self, input):

        # Perform per-channel matrix multiplication
        #Y = torch.stack([
        #    torch.matmul(input[:, c, :], self.W[c])  # matmul for each channel
        #    for c in range(self.channels)
        #], dim=1)

        # input shape: (batch_size, channels, in_size)
        # self.W shape: (channels, in_size, out_size)
        # output shape: (batch_size, channels, out_size)
        Y = torch.einsum('bci,cio->bco', input, self.W)

        Z = Y + self.W0

        #print(result.shape)  # (B, C, H, W)
        return Z