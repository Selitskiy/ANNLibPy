import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import math

from LrReLULayers import LrReLU, LrLReLU, LrELU
from cosSimAttention import cosPeTransformer, cosPcTransformer, cosPcTransformerMH
from channelMult import ChannelReplicate, PerChannelLinear, channelsLinear

# Define model
class OFTNN(nn.Module):
    def __init__(self, _inLen, _outLen, _actK=None, _actQ=None):
        super().__init__()
        
        #Encoder
        self.inLen = _inLen
        self.outLen = _outLen
        self.hid1Len = self.inLen*_outLen
        self.hid2Len = (self.inLen*2+1)*_outLen

        self.pet0 = cosPeTransformer(self.inLen)

        self.fc1 = nn.Linear(self.inLen, self.hid1Len)
        self.lrrelu1 = LrELU(self.hid1Len)
        self.dropout1 = nn.Dropout(p=0.2)
        self.fc2 = nn.Linear(self.hid1Len, self.hid2Len)
        self.lrrelu2 = LrELU(self.hid2Len)
        self.dropout2 = nn.Dropout(p=0.2)
        self.fc3 = nn.Linear(self.hid2Len, self.outLen)

    def forward(self, x):

        x = self.pet0(x)
        x = self.fc1(x)
        x = self.lrrelu1(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.lrrelu2(x)
        x = self.dropout2(x)
        x = self.fc3(x)

        return x
    

# Define model
class OFTTNN(nn.Module):
    def __init__(self, _inLen, _outLen, _bottleLen, _actK=None, _actQ=None):
        super().__init__()
        
        #Encoder
        self.inLen = _inLen
        self.prodLen = self.inLen*_outLen

        # Decoder
        self.bottleLen = _bottleLen
        self.outLen = _outLen
        self.hid1Len = self.bottleLen*_outLen
        self.hid2Len = (self.bottleLen*2+1)*_outLen

        #self.pet0 = cosPeTransformer(self.inLen)
        self.fc0 = nn.Linear(self.inLen, self.prodLen)
        self.pct0 = cosPcTransformer(self.prodLen, _actK, _actQ)

        self.fc1 = nn.Linear(self.prodLen, self.hid1Len)

        self.lrrelu1 = LrELU(self.hid1Len)
        #self.dropout1 = nn.Dropout(p=0.2)
        self.fc2 = nn.Linear(self.hid1Len, self.hid2Len)
        self.lrrelu2 = LrELU(self.hid2Len)
        #self.dropout2 = nn.Dropout(p=0.2)
        self.fc3 = nn.Linear(self.hid2Len, self.outLen)

    def forward(self, x):

        #x = self.pet0(x)
        x = self.fc0(x)
        x = self.pct0(x)

        x = self.fc1(x)

        x = self.lrrelu1(x)
        #x = self.dropout1(x)
        x = self.fc2(x)
        x = self.lrrelu2(x)
        #x = self.dropout2(x)
        x = self.fc3(x)

        return x
    

class OFTTCNN(nn.Module):
    def __init__(self, _inLen, _outLen, _bottleLen, _actK=None, _actQ=None):
        super().__init__()
        
        #Encoder
        self.inLen = _inLen
        self.prodLen = self.inLen*_outLen

        # Decoder
        self.bottleLen = _bottleLen
        self.outLen = 1
        self.channels = _outLen
        self.hid1Len = self.bottleLen
        self.hid2Len = (self.bottleLen*2+1)

        #self.pet0 = cosPeTransformer(self.inLen)
        #self.fc0 = nn.Linear(self.inLen, self.prodLen)
        self.pct0 = cosPcTransformer(self.inLen, _actK, _actQ)

        self.cr = ChannelReplicate(self.channels)
        self.fc1 = channelsLinear(self.channels, self.inLen, self.hid1Len)
        #self.fc1 = nn.Linear(self.prodLen, self.hid1Len)

        self.lrrelu1 = LrELU(self.hid1Len)
        #self.dropout1 = nn.Dropout(p=0.2)
        self.fc2 = channelsLinear(self.channels, self.hid1Len, self.hid2Len)
        #self.fc2 = nn.Linear(self.hid1Len, self.hid2Len)

        self.lrrelu2 = LrELU(self.hid2Len)
        #self.dropout2 = nn.Dropout(p=0.2)
        self.fc3 = channelsLinear(self.channels, self.hid2Len, self.outLen)
        #self.fc3 = nn.Linear(self.hid2Len, self.outLen)

        self.fl = nn.Flatten(start_dim=1)

    def forward(self, x):

        #x = self.pet0(x)
        #x = self.fc0(x)
        x = self.pct0(x)

        x = self.cr(x)
        x = self.fc1(x)

        x = self.lrrelu1(x)
        #x = self.dropout1(x)

        x = self.fc2(x)
        x = self.lrrelu2(x)
        #x = self.dropout2(x)

        x = self.fc3(x)
        x = self.fl(x)

        return x
    

class OFTTCNN2(nn.Module):
    def __init__(self, _inLen, _outLen, _bottleLen, _actK=None, _actQ=None):
        super().__init__()
        
        #Encoder
        self.inLen = _inLen
        self.prodLen = self.inLen*_outLen

        # Decoder
        self.bottleLen = _bottleLen
        self.outLen = 1
        self.channels = _outLen
        self.hid1Len = self.bottleLen
        self.hid2Len = (self.bottleLen*2+1)

        #self.pet0 = cosPeTransformer(self.inLen)
        #self.fc0 = nn.Linear(self.inLen, self.prodLen)
        

        self.cr = ChannelReplicate(self.channels)
        self.pct0 = cosPcTransformerMH(self.channels, self.inLen, _actK, _actQ)

        self.fc1 = channelsLinear(self.channels, self.inLen, self.hid1Len)
        #self.fc1 = nn.Linear(self.prodLen, self.hid1Len)

        self.lrrelu1 = LrELU(self.hid1Len)
        #self.dropout1 = nn.Dropout(p=0.2)
        self.fc2 = channelsLinear(self.channels, self.hid1Len, self.hid2Len)
        #self.fc2 = nn.Linear(self.hid1Len, self.hid2Len)

        self.lrrelu2 = LrELU(self.hid2Len)
        #self.dropout2 = nn.Dropout(p=0.2)
        self.fc3 = channelsLinear(self.channels, self.hid2Len, self.outLen)
        #self.fc3 = nn.Linear(self.hid2Len, self.outLen)

        self.fl = nn.Flatten(start_dim=1)

    def forward(self, x):

        #x = self.pet0(x)
        #x = self.fc0(x)
        

        x = self.cr(x)
        x = self.pct0(x)

        x = self.fc1(x)

        x = self.lrrelu1(x)
        #x = self.dropout1(x)

        x = self.fc2(x)
        x = self.lrrelu2(x)
        #x = self.dropout2(x)

        x = self.fc3(x)
        x = self.fl(x)

        return x