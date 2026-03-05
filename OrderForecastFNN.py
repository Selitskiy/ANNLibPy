import torch
#from torch.utils.data import Dataset
from torch import nn
import numpy as np

from LrReLULayers import LrReLU, LrLReLU, LrELU
from channelMult import ChannelReplicate, PerChannelLinear, channelsLinear

# Kolmogorov-Arnold emulation with Learning ReLUs, common matrixes for multiple output dimensions
class OFANN(nn.Module):
    def __init__(self, _inLen, _outLen):
        super().__init__()
        
        self.inLen = _inLen
        self.outLen = _outLen
        self.hid1Len = _inLen*_outLen
        self.hid2Len = (_inLen*2+1)*_outLen

        #self.linear_stack = nn.Sequential(
            #nn.Dropout(p=0.1),
        #    nn.Linear(self.inLen, self.hid1Len),
            #nn.CELU(),
        #    LrReLU(self.hid1Len),
            #nn.Tanh(),
        #    nn.Dropout(p=0.1),
        #    nn.Linear(self.hid1Len,self.hid2Len),
            #nn.CELU(),
        #    nn.Tanh(),
        #    nn.Dropout(p=0.1),
        #    nn.Linear(self.hid2Len, self.outLen),
        #)

        self.fc1 = nn.Linear(self.inLen, self.hid1Len)
        self.lrrelu1 = LrELU(self.hid1Len)
        #self.tanh1 = nn.Tanh()
        self.dropout1 = nn.Dropout(p=0.2)
        self.fc2 = nn.Linear(self.hid1Len, self.hid2Len)
        self.lrrelu2 = LrELU(self.hid2Len)
        #self.tanh2 = nn.Tanh()
        self.dropout2 = nn.Dropout(p=0.2)
        self.fc3 = nn.Linear(self.hid2Len, self.outLen)

    def forward(self, x):
        #logits = self.linear_stack(x)
        x = self.fc1(x)
        x = self.lrrelu1(x)
        #x = self.tanh1(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.lrrelu2(x)
        #x = self.tanh2(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        return x
    

class OFCANN(nn.Module):
    def __init__(self, _inLen, _outLen):
        super().__init__()
        
        self.inLen = _inLen
        self.outLen = 1
        self.channels = _outLen
        self.hid1Len = _inLen
        self.hid2Len = (_inLen*2+1)

        #self.fc1 = nn.Linear(self.inLen, self.hid1Len*self.channels)
        #self.ufl = nn.Unflatten(1, (self.channels, self.hid1Len))

        self.cr = ChannelReplicate(self.channels)
        #self.fc1 = PerChannelLinear(self.inLen, self.hid1Len)
        self.fc1 = channelsLinear(self.channels, self.inLen, self.hid1Len)

        self.lrrelu1 = LrELU(self.hid1Len)
        self.dropout1 = nn.Dropout(p=0.2)

        #self.fc2 = PerChannelLinear(self.hid1Len, self.hid2Len)
        self.fc2 = channelsLinear(self.channels, self.hid1Len, self.hid2Len)
        self.lrrelu2 = LrELU(self.hid2Len)
        self.dropout2 = nn.Dropout(p=0.2)
        
        #self.fc3 = PerChannelLinear(self.hid2Len, self.outLen)
        self.fc3 = channelsLinear(self.channels, self.hid2Len, self.outLen)

        self.fl = nn.Flatten(start_dim=1)

    def forward(self, x):

        #x = self.fc1(x)
        #x = self.ufl(x)

        x = self.cr(x)
        x = self.fc1(x)
        
        x = self.lrrelu1(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.lrrelu2(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        
        x = self.fl(x)
        return x