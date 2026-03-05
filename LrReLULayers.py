import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# Learning Rate ReLU (neuron-specific)
class LrReLU(nn.Module):
    def __init__(self, in_size, slope=0):
        super(LrReLU, self).__init__()

        self.in_size = in_size
        self.slope = slope

        #bound = math.sqrt(3 / in_size)

        # Define learnable weights 
        if slope == 0:
            #self.A = nn.Parameter(1+(2*torch.rand(in_size)-1) * bound)
            self.A = nn.Parameter(torch.rand(in_size))
        else:
            self.A = nn.Parameter(torch.ones(in_size) * slope)

        # Non-learnable tensor
        #self.O = torch.zeros(in_size)  # Not registered as a parameter

    def forward(self, input):
        # Clamp A in-place: set values > slope to slope, < 0 to 0
        with torch.no_grad():
            if self.slope:
                self.A.data.clamp_(0, self.slope)
                #self.A.data[self.A.data > self.slope] = self.slope
            else:
                self.A.data.clamp_(0)                       
            #self.A.data[self.A.data < 0] = 0
            
        X = input * (input >= 0)

        Y = X * self.A
        return Y
    
# Learning Rate Leaky ReLU (neuron-specific)
class LrLReLU(nn.Module):
    def __init__(self, in_size, slopeA=0, slopeB=0):
        super(LrLReLU, self).__init__()

        self.in_size = in_size
        self.slopeA = slopeA
        self.slopeB = slopeB


        # Define learnable weights 
        if slopeA == 0:
            self.A = nn.Parameter(torch.rand(in_size))
        else:
            self.A = nn.Parameter(torch.ones(in_size) * slopeA)

        if slopeB == 0:
            self.B = nn.Parameter(torch.rand(in_size) * .1)
        else:
            self.B = nn.Parameter(torch.ones(in_size) * slopeB)

    def forward(self, input):
        # Clamp A in-place: set values > slope to slope, < 0 to 0
        with torch.no_grad():
            if self.slopeA:
                self.A.data.clamp_(0, self.slopeA)
            else:
                self.A.data.clamp_(0)      

            if self.slopeB:
                self.B.data.clamp_(0, self.slopeB)
            else:
                self.B.data.clamp_(0)                 
            
        PM = input>=0
        NM = input<0
        YP = input * PM * self.A
        YN = input * NM * self.B
        Y = YP + YN

        return Y    

# Learning Rate Exponential tail ReLU (neuron-specific)
class LrELU(nn.Module):
    def __init__(self, in_size, slope=0, alpha=0):
        super(LrELU, self).__init__()

        self.in_size = in_size
        self.slope = slope
        self.alpha = alpha

        # Define learnable weights 
        if slope == 0:
            self.A = nn.Parameter(torch.rand(in_size))
        else:
            self.A = nn.Parameter(torch.ones(in_size) * slope)

        if alpha == 0:
            self.B = nn.Parameter(torch.rand(in_size))
        else:
            self.B = nn.Parameter(torch.ones(in_size) * alpha)

    def forward(self, input):
        # Clamp A in-place: set values > slope to slope, < 0 to 0
        #with torch.no_grad():
        if self.slope:
            self.A.data.clamp_(0, self.slope)
        else:
            self.A.data.clamp_(0)      

        if self.alpha:
            self.B.data.clamp_(0, self.alpha)
        else:
            self.B.data.clamp_(0)                 
            
        PM = input>=0
        NM = input<0
        YP = input * PM * self.A
        YN = self.B * (torch.exp(input * NM) - 1) 
        Y = YP + YN

        return Y  
    
# Test
if __name__ == "__main__":
    # Create dummy input: 
    batch_size=5 
    features=10
    input_tensor = torch.rand(batch_size, features)

    # Create instance of custom layer
    layer = LrReLU(features, 0.5)

    # Pass input through layer
    output_tensor = layer(input_tensor)
    print(output_tensor)

#class CustomLoss(nn.Module):
#    def __init__(self, weight):
#        super(CustomLoss, self).__init__()
#        self.weight = weight

#    def forward(self, input, target):
        # Compute the loss
#        loss = torch.mean(self.weight * (input - target) ** 2)
#        return loss