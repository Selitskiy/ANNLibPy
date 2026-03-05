import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# Cosine Similarity Attention (between input vectors - elements; attention matrix element-wise normalisation; linear Q and K transformations)
class cosPeTransformer(nn.Module):
    def __init__(self, in_size):
        super(cosPeTransformer, self).__init__()

        # square K and Q matrixes
        self.in_size = in_size
        self.out_size = in_size
            
        # Initialize weight coefficients - glorot
        bound = math.sqrt(6 / (self.out_size + self.in_size))
        self.Wq = nn.Parameter(bound * (2. * torch.rand(self.out_size, self.in_size) - 1.))
        self.Wq0 = nn.Parameter(torch.zeros(self.out_size))

        self.Wk = nn.Parameter(bound * (2. * torch.rand(self.out_size, self.in_size) - 1.))
        self.Wk0 = nn.Parameter(torch.zeros(self.out_size))


    def forward(self, input):
        eps=1e-12
        if input.dim() == 1:
            input = input.unsqueeze(0)
        
        K = input @ self.Wk + self.Wk0
        Q = input @ self.Wq + self.Wq0

        DK2 = torch.sum(K * K, dim=1, keepdim=True)
        DQ2 = torch.sum(Q * Q, dim=1, keepdim=True)
        DQK2 = DQ2 @ DK2.T
        DQK = torch.sqrt(DQK2).clamp_min(eps)

        Y = (Q @ K.T) / DQK

        SM = torch.softmax(Y.T, dim = 1)
            
        Z = SM @ input
        return Z
    

# Cosine Similarity Attention (between components/dimensions of input vectors - elements; attention matrix element-wise normalisation; linear Q and K transformations)
class cosPcTransformer(nn.Module):
    def __init__(self, in_size, actK = None, actQ = None):
        super(cosPcTransformer, self).__init__()

        # square K and Q matrixes
        self.in_size = in_size
        self.out_size = in_size

        self.actK = actK
        self.actQ = actQ
            
        # Initialize weight coefficients - glorot
        bound = math.sqrt(6 / (self.out_size + self.in_size))
        self.Wq = nn.Parameter(bound * (2. * torch.rand(self.out_size, self.in_size) - 1.))
        self.Wq0 = nn.Parameter(torch.zeros(self.out_size))

        self.Wk = nn.Parameter(bound * (2. * torch.rand(self.out_size, self.in_size) - 1.))
        self.Wk0 = nn.Parameter(torch.zeros(self.out_size))


    def forward(self, input):
        eps=1e-12
        if input.dim() == 1:
            input = input.unsqueeze(0)

        K = input @ self.Wk + self.Wk0
        if self.actK is not None:
            K = self.actK(K)

        Q = input @ self.Wq + self.Wq0
        if self.actQ is not None:
            Q = self.actQ(Q)

        DK2 = torch.sum(K.T * K.T, dim=1, keepdim=True)
        DQ2 = torch.sum(Q.T * Q.T, dim=1, keepdim=True)
        DQK2 = DQ2 @ DK2.T
        DQK = torch.sqrt(DQK2).clamp_min(eps)

        Y = (Q.T @ K) / DQK

        SM = torch.softmax(Y, dim = 0)
            
        Z = input @ SM
        return Z
    

# Cosine Similarity Attention (between components/dimensions of input vectors - elements; attention matrix element-wise normalisation; linear Q and K transformations)
class cosPcTransformerMH(nn.Module):
    def __init__(self, channels, in_size, actK = None, actQ = None):
        super(cosPcTransformerMH, self).__init__()

        # square K and Q matrixes
        self.in_size = in_size
        self.channels = channels
        self.out_size = in_size

        self.actK = actK
        self.actQ = actQ
            
        # Initialize weight coefficients - glorot
        self.Wq = nn.Parameter(torch.empty(self.channels, self.in_size, self.out_size))
        self.Wq0 = nn.Parameter(torch.zeros(self.channels, self.out_size))
        nn.init.xavier_uniform_(self.Wq)

        self.Wk = nn.Parameter(torch.empty(self.channels, self.in_size, self.out_size))
        self.Wk0 = nn.Parameter(torch.zeros(self.channels, self.out_size))
        nn.init.xavier_uniform_(self.Wk)


    def forward(self, input):
        eps=1e-12

        # K, Q: [batch, channels, out_size]
        Knb = torch.stack([
            torch.matmul(input[:, c, :], self.Wk[c, :, :])  # matmul for each channel
            for c in range(self.channels)
        ], dim=1)
        K = Knb + self.Wk0
        #K = torch.einsum('bci,cio->bco', input, self.Wk) + self.Wk0
        if self.actK is not None:
            K = self.actK(K)

        Qnb = torch.stack([
            torch.matmul(input[:, c, :], self.Wq[c, :, :])  # matmul for each channel
            for c in range(self.channels)
        ], dim=1)
        Q = Qnb + self.Wq0
        #Q = torch.einsum('bci,cio->bco', input, self.Wq) + self.Wq0
        if self.actQ is not None:
            Q = self.actQ(Q)

        Yu = torch.stack([
            torch.matmul(Q.T[:, c, :], K[:, c, :])  # matmul for each channel
            for c in range(self.channels)
        ], dim=1)
        #Yu = torch.einsum('bci,cio->bco', Q.T, K)
        DK2 = torch.sum(K.T * K.T, dim=2, keepdim=True)
        DQ2 = torch.sum(Q.T * Q.T, dim=2, keepdim=True)

        DQK2 = torch.stack([
            torch.matmul(DQ2[:, c, :], DK2.T[:, c, :])  # matmul for each channel
            for c in range(self.channels)
        ], dim=1)        
        #DQK2 = DQ2 @ DK2.T

        DQK = torch.sqrt(DQK2).clamp_min(eps)

        Y = Yu / DQK

        SM = torch.softmax(Y, dim = 0)

        Z = torch.stack([
            torch.matmul(input[:, c, :], SM[:, c, :])  # matmul for each channel
            for c in range(self.channels)
        ], dim=1)    

        
        return Z
    


class CustomBlock(nn.Module):
    """
    Implements the MATLAB block:
        K = Wk * X + Wk0
        Q = Wq * X + Wq0

        DK2 = sum(K'.*K', 1)
        DQ2 = sum(Q'.*Q', 1)
        DQK = sqrt(DQ2' * DK2)

        Y  = (Q * K') ./ DQK
        SM = softmax(Y, 'DataFormat','CB')
        Z  = (X' * SM)'
    """
    def __init__(self, C):
        super().__init__()
        self.Wk  = nn.Parameter(torch.empty(C, C))
        self.Wq  = nn.Parameter(torch.empty(C, C))
        self.Wk0 = nn.Parameter(torch.zeros(C, 1))
        self.Wq0 = nn.Parameter(torch.zeros(C, 1))
        # Xavier init for weights
        nn.init.xavier_uniform_(self.Wk)
        nn.init.xavier_uniform_(self.Wq)

    def forward(self, X, eps=1e-12):
        """
        X: [C,B] or [N,C,B]
        returns Z with same shape as X
        """
        if X.dim() == 2:
            K = self.Wk @ X + self.Wk0          # [C,B]
            Q = self.Wq @ X + self.Wq0          # [C,B]

            DK2 = (K**2).sum(dim=1)             # [C]
            DQ2 = (Q**2).sum(dim=1)             # [C]

            DQK = torch.sqrt((DQ2[:, None] * DK2[None, :]).clamp_min(eps))  # [C,C]
            Y = (Q @ K.t()) / DQK               # [C,C]

            SM = F.softmax(Y, dim=0)            # 'CB' => softmax over rows (C)
            Z = (X.t() @ SM).t()                # [C,B]
            return Z

        elif X.dim() == 3:
            # [N,C,B] batched path
            K = torch.matmul(self.Wk, X) + self.Wk0    # [N,C,B]
            Q = torch.matmul(self.Wq, X) + self.Wq0    # [N,C,B]

            DK2 = (K**2).sum(dim=2)                    # [N,C]
            DQ2 = (Q**2).sum(dim=2)                    # [N,C]

            DQK = torch.sqrt((DQ2[:, :, None] * DK2[:, None, :]).clamp_min(eps))  # [N,C,C]
            Y = torch.matmul(Q, K.transpose(1, 2)) / DQK                           # [N,C,C]

            SM = F.softmax(Y, dim=1)                   # softmax over C (rows)
            Z = torch.matmul(X.transpose(1, 2), SM).transpose(1, 2)                # [N,C,B]
            return Z

        else:
            raise ValueError("X must be 2D [C,B] or 3D [N,C,B].")



class MatlabBlock(nn.Module):
    """
    Implements:
        K = Wk * X + Wk0
        Q = Wq * X + Wq0

        DK2 = sum(K .* K, 1)
        DQ2 = sum(Q .* Q, 1)
        DQK = sqrt((DQ2' * DK2))

        Y  = (Q' * K) ./ DQK
        SM = softmax(Y', 'DataFormat','CB')  # => column-wise softmax on Y
        Z  = X * SM
    """
    def __init__(self, C):
        super().__init__()
        self.Wk  = nn.Parameter(torch.empty(C, C))
        self.Wq  = nn.Parameter(torch.empty(C, C))
        self.Wk0 = nn.Parameter(torch.zeros(C, 1))
        self.Wq0 = nn.Parameter(torch.zeros(C, 1))
        nn.init.xavier_uniform_(self.Wk)
        nn.init.xavier_uniform_(self.Wq)

    def forward(self, X, eps=1e-12):
        if X.dim() == 2:
            # [C, B]
            K = self.Wk @ X + self.Wk0    # [C, B]
            Q = self.Wq @ X + self.Wq0    # [C, B]

            DK2 = (K * K).sum(dim=0)      # [B]
            DQ2 = (Q * Q).sum(dim=0)      # [B]

            DQK = torch.sqrt((DQ2[:, None] * DK2[None, :]).clamp_min(eps))  # [B, B]
            Y = (Q.t() @ K) / DQK         # [B, B]

            SM = F.softmax(Y, dim=0)      # column-wise softmax (matches softmax(Y', 'CB'))
            Z = X @ SM                    # [C, B]
            return Z

        elif X.dim() == 3:
            # [N, C, B]
            K = torch.matmul(self.Wk, X) + self.Wk0    # [N, C, B]
            Q = torch.matmul(self.Wq, X) + self.Wq0    # [N, C, B]

            DK2 = (K * K).sum(dim=1)                    # [N, B]
            DQ2 = (Q * Q).sum(dim=1)                    # [N, B]

            DQK = torch.sqrt((DQ2[:, :, None] * DK2[:, None, :]).clamp_min(eps))  # [N, B, B]
            Y = torch.matmul(Q.transpose(1, 2), K) / DQK                         # [N, B, B]

            SM = F.softmax(Y, dim=1)                    # column-wise softmax
            Z = torch.matmul(X, SM)                     # [N, C, B]
            return Z

        else:
            raise ValueError("X must be 2D [C,B] or 3D [N,C,B].")



class MatlabBlockNCB(nn.Module):
    """
    PyTorch module for the MATLAB block with layout [N, C, B]:

        K = Wk * X + Wk0
        Q = Wq * X + Wq0
        DK2 = sum(K .* K, 1)      # sum over channels -> [N,B]
        DQ2 = sum(Q .* Q, 1)
        DQK = sqrt(DQ2' * DK2)    # outer product per batch -> [N,B,B]
        Y  = (Q' * K) ./ DQK      # [N,B,B]
        SM = softmax(Y', 'CB')    # column-wise softmax on Y
        Z  = X * SM               # [N,C,B]
    """
    def __init__(self, C):
        super().__init__()
        self.Wk  = nn.Parameter(torch.empty(C, C))
        self.Wq  = nn.Parameter(torch.empty(C, C))
        self.Wk0 = nn.Parameter(torch.zeros(C, 1))
        self.Wq0 = nn.Parameter(torch.zeros(C, 1))
        nn.init.xavier_uniform_(self.Wk)
        nn.init.xavier_uniform_(self.Wq)

    def forward(self, X, eps=1e-12):
        # X: [N, C, B]
        K = torch.matmul(self.Wk, X) + self.Wk0      # [N, C, B]
        Q = torch.matmul(self.Wq, X) + self.Wq0      # [N, C, B]

        DK2 = (K * K).sum(dim=1)                     # [N, B]
        DQ2 = (Q * Q).sum(dim=1)                     # [N, B]

        DQK = torch.sqrt((DQ2[:, :, None] * DK2[:, None, :]).clamp_min(eps))  # [N, B, B]
        Y = torch.matmul(Q.transpose(1, 2), K) / DQK                          # [N, B, B]

        SM = F.softmax(Y, dim=1)                     # column-wise softmax
        Z = torch.matmul(X, SM)                      # [N, C, B]
        return Z
    

class MatlabBlockNCB_Linear(nn.Module):
    def __init__(self, C):
        super().__init__()
        # Linear across channels, shared over all positions B
        self.Wk = nn.Linear(C, C, bias=True)
        self.Wq = nn.Linear(C, C, bias=True)

        # align parameter shapes to [C,C] and [C,1]-style usage
        # We'll use einsum to apply them on dim=1 (channels)
        nn.init.xavier_uniform_(self.Wk.weight)
        nn.init.xavier_uniform_(self.Wq.weight)

    def forward(self, X, eps=1e-12):
        # X: [N, C, B]
        # Apply linear across channel dim for each position: (N,C,B) -> (N,C,B)
        # Using einsum to match (B positions share the same linear map)
        K = torch.einsum('oc,ncb->nob', self.Wk.weight, X) + self.Wk.bias[:, None]
        Q = torch.einsum('oc,ncb->nob', self.Wq.weight, X) + self.Wq.bias[:, None]

        DK2 = (K * K).sum(dim=1)                                   # [N, B]
        DQ2 = (Q * Q).sum(dim=1)                                   # [N, B]
        DQK = torch.sqrt((DQ2[:, :, None] * DK2[:, None, :]).clamp_min(eps))  # [N,B,B]
        Y = torch.matmul(Q.transpose(1, 2), K) / DQK               # [N, B, B]
        SM = torch.softmax(Y, dim=1)                               # [N, B, B]
        Z = torch.matmul(X, SM)                                    # [N, C, B]
        return Z
    
# Create a random tensor of shape (2, 3, 4)
# tensor = torch.rand(2, 3, 4)


class MatlabBlockNB(nn.Module):
    """
    PyTorch module for the MATLAB block with layout [N, E]:

        K = X @ Wk^T + Wk0
        Q = X @ Wq^T + Wq0
        DK2 = sum(K .* K, 2)   -> [N]
        DQ2 = sum(Q .* Q, 2)   -> [N]
        DQK = sqrt(DQ2 ⊗ DK2)  -> [N, N]
        Y  = (Q @ K^T) / DQK   -> [N, N]
        SM = column-wise softmax(Y)
        Z  = SM @ X            -> [N, E]
    """
    def __init__(self, E):
        super().__init__()
        # Parameters equivalent to [E,E] weight and [E] bias
        self.Wk = nn.Parameter(torch.empty(E, E))
        self.Wq = nn.Parameter(torch.empty(E, E))
        self.Wk0 = nn.Parameter(torch.zeros(E))
        self.Wq0 = nn.Parameter(torch.zeros(E))

        nn.init.xavier_uniform_(self.Wk)
        nn.init.xavier_uniform_(self.Wq)

    def forward(self, X, eps=1e-12):
        # X: [N, E]
        K = X @ self.Wk.T + self.Wk0           # [N, E]
        Q = X @ self.Wq.T + self.Wq0           # [N, E]

        DK2 = (K * K).sum(dim=1)               # [N]
        DQ2 = (Q * Q).sum(dim=1)               # [N]

        DQK = torch.sqrt((DQ2[:, None] * DK2[None, :]).clamp_min(eps))  # [N, N]
        Y = (Q @ K.T) / DQK                    # [N, N]

        SM = F.softmax(Y, dim=0)               # column-wise softmax on Y
        Z = SM @ X                             # [N, E]
        return Z
