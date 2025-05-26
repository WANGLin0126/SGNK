import torch
import torch.nn as nn

class SimplifiedGraphNeuralKernel(nn.Module):
    def __init__(self):
        super(SimplifiedGraphNeuralKernel, self).__init__()

    def nodes_gram(self, x, X):
        
        Sigma_xX    = torch.matmul(x, X.t())
        Sigma_xx    = torch.diag(torch.matmul(x, x.t())).reshape(-1,1)
        Sigma_XX    = torch.diag(torch.matmul(X, X.t())).reshape(1,-1)
        
        a_xX  = torch.clip(2 * Sigma_xX/(torch.sqrt(1 + 2*Sigma_xx)*torch.sqrt(1 + 2*Sigma_XX)),-1,1)
        Sigma_xX = 2/torch.pi * torch.asin(a_xX)
        
        return Sigma_xX
