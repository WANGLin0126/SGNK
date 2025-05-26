import torch
import torch.nn as nn

class SimplifiedGraphNeuralKernel(nn.Module):
    def __init__(self, K=3 ):
        super(SimplifiedGraphNeuralKernel, self).__init__()
        self.K = K
    
    def GCF(self, x, A, k=1):
        """
        Graph Convolutional Filtering
        """
        A = torch.clip(A + torch.eye(A.shape[0]).to(A.device),0,1)
        D = torch.diag(torch.sum(A, 1))
        D = torch.inverse(torch.sqrt(D))
        A = torch.matmul(torch.matmul(D, A), D)
        A = torch.matrix_power(A, k)
        x = torch.matmul(A, x)
        return x

    def similarity(self, g1, g2, A1, A2):
        x, X = g1.node_features, g2.node_features
        A1 ,A2 = A1.to_dense(), A2.to_dense()
        x, X = self.GCF(x, A1, self.K), self.GCF(X, A2, self.K)

        Sigma_xX    = torch.matmul(x, X.t())
        Sigma_xx    = torch.diag(torch.matmul(x, x.t())).reshape(-1,1)
        Sigma_XX    = torch.diag(torch.matmul(X, X.t())).reshape(1,-1)
        
        a_xX  = torch.clip(2 * Sigma_xX/(torch.sqrt(1 + 2*Sigma_xx)*torch.sqrt(1 + 2*Sigma_XX)),-1,1)
        Sigma_xX = 2/torch.pi * torch.asin(a_xX)

        graph_kernel = torch.sum(Sigma_xX)
        
        return graph_kernel
