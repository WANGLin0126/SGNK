import torch
import math
import torch.nn as nn

class SimplifiedGraphNeuralTangentKernel(nn.Module):
    def __init__(self,K=2, L=1):
        super(SimplifiedGraphNeuralTangentKernel, self).__init__()
        self.K = K
        self.L = L
    
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
    
    def update_sigma(self, S, diag1, diag2):
        S    = S / diag1[:, None] / diag2[None, :]
        S    = torch.clip(S,-0.9999,0.9999)
        S    = (S * (math.pi - torch.arccos(S)) + torch.sqrt(1 - S * S)) / math.pi
        degree_sigma   = (math.pi - torch.arccos(S)) / math.pi
        S    = S * diag1[:, None] * diag2[None, :]
        return S, degree_sigma
    
    def update_diag(self, S):
        diag = torch.sqrt(torch.diag(S))
        S    = S / diag[:, None] / diag[None, :]
        S    = torch.clip(S,-0.9999,0.9999)
        S  = (S * (math.pi - torch.arccos(S)) + torch.sqrt(1 - S * S)) / math.pi
        S    = S * diag[:, None] * diag[None, :]
        return S, diag
    
    def diag(self, g, A):
        diag_list = []
        sigma = torch.matmul(g, g.t())

        for l in range(self.L):
            sigma, diag = self.update_diag(sigma)
            diag_list.append(diag)
        return diag_list

    def similarity(self, g1, g2, A1, A2):   
        x, X = g1.node_features, g2.node_features
        A1 ,A2 = A1.to_dense(), A2.to_dense()
        x, X = self.GCF(x, A1, self.K), self.GCF(X, A2, self.K)

        sigma = torch.matmul(x, X.t())
        theta = sigma
        diag_list1, diag_list2 = self.diag(x, A1), self.diag(X, A2)

        for l in range(self.L):
            sigma, degree_sigma = self.update_sigma(sigma, diag_list1[l], diag_list2[l])
            theta = theta * degree_sigma + sigma

        return sum(sum(theta))
