"""
nodes classification test by using the KRR model and the proposed SGNK and SGTK kernels
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
import torch
from torch import nn
from sgtk import SimplifiedGraphNeuralTangentKernel
from sgnk import SimplifiedGraphNeuralKernel
from LoadData import load_data
from utils import sub_E
import argparse
import numpy as np

device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

parser = argparse.ArgumentParser(description='kernel computation')
parser.add_argument('--dataset', type=str, default="Cora", help='name of dataset [Cora, Citeseer, Pubmed, Photo, Computers] (default: Cora)')
parser.add_argument('--K', type=int, default=3, help='number of aggr in kernel method (default: 3)')
parser.add_argument('--L', type=int, default=1, help='the number of layers after each aggr (default: 1)')
parser.add_argument('--kernel', type=str, default='SGNK', help='kernel method in KRR [SGTK, SGNK] (default: SGTK)')
args = parser.parse_args()

def KRR(G_t, G_s, y_t, y_s, kernel, ridge):
    K_ss      = kernel(G_s, G_s)
    K_ts      = kernel(G_t, G_s)
    n        = torch.tensor(len(G_s), device = G_s.device)
    regulizer = ridge * torch.trace(K_ss) * torch.eye(n, device = G_s.device) / n
    b         = torch.linalg.solve(K_ss + regulizer, y_s)
    pred      = torch.matmul(K_ts, b)

    pred      = nn.functional.softmax(pred, dim = 1)
    correct   = torch.eq(pred.argmax(1).to(torch.float32), y_t.argmax(1).to(torch.float32)).sum().item()
    acc       = correct / len(y_t)

    return pred, acc

def GCF(adj, x, K=1):
    """
    Graph convolution filter
    parameters:
        adj: torch.Tensor, adjacency matrix, must be self-looped
        x: torch.Tensor, features
        K: int, number of hops
    return:
        torch.Tensor, filtered features
    """
    D = torch.sum(adj,dim=1)
    D = torch.pow(D,-0.5)
    D = torch.diag(D)

    filter = torch.matmul(torch.matmul(D,adj),D)
    for i in range(K):
        x = torch.matmul(filter,x)
    return x

if args.kernel == 'SGTK':
    SGTK        = SimplifiedGraphNeuralTangentKernel( L=args.L).to(device)
    kernel = SGTK.nodes_gram
elif args.kernel == 'SGNK':
    SGNK       = SimplifiedGraphNeuralKernel().to(device)
    kernel = SGNK.nodes_gram
else:
    raise ValueError('Kernel not found!')

# load dataset
root = './datasets/'
adj, x, labels, train_mask, _, test_mask,  \
                        x_train, _, x_test, \
                        y_train, _, y_test, \
                        y_train_one_hot, _, y_test_one_hot, _, idx_train, idx_val, idx_test= load_data(root=root, name=args.dataset)

n_class    = len(torch.unique(labels))
n,dim      = x.shape

x = GCF(adj, x, args.K)

print(f"Dataset       :{args.dataset}")
print(f"Training Set  :{len(y_train)}")
print(f"Testing Set   :{len(y_test)}")
print(f"Classes       :{n_class}")
print(f"Dim           :{dim}")
print(f"K             :{args.K}")
print(f"L             :{args.L}")
print(f"Kernel        :{args.kernel}")

x_test = x[test_mask].to(device)
x_train = x[train_mask].to(device)
y_test = y_test.to(device)
y_train = y_train.to(device)
y_train_one_hot = y_train_one_hot.to(torch.float32).to(device)
y_test_one_hot = y_test_one_hot.to(torch.float32).to(device)
A_train = sub_E(idx_train, adj).to(device)
A_test  = sub_E(idx_test, adj).to(device)

ridge_list = np.logspace(-2, 2, 120)
ridge_list = torch.tensor(ridge_list).to(device)


Acc_list = []
for i in range(len(ridge_list)):
    ridge = ridge_list[i]
    _,acc = KRR(x_test, x_train, y_test_one_hot, y_train_one_hot, kernel, ridge)
    Acc_list.append(acc)
    
max_acc, idx = torch.max(torch.tensor(Acc_list), dim=0)

print(f"Rdige         :{ridge_list[idx]:.4f}")
print(f"Accuracy      :{max_acc:.4f}")
