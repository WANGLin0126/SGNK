import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
import torch
from sgtk import SimplifiedGraphNeuralTangentKernel
from sgnk import SimplifiedGraphNeuralKernel
from LoadData import load_data
import argparse
import numpy as np
import time
from sklearn.svm import SVC
import argparse
import pandas as pd

device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

def GCF(adj, x, k=1):
    """
    Graph convolution filter
    parameters:
        adj: torch.Tensor, adjacency matrix, must be self-looped
        x: torch.Tensor, features
        k: int, number of hops
    return:
        torch.Tensor, filtered features
    """
    D = torch.sum(adj,dim=1)
    D = torch.pow(D,-0.5)
    D = torch.diag(D)

    filter = torch.matmul(torch.matmul(D,adj),D)
    for i in range(k):
        x = torch.matmul(filter,x)
    return x

def classification(C_list, K, labels, idx_train, idx_val, idx_test):
    K_train = K[np.ix_(idx_train, idx_train)]
    K_test = K[np.ix_(idx_test, idx_train)]
    K_val = K[np.ix_(idx_val, idx_train)]
    ACC_val = []
    ACC_test = []
    for C in C_list:
        svm = SVC(kernel='precomputed', probability=True, C=C, tol=1e-5, max_iter=5000).fit(K_train.to('cpu').numpy(), labels[idx_train].numpy())
        y_pred_val = svm.predict(K_val.to('cpu').numpy())
        y_pred_test = svm.predict(K_test.to('cpu').numpy())
        acc_val = np.sum(y_pred_val == labels[idx_val].numpy())/y_pred_val.size
        acc_test = np.sum(y_pred_test == labels[idx_test].numpy())/y_pred_test.size
        ACC_val.append(acc_val)
        ACC_test.append(acc_test)

    return ACC_val, ACC_test

parser = argparse.ArgumentParser(description='Kernel Computation')
parser.add_argument('--dataset', type=str, default="Cora", help='name of dataset [Cora, Citeseer, Pubmed, Photo, Computers] (default: Cora)')
parser.add_argument('--K', type=int, default=3, help='number of aggr in kernel method (default: 1)')
parser.add_argument('--L', type=int, default=1, help='the number of layers after each aggr (default: 1)')
parser.add_argument('--kernel', type=str, default='SGNK', help='kernel method in KRR [SGTK, SGNK] (default: SGNK)')
args = parser.parse_args()

# load dataset
root = './datasets/'
adj, x, labels, train_mask, val_mask, test_mask,  \
                        _, _, _, \
                        y_train, y_val, y_test, \
                        y_train_one_hot, y_val_one_hot, y_test_one_hot, _, idx_train, idx_val, idx_test= load_data(root=root, name=args.dataset)

n_class    = len(torch.unique(labels))
n,dim      = x.shape

print(f"Dataset       :{args.dataset}")

C_list = np.logspace(-3, 3, 100)

df      = pd.DataFrame(columns=['C', 'K', 'Validation', 'Test'])
df_nor  = pd.DataFrame(columns=['C', 'K', 'Validation', 'Test'])

for k in range(args.K):
    k = k + 1

    print(f"pow: {k}, kernel: {args.kernel}", end = " ")

    if args.kernel == 'SGTK':
        SGTK        = SimplifiedGraphNeuralTangentKernel( L=args.L).to(device)
        kernel = SGTK.nodes_gram
        x = GCF(adj, x, k)
    elif args.kernel == 'SGNK':
        SGNK       = SimplifiedGraphNeuralKernel( L=args.L).to(device)
        kernel = SGNK.nodes_gram
        x = GCF(adj, x, k)
    else:
        raise ValueError('Kernel not found!')

    K      = kernel(x, x)
    
    a = time.time()
    acc_val, acc_test = classification(C_list, K, labels, idx_train, idx_val, idx_test)
    b = time.time()
    print(f"Search time: {b-a:.2f}s")
    k_list = [k]*len(C_list)

    new_rows = pd.DataFrame({'C': C_list,
                        'K': k_list,
                        'Validation': acc_val,
                        'Test': acc_test,},
                        columns=['C','K', 'Validation', 'Test'])
    
    df = pd.concat([df, new_rows],ignore_index=True)

    # also normalized gram matrix
    K_nor = np.copy(K)
    K_diag = np.sqrt(np.diag(K_nor))
    K_nor /= K_diag[:, None]
    K_nor /= K_diag[None, :]
    K_nor = torch.tensor(K_nor).to(device)

    acc_val, acc_test = classification(C_list, K_nor, labels, idx_train, idx_val, idx_test)


    new_rows = pd.DataFrame({'C': C_list,
                        'K': k_list,
                        'Validation': acc_val,
                        'Test': acc_test,},
                        columns=['C','K', 'Validation', 'Test'])
    
    df_nor = pd.concat([df_nor, new_rows], ignore_index=True)

df['Normalized'] = False
df_nor['Normalized'] = True

all_df = pd.concat([df, df_nor], ignore_index=True)
all_df.to_csv( './outputs/'+args.dataset +'_' + args.kernel + '_search.csv')

rows = []
for k in range(args.K):
    k = k + 1
    for norm in [True, False]:
        max_index = all_df[(all_df['K'] == k) & (all_df['Normalized'] == norm)]['Test'].idxmax()
        rows.append(max_index)

print(all_df.loc[rows])

print(f"{args.dataset}",end = " ")