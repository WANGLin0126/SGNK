"""
generate the kernel matrix for the dataset
python gram.py --dataset MUTAG --kernel SGNK --K 2
"""

import util
import time
import numpy as np
import argparse
import os
from sgtk import SimplifiedGraphNeuralTangentKernel
from sgnk import SimplifiedGraphNeuralKernel
import torch
os.environ['CUDA_LAUNCH_BLOCKING'] = '0,1'

from tqdm import tqdm
import time

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser(description='kernel computation')
parser.add_argument('--dataset', type=str, default="MUTAG", help='name of dataset (default: MUTAG)')
parser.add_argument('--L', type=int, default=1, help='the number of layers after each aggr (default: 1)')
parser.add_argument('--K', type=int, default=3, help='number of layers')
parser.add_argument('--kernel', type=str, default='SGTK', help='kernel type, [SGTK,SGNK]')
args = parser.parse_args()

if args.dataset in ['IMDBBINARY', 'IMDBMULTI']:
    # social network
    degree_as_tag = True
elif args.dataset in ['MUTAG', 'PTC']:
    # bioinformatics
    degree_as_tag = False
    
graphs, _  = util.load_data(args.dataset, degree_as_tag)
labels = np.array([g.label for g in graphs]).astype(int)

SGTK = SimplifiedGraphNeuralTangentKernel(K=args.K, L=args.L)
SGNK = SimplifiedGraphNeuralKernel(K=args.K)

A_list = []
diag_list = []
diag_list2 = []

# procesing the data for kernel calculation
for i in range(len(graphs)):
    n = len(graphs[i].neighbors)
    for j in range(n):
        graphs[i].neighbors[j].append(j)
    edges = graphs[i].g.edges
    m = len(edges)

    row = [e[0] for e in edges]
    col = [e[1] for e in edges]
    A = torch.sparse_coo_tensor(torch.tensor([row, col]),torch.tensor([1] * len(edges)),size=(n, n)).to(torch.float32)

    A_list.append(A.to(graphs[0].node_features.device))

    shape = torch.Size([n, n])
    indices = torch.arange(0, n).unsqueeze(0).repeat(2, 1)
    values = torch.ones(n)
    sparse_eye = torch.sparse_coo_tensor(indices, values, shape).to(graphs[0].node_features.device)

    A_list[-1] = A_list[-1] + A_list[-1].t() + sparse_eye


if args.kernel == 'SGTK':
    def calc(T):
        return SGTK.similarity(graphs[T[0]], graphs[T[1]], A_list[T[0]], A_list[T[1] ])
elif args.kernel == 'SGNK':
    def calc(T):
        return SGNK.similarity(graphs[T[0]], graphs[T[1]], A_list[T[0]], A_list[T[1]])


calc_list = [(i, j) for i in range(len(graphs)) for j in range(i, len(graphs))]

print(f"# K          : {args.K}")
print(f"----------Calculating {args.kernel} kernel matrix----------")
results = []

Time = 0.
for T in tqdm(calc_list):
    a = time.time()
    r = calc(T).item()
    b = time.time()
    Time = Time + b - a
    results.append(r)


print(f" {args.kernel} Time: {Time}")

gram = torch.zeros((len(graphs), len(graphs)))
for t, v in zip(calc_list, results):
    gram[t[0], t[1]] = v
    gram[t[1], t[0]] = v
    
args.path =  './outputs/' + args.dataset+ '-K-' + str(args.K) + '-L-' + str(args.L)

if not os.path.exists(args.path):
    os.makedirs(args.path)
    
np.save(args.path+'/'+ args.dataset+'_'+args.kernel+'_gram', gram)
np.save(args.path+'/'+ args.dataset+'_'+args.kernel+'_labels', labels)
 