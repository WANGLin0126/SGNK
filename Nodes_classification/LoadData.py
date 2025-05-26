import torch
from torch_geometric.datasets import Planetoid, Amazon
from torch.nn import functional as F
import scipy.sparse as sp
from sklearn.preprocessing import normalize

def load_data(root,name):
    if name in ['Cora','Citeseer','Pubmed','CS','Physics']:
        dataset    = Planetoid(root=root,name=name,split='public')
        train_mask = dataset[0]['train_mask']
        val_mask   = dataset[0]['val_mask']
        test_mask  = dataset[0]['test_mask']
        x          = dataset[0]['x']           # all features
        y          = dataset[0]['y']           # all labels
    elif name in ['Computers','Photo']:
        dataset    = Amazon(root=root,name=name)
        x          = dataset[0]['x']           # all features
        y          = dataset[0]['y']           # all labels
        n_class    = len(torch.unique(y))
        n,_        = x.shape
        idx_train = []
        idx_test  = []
        for i in range(n_class):
            idx = torch.where(y==i)[0]
            idx_train.append(idx[:20])
            idx_test.append(idx[20:120])
        idx_train = torch.cat(idx_train)
        idx_test  = torch.cat(idx_test)
        train_mask = torch.zeros(n,dtype=torch.bool)
        test_mask  = torch.zeros(n,dtype=torch.bool)
        train_mask[idx_train] = True
        test_mask[idx_test]   = True
        val_mask = ~ (train_mask | test_mask)
    else:
        raise ValueError('Dataset not found!')

    x         = torch.tensor(normalize(x)).to(torch.float32)

    edge_index = dataset[0]['edge_index']
    n_class    = len(torch.unique(y))
    n,_      = x.shape

    adj = sp.coo_matrix((torch.ones(edge_index.shape[1]), edge_index), shape=(n, n)).toarray()
    adj = torch.tensor(adj)
    adj = adj + torch.eye(adj.shape[0])  

    x_train    = x[train_mask]
    x_val      = x[val_mask]
    x_test     = x[test_mask]

    y_train    = y[train_mask]
    y_val      = y[val_mask]
    y_test     = y[test_mask]

    idx_train = torch.where(train_mask)[0]
    idx_val   = torch.where(val_mask)[0]
    idx_test  = torch.where(test_mask)[0]

    y_one_hot       = F.one_hot(y, n_class)
    y_train_one_hot = y_one_hot[train_mask]
    y_val_one_hot   = y_one_hot[val_mask]
    y_test_one_hot  = y_one_hot[test_mask]

    return adj, x, y, train_mask, val_mask, test_mask,  \
                        x_train, x_val, x_test, \
                        y_train, y_val, y_test, \
                        y_train_one_hot, y_val_one_hot, y_test_one_hot, y_one_hot, idx_train, idx_val, idx_test