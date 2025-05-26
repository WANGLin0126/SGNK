'''
Tuning the hyper-parameters using grid search.
python search.py --dataset MUTAG --kernel SGNK --K 3 --L 1
'''

import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
import argparse
import pandas as pd

def search(dataset, data_dir, kernel, K, L):
    path = data_dir+'/'+dataset+'-K-'+K+'-L-'+L+'/'
    gram = np.load( path + dataset+'_'+kernel+'_gram.npy')
    if gram.min() != 0:
        gram /= gram.min()

    # gram = F.normalize(torch.tensor(gram))

    labels = np.load(path + dataset+'_'+kernel+'_labels.npy')
    
    train_fold_idx = [np.loadtxt('dataset/{}/10fold_idx/train_idx-{}.txt'.format(
        dataset, i)).astype(int) for i in range(1, 11)]
    test_fold_idx = [np.loadtxt('dataset/{}/10fold_idx/test_idx-{}.txt'.format(
        dataset, i)).astype(int) for i in range(1, 11)]

    C_list = np.logspace(-2, 4, 120)
    svc = SVC(kernel='precomputed', cache_size=16000, max_iter=5e5)
    clf = GridSearchCV(svc, {'C' : C_list}, 
                cv=zip(train_fold_idx, test_fold_idx),
                n_jobs=80, verbose=0, return_train_score=True)
    clf.fit(gram, labels)
    df = pd.DataFrame({'C': C_list, 
                       'train': clf.cv_results_['mean_train_score'], 
                       'test': clf.cv_results_['mean_test_score'], 
                       'std': clf.cv_results_['std_test_score']},
                        columns=['C', 'train', 'test','std'])

    # also normalized gram matrix 
    gram_nor = np.copy(gram)
    gram_diag = np.sqrt(np.diag(gram_nor))
    gram_nor /= gram_diag[:, None]
    gram_nor /= gram_diag[None, :]

    svc = SVC(kernel='precomputed', cache_size=16000, max_iter=5e5)
    clf = GridSearchCV(svc, {'C' : C_list},
                cv=zip(train_fold_idx, test_fold_idx),
                n_jobs=80, verbose=0, return_train_score=True)
    clf.fit(gram_nor, labels)
    df_nor = pd.DataFrame({'C': C_list,
        'train': clf.cv_results_['mean_train_score'],
        'test': clf.cv_results_['mean_test_score'],
        'std': clf.cv_results_['std_test_score']},
        columns=['C', 'train', 'test','std'])

    df['normalized'] = False
    df_nor['normalized'] = True
    all_df = pd.concat([df, df_nor])[['C', 'normalized', 'train', 'test','std']]
    all_df.to_csv(path+dataset+'_'+kernel+'_grid_search.csv')
    # print(max(all_df['test']))
    max_index = all_df['test'].idxmax()
    print(kernel)
    print(all_df.loc[max_index])
    
parser = argparse.ArgumentParser(description='hyper-parameter search')
parser.add_argument('--data_dir', type=str, default='./outputs',  help='data_dir')
parser.add_argument('--dataset', type=str, default='MUTAG',  help='dataset')
parser.add_argument('--kernel', type=str, default='SGNK',  help='kernel')
parser.add_argument('--K', type=str, default='3',  help='number of aggregation times')
parser.add_argument('--L', type=str, default='1',  help='number of update times')
args = parser.parse_args()

search(args.dataset, args.data_dir, args.kernel, args.K, args.L)