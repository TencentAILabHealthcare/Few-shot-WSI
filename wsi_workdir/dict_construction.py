import numpy as np
from openselfsup.third_party import clustering
from scipy.spatial.distance import cdist
import os
import warnings
import time
import pickle as pkl
from sklearn.neighbors import KNeighborsClassifier
from scipy.special import softmax
import argparse

Kmeans = clustering.__dict__['Kmeans']

pth = 'wsi_workdir/workdir/extracted_feats'
dict_pth = 'wsi_workdir/workdir/dict'

def main(args):
    model = args.model
    # number of prototypes
    num_prototypes = args.num_prototypes
    # number of shift vector for each prototypes
    num_shift_vectors = args.num_shift_vectors
    # loading features
    features = np.load(f'{pth}/{model}/{args.features}.npy', 'r')
    labels = np.load('wsi_workdir/workdir/extracted_feats/NCT_train_labels.npy', 'r')
    print(f'{model} features loaded.')

    if args.novel_class != None:
        if args.novel_class == 78:
            features = features[(labels!=7) * (labels!=8) ]
        else:
            features = features[labels!=args.novel_class]

    os.makedirs(f'{dict_pth}/{model}',exist_ok=True)
    print(f'using {len(features)} features to cluster...')
    kmeans = Kmeans(k=num_prototypes, pca_dim=-1)
    kmeans.cluster(features, seed=66)
    assignments = kmeans.labels.astype(np.int64)

    # compute the prototype for each cluster
    prototypes = np.array([np.mean(features[assignments==i],axis=0)
                            for i in range(num_prototypes)])

    # compute covariance matrix for each cluster
    covariance = np.array([np.cov(features[assignments==i].T) 
                            for i in range(num_prototypes)])

    # save the legacy dict : {prototype: covariance}
    np.save(f'{dict_pth}/{model}/NCT_PROTO_BANK_{num_prototypes}.npy', prototypes)
    np.save(f'{dict_pth}/{model}/NCT_COV_BANK_{num_prototypes}.npy', covariance)

    # generate shift vector bank.
    SHIFT_BANK = []
    for cov in covariance:
        SHIFT_BANK.append(
            # sample shift vector from zero-mean multivariate Gaussian distritbuion N(0, cov)
            np.random.multivariate_normal(np.zeros(cov.shape[0]),
                                          cov, 
                                          size=num_shift_vectors))

    SHIFT_BANK = np.array(SHIFT_BANK)
    # save the shift bank
    np.save(f'{dict_pth}/{model}/NCT_SHIFT_BANK_{num_prototypes}.npy', SHIFT_BANK)
    print('legacy dict constructed', f'saving to {dict_pth}/{model}/NCT_SHIFT_BANK_{num_prototypes}.npy')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='legacy dictionary construction')
    parser.add_argument('--model', type=str, required=True, help='model name')
    parser.add_argument('--features', type=str, required=False, default='NCT_train', help='features file name')
    parser.add_argument('--novel_class', required=False, type=int, default=None, help='excluding which class')
    parser.add_argument('--num_prototypes', type=int, default=16)
    parser.add_argument('--num_shift_vectors', type=int, default=2000)
    args = parser.parse_args()
    main(args)
