import argparse
import datetime
import scipy
import numpy as np
from scipy.stats import t
from sklearn.linear_model import RidgeClassifier, LogisticRegression
from sklearn.neighbors import NearestCentroid
from sklearn.metrics import f1_score
from scipy.spatial.distance import cdist
from tqdm.contrib.concurrent import process_map

def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * t._ppf((1+confidence)/2., n-1)
    return m, h

def aug_base_samples(features, labels, num_aug_shots=50):
    samples, gt_labels = [], []
    # for each class
    for label in np.unique(labels):
        selected_samples = features[labels==label]
        # find the most closest prototypes
        proto_id = np.argmin(cdist(selected_samples, PROTO_BANK), axis=1)
        generated_samples = []
        for ix, sample in zip(proto_id, selected_samples):
            generated_samples.append(
                np.concatenate([
                    [sample], 
                    # generate new latent augmented samples by z' = z + delta
                    # delta is sampled from pre-generated shift bank, indexed by prototype id. 
                    sample[np.newaxis, :] + SHIFT_BANK[ix][np.random.choice(NUM_SHIFTs,num_aug_shots)] 
                ])
            )
        samples.append(np.concatenate(generated_samples, axis=0))
        gt_labels.extend([label]*len(samples[-1]))
    return np.concatenate(samples, axis=0), gt_labels


def meta_testing(task_pth):
    task_dataset = np.load(task_pth, allow_pickle=True)
    support_xs, support_ys, query_xs, query_ys = task_dataset
    if len(np.shape(support_xs)) == 3:
        support_xs = np.reshape(support_xs[:,:args.aug_times,:], (-1, 512))
        support_ys = np.repeat(support_ys, args.aug_times)
    if args.clf == 'Ridge':
        clf = RidgeClassifier()
    elif args.clf == 'logistic_regression':
        clf = LogisticRegression(max_iter=1000)
    elif args.clf == 'nearest_centroid':
        clf = NearestCentroid()
    else:
        raise NotImplementedError
    clf.fit(support_xs, support_ys)
    y_pred = clf.predict(query_xs)
    return (query_ys, y_pred)

def meta_testing_LatentAug(task_pth):
    task_dataset = np.load(task_pth, allow_pickle=True)
    support_xs, support_ys, query_xs, query_ys = task_dataset
    if len(np.shape(support_xs)) == 3:
        support_xs = np.reshape(support_xs[:,:args.aug_times], (-1, 512))
        support_ys = np.repeat(support_ys, args.aug_times)
    support_xs, support_ys = aug_base_samples(support_xs, support_ys, num_aug_shots=args.num_aug_shots-1)
    if args.clf == 'Ridge':
        clf = RidgeClassifier()
    elif args.clf == 'logistic_regression':
        clf = LogisticRegression(max_iter=1000)
    elif args.clf == 'nearest_centroid':
        clf = NearestCentroid()
    else:
        raise NotImplementedError
    clf.fit(support_xs, support_ys)
    y_pred = clf.predict(query_xs)
    return (query_ys, y_pred)

def evaluate(y_trues, y_preds):
    f1s = []
    for y_true, y_pred in zip(y_trues, y_preds):
        f1s.append(f1_score(y_true, y_pred,average=None))
    return np.array(f1s)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='distributed meta-testing')
    parser.add_argument('--num_threads', type=int, default=48, help='Number of threads for parallel processing, too large may result in errors')
    parser.add_argument('--task', type=str, default='mixture', required=False, help='one of [near,mixture,out]')
    parser.add_argument('--num_task', type=int, default=300, help='Number of tasks')
    parser.add_argument('--num_shots', type=int, default=5, help='Number of shots, e.g., 1, 5, 10')
    parser.add_argument('--mode', type=str, default='linear', help='one of [linear, latent_aug]')
    parser.add_argument('--clf', type=str, default='Ridge')
    parser.add_argument('--num_aug_shots', type=int, default=100, help='Number of data augmentation shots')
    parser.add_argument('--aug_times', type=int, default=0, help='Number of latent_augmentation times')
    parser.add_argument('--num_prototypes', type=int, default=16, help='Number of prototypes')
    parser.add_argument('--dict_pth', type=str, default='wsi_workdir/workdir/dict')
    parser.add_argument('--model', type=str, required=True, help='CLP or FSP')
    parser.add_argument('--task_data_pth', type=str, default='wsi_workdir/workdir/tasks')
    parser.add_argument('--novel_class', type=int)
    args = parser.parse_args()
    args.task_data_pth = f'{args.task_data_pth}/{args.task}'

    print(datetime.datetime.now())
    if args.mode != 'linear':
        print('loading bank')
        PROTO_BANK = np.load(f'{args.dict_pth}/{args.model}/NCT_PROTO_BANK_{args.num_prototypes}.npy','r')
        SHIFT_BANK = np.load(f'{args.dict_pth}/{args.model}/NCT_SHIFT_BANK_{args.num_prototypes}.npy','r')
        print('bank loaded')
        NUM_SHIFTs = len(SHIFT_BANK[0])

    task_paths = []
    
    # linear or latent_aug
    if args.task == 'near':
        for i in range(args.num_task):
            task_paths.append(
                f'{args.task_data_pth}/task_{i}/9-way-{args.num_shots}-shot_wo_{args.novel_class}_{args.model}.npy')
    elif args.task == 'mixture':
        for i in range(args.num_task):
            task_paths.append(
                f'{args.task_data_pth}/task_{i}/5-way-{args.num_shots}-shot_{args.model}.npy')
    elif args.task == 'out' or args.task == 'out_homo':
        for i in range(args.num_task):
            task_paths.append(
                f'{args.task_data_pth}/task_{i}/3-way-{args.num_shots}-shot_{args.model}.npy')
    elif args.task == 'NCT_78_aug':
        for i in range(args.num_task):
            task_paths.append(
                f'{args.task_data_pth}/task_{i}/9-way-{args.num_shots}-shot.npy')
    else:
        raise NotImplementedError

    if args.mode == 'linear':
        results = process_map(meta_testing, task_paths, max_workers=args.num_threads)
    elif args.mode=='latent_aug':
        results = process_map(meta_testing_LatentAug, task_paths, max_workers=args.num_threads)
    else:
        raise NotImplementedError

    preds = np.array([x[1] for x in results])
    trues = np.array([x[0] for x in results])
    print(datetime.datetime.now())
    print('configs:\n', args)
    print('model:', args.model, 'num_shots:', args.num_shots, 'num_task:', args.num_task, 'mode:', args.mode, 'clf:', args.clf)
    f1s = evaluate(trues, preds)
    means, cis = [], []
    for f1 in np.transpose(f1s,(1,0)):
        m, h = mean_confidence_interval(f1)
        means.append(m)
        cis.append(h)
    
    for m, h in zip(means, cis):
        print(f'{m*100:.2f} {h*100:.2f}')
    print(f'{np.mean(means)*100:.2f} {np.mean(cis)*100:.2f}')
