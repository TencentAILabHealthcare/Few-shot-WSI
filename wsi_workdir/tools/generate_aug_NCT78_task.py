import numpy as np
import  argparse
import os
import warnings
import threading
from tqdm import tqdm
warnings.filterwarnings("ignore")


def aug_NCT78_task(out_dir, task_ids, num_shots, options):
    task_name = f'9-way-{num_shots}-shot'
    out_dir = f'{out_dir}/NCT_78_aug'
    for _ in tqdm(range(len(task_ids))):
        task_id = task_ids[_]
        os.makedirs(f'{out_dir}/task_{task_id}' ,exist_ok=True)
        support_idxs, query_idxs = \
            np.load(f'{out_dir}/task_{task_id}/{task_name}_idxs.npy', allow_pickle=True)
        support_xs, support_ys, query_xs, query_ys = \
            np.load(f'{out_dir}/task_{task_id}/{task_name}.npy', allow_pickle=True)
        

        support_xs_aug = []
        for aug_feat in aug_feats:
            support_xs_aug.append(aug_feat[support_idxs])

        support_xs_aug = np.array(support_xs_aug)
        support_xs_aug = np.transpose(support_xs_aug, (1,0,2))
        support_xs = np.concatenate(
            [support_xs[:, np.newaxis, ], support_xs_aug],
             axis=1
        )       
        np.save(f"{out_dir}/task_{task_id}/{task_name}_aug_{options['aug_times']}.npy", (support_xs, support_ys, query_xs, query_ys))
        # np.save(f'{out_dir}/task_{task_id}/{task_name}_idxs.npy', (support_idxs, query_idxs))      
        # print('saving to', f'{out_dir}/task_{task_id}/{task_name}_idxs.npy')

def generate_NCT78_task(out_dir, task_ids, num_shots, options):
    task_name = f'9-way-{num_shots}-shot'
    out_dir = f'{out_dir}/NCT_78_aug'
    for _ in tqdm(range(len(task_ids))):
        task_id = task_ids[_]
        os.makedirs(f'{out_dir}/task_{task_id}',exist_ok=True)
        support_xs, support_ys = [], []
        query_xs, query_ys = [], []
        support_idxs, query_idxs = [], []

        for label in np.arange(9):
            #### choose global index ###
            _support_idxs = np.random.choice(train_label_idxs[label], num_shots, replace=False)
            _query_idxs = np.random.choice(test_label_idxs[label], num_querys)
            ## collect index ##
            support_idxs.append(_support_idxs)
            query_idxs.append(_query_idxs)

            ## collect support and query features ###
            support_xs.append(train_feats[_support_idxs])
            support_ys.append(train_labels[_support_idxs])
            query_xs.append(test_feats[_query_idxs])
            query_ys.append(test_labels[_query_idxs])

        ### concatenate features and labels ###
        support_xs = np.concatenate(support_xs)
        support_ys = np.concatenate(support_ys)
        query_xs = np.concatenate(query_xs)
        query_ys = np.concatenate(query_ys)

        ### get indexes ####
        support_idxs = np.concatenate(support_idxs)
        query_idxs = np.concatenate(query_idxs)

        np.save(f"{out_dir}/task_{task_id}/{task_name}.npy", (support_xs, support_ys, query_xs, query_ys))
        np.save(f'{out_dir}/task_{task_id}/{task_name}_idxs.npy', (support_idxs, query_idxs))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Crop the WSIs into patches')
    parser.add_argument('--task', type=str, default='NCT78_aug')
    parser.add_argument('--num_threads', type=int, default=10, help='Number of threads for parallel processing, too large may result in errors')
    parser.add_argument('--num_task', type=int, default=300, help='number of tasks')
    parser.add_argument('--num_shots', type=int, default=5, help='number of shots')
    parser.add_argument('--aug_times', type=int, default=0, help='number of augmentation times')
    parser.add_argument('--task_out_dir', type=str, default='wsi_workdir/workdir/tasks')
    args = parser.parse_args()
    
    num_querys = 15

    print('generating tasks, this could take a while for big dataset, please be patient')
    if not os.path.exists(args.task_out_dir):
        os.makedirs(args.task_out_dir, exist_ok=True)
    

    if args.task == 'NCT78':
        task_gen_func = generate_NCT78_task
        root = 'wsi_workdir/workdir/extracted_feats' 
        train_feats = np.load(f'{root}/clp_wo_78/NCT_train.npy', 'r')
        train_labels = np.load(f'{root}/NCT_train_labels.npy', 'r')
        train_label_idxs = [np.where(train_labels==label)[0] for label in range(9)]

        test_feats = np.load(f'{root}/clp_wo_78/NCT_test.npy', 'r')
        test_labels = np.load(f'{root}/NCT_test_labels.npy', 'r')
        test_label_idxs = [np.where(test_labels==label)[0] for label in range(9)]
        options = None

    elif args.task == 'NCT78_aug':
        task_gen_func = aug_NCT78_task
        options = {
            'aug_times':args.aug_times,
        }
        root = 'wsi_workdir/workdir/extracted_feats/clp_wo_78/NCT_aug'

        aug_feats = []
        print('loading features')
        for aug_times in tqdm(range(args.aug_times)):
            aug_feats.append(np.load(f'{root}/NCT_aug_{aug_times}.npy', 'r'))
        print('features loaded')

    else:
        raise NotImplementedError

    each_thread = int(np.floor(args.num_task/args.num_threads))
    task_ids = np.arange(args.num_task)
    threads = []
    for i in range(args.num_threads):
        if i < (args.num_threads-1):
            t = threading.Thread(
                    target=task_gen_func, 
                    args=(args.task_out_dir,
                          task_ids[each_thread*i:each_thread*(i+1)],
                          args.num_shots,
                          options))
        else:
            t = threading.Thread(
                    target=task_gen_func, 
                    args=(args.task_out_dir,
                          task_ids[each_thread*i:], 
                          args.num_shots,
                          options))
        threads.append(t)

    for thread in threads:
        thread.start()