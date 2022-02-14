import numpy as np
import  argparse
import os
import warnings
import threading
from tqdm import tqdm
warnings.filterwarnings("ignore")


def generate_near_domain_task(out_dir, task_ids, num_shots, options):
    out_dir = f'{out_dir}/near'
    nv = options['novel_class']
    if options['initialization'] or options['overwrite']:
        for _ in tqdm(range(len(task_ids))):
            task_id = task_ids[_]
            os.makedirs(f'{out_dir}/task_{task_id}',exist_ok=True)
            support_xs, support_ys = [], []
            query_xs, query_ys = [], []
            support_idxs, query_idxs = [], []
            for label in range(9):
                #### choose global index ###
                _support_idxs = np.random.choice(train_label_idxs[label], num_shots)
                _query_idxs = np.random.choice(test_label_idxs[label], num_querys)
                ## collect index ##
                support_idxs.append(_support_idxs)
                query_idxs.append(_query_idxs)

                ## collect support and query features ###
                support_xs.append(NCT_train_feats[_support_idxs])
                support_ys.append(NCT_train_labels[_support_idxs])
                query_xs.append(NCT_test_feats[_query_idxs])
                query_ys.append(NCT_test_labels[_query_idxs])

            ### concatenate features and labels ###
            support_xs = np.concatenate(support_xs)
            support_ys = np.concatenate(support_ys)
            query_xs = np.concatenate(query_xs)
            query_ys = np.concatenate(query_ys)

            ### get indexes ####
            support_idxs = np.concatenate(support_idxs)
            query_idxs = np.concatenate(query_idxs)

            task_name = f'9-way-{num_shots}-shot_wo_{nv}'
            np.save(f"{out_dir}/task_{task_id}/{task_name}_{options['model']}.npy", (support_xs, support_ys, query_xs, query_ys))
            np.save(f'{out_dir}/task_{task_id}/{task_name}_idxs.npy', (support_idxs, query_idxs))
    else:
        for _ in tqdm(range(len(task_ids))):
            task_id = task_ids[_]
            os.makedirs(f'{out_dir}/task_{task_id}',exist_ok=True)
            task_name = f'9-way-{num_shots}-shot_wo_{nv}'

            support_idxs, query_idxs = \
                np.load(f'{out_dir}/task_{task_id}/{task_name}_idxs.npy', allow_pickle=True)
            
            support_xs = NCT_train_feats[support_idxs]
            query_xs = NCT_test_feats[query_idxs]
            support_ys = NCT_train_labels[support_idxs]
            query_ys = NCT_test_labels[query_idxs]
            np.save(f"{out_dir}/task_{task_id}/{task_name}_{options['model']}.npy", (support_xs, support_ys, query_xs, query_ys))
            # np.save(f'{out_dir}/task_{task_id}/{task_name}_idxs.npy', (support_idxs, query_idxs))    

def generate_mixture_domain_task(out_dir, task_ids, num_shots, options):
    task_name = f'5-way-{num_shots}-shot'
    out_dir = f'{out_dir}/mixture'
    if options['initialization'] or options['overwrite']:
        for _ in tqdm(range(len(task_ids))):
            task_id = task_ids[_]
            os.makedirs(f'{out_dir}/task_{task_id}',exist_ok=True)
            support_xs, support_ys = [], []
            query_xs, query_ys = [], []
            support_idxs, query_idxs = [], []

            for label in np.arange(5):
                #### choose global index ###
                _support_idxs = np.random.choice(label_idxs[label], num_shots, replace=False)
                _query_idxs = np.random.choice(label_idxs[label], num_querys)
                ## collect index ##
                support_idxs.append(_support_idxs)
                query_idxs.append(_query_idxs)

                ## collect support and query features ###
                support_xs.append(feats[_support_idxs])
                support_ys.append(labels[_support_idxs])
                query_xs.append(feats[_query_idxs])
                query_ys.append(labels[_query_idxs])

            ### concatenate features and labels ###
            support_xs = np.concatenate(support_xs)
            support_ys = np.concatenate(support_ys)
            query_xs = np.concatenate(query_xs)
            query_ys = np.concatenate(query_ys)

            ### get indexes ####
            support_idxs = np.concatenate(support_idxs)
            query_idxs = np.concatenate(query_idxs)

            np.save(f"{out_dir}/task_{task_id}/{task_name}_{options['model']}.npy", (support_xs, support_ys, query_xs, query_ys))
            np.save(f'{out_dir}/task_{task_id}/{task_name}_idxs.npy', (support_idxs, query_idxs))
    else:
        for _ in tqdm(range(len(task_ids))):
            task_id = task_ids[_]
            support_idxs, query_idxs = \
                np.load(f'{out_dir}/task_{task_id}/{task_name}_idxs.npy', allow_pickle=True)
            support_xs = feats[support_idxs]
            query_xs = feats[query_idxs]
            support_ys = labels[support_idxs]
            query_ys = labels[query_idxs]

            np.save(f"{out_dir}/task_{task_id}/{task_name}_{options['model']}.npy", (support_xs, support_ys, query_xs, query_ys))
            # print('saving to',f'{out_dir}/task_{task_id}/{task_name}.npy')

def generate_out_domain_task(out_dir, task_ids, num_shots, options):
    task_name = f'3-way-{num_shots}-shot'
    if options['mode'] == 'hetero':
        out_dir = f'{out_dir}/out'
    elif options['mode'] == 'homo':
        out_dir = f'{out_dir}/out_homo'
    else:
        raise NotImplementedError

    if options['initialization'] or options['overwrite']:
        for _ in tqdm(range(len(task_ids))):
            task_id = task_ids[_]
            os.makedirs(f'{out_dir}/task_{task_id}',exist_ok=True)
            support_xs, support_ys = [], []
            query_xs, query_ys = [], []
            support_idxs, query_idxs = [], []

            if options['mode'] == 'homo':
                support_id = np.random.choice(np.unique(PAIP_train_ids))
            else:
                support_id = None

            for label in range(3):
                #### choose global index ###
                if options['mode'] == 'hetero': # hetero mode
                    _train_label_idxs = train_label_idxs[label]
                elif options['mode'] == 'homo':
                    _train_label_idxs =  np.where((PAIP_train_labels==label)\
                     * (PAIP_wsi_ids==support_id))[0]

                _support_idxs = np.random.choice(_train_label_idxs, num_shots, replace=False)
                _query_idxs = np.random.choice(test_label_idxs[label], num_querys)
                ## collect index ##
                support_idxs.append(_support_idxs)
                query_idxs.append(_query_idxs)

                ## collect support and query features ###
                support_xs.append(PAIP_train_feats[_support_idxs])
                support_ys.append(PAIP_train_labels[_support_idxs])
                query_xs.append(PAIP_test_feats[_query_idxs])
                query_ys.append(PAIP_test_labels[_query_idxs])

            ### concatenate features and labels ###
            support_xs = np.concatenate(support_xs)
            support_ys = np.concatenate(support_ys)
            query_xs = np.concatenate(query_xs)
            query_ys = np.concatenate(query_ys)

            ### get indexes ####
            support_idxs = np.concatenate(support_idxs)
            query_idxs = np.concatenate(query_idxs)

            np.save(f"{out_dir}/task_{task_id}/{task_name}_{options['model']}.npy", (support_xs, support_ys, query_xs, query_ys))
            np.save(f'{out_dir}/task_{task_id}/{task_name}_idxs.npy', (support_idxs, query_idxs))
    else:

        for _ in tqdm(range(len(task_ids))):
            task_id = task_ids[_]
            os.makedirs(f'{out_dir}/task_{task_id}', exist_ok=True)
            support_idxs, query_idxs = \
                np.load(f'{out_dir}/task_{task_id}/3-way-{num_shots}-shot_idxs.npy', allow_pickle=True)
            support_xs = PAIP_train_feats[support_idxs]
            query_xs = PAIP_test_feats[query_idxs]
            support_ys = PAIP_train_labels[support_idxs]
            query_ys = PAIP_test_labels[query_idxs]

            np.save(f"{out_dir}/task_{task_id}/{task_name}_{options['model']}.npy", (support_xs, support_ys, query_xs, query_ys))
            np.save(f'{out_dir}/task_{task_id}/{task_name}_idxs.npy', (support_idxs, query_idxs))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='generate tasks')
    parser.add_argument('--num_threads', type=int, default=10, help='Number of threads for parallel processing, too large may result in errors')
    parser.add_argument('--task', type=str, default=None, required=True, help='one of [near, middle, out]')
    parser.add_argument('--num_task', type=int, default=1000, help='number of tasks to generate')
    parser.add_argument('--num_shots', type=int, default=10, help='number of support shot per task')
    parser.add_argument('--aug_times', type=int, default=0, help='number of augmentation times,\
                                                                  the data augmented features are pre-extracted.')
    parser.add_argument('--mode', type=str, default='hetero', help='one of hetero or homo')
    parser.add_argument('--model', type=str, default='clp', help='one of fsp or clp')
    parser.add_argument('--initialization', action='store_true', default=False, help='first time to generate task?')
    parser.add_argument('--overwrite', action='store_true', default=False, help='overwrite existing tasks,\
                                                                        be careful with this option.')
    parser.add_argument('--task_out_dir', type=str, default='wsi_workdir/workdir/tasks')
    parser.add_argument('--novel_class', type=int, required=False, help='novel class in near domain task')
    args = parser.parse_args()

    num_querys = 15
    print(args.initialization)
    options=dict(overwrite=args.overwrite,
                initialization=args.initialization,
                model=args.model)

    print('generating tasks, this could take a while for big dataset, please be patient')
    if not os.path.exists(args.task_out_dir):
        os.makedirs(args.task_out_dir, exist_ok=True)
    
    if args.task == 'near':
        task_gen_func = generate_near_domain_task
        root = 'wsi_workdir/workdir/extracted_feats' 
        NCT_train_feats = np.load(f'{root}/{args.model}/NCT_train.npy', 'r')
        NCT_train_labels = np.load(f'{root}/NCT_train_labels.npy', 'r')
        train_label_idxs = [np.where(NCT_train_labels==label)[0] for label in range(9)]

        NCT_test_feats = np.load(f'{root}/{args.model}/NCT_test.npy', 'r')
        NCT_test_labels = np.load(f'{root}/NCT_test_labels.npy', 'r')
        test_label_idxs = [np.where(NCT_test_labels==label)[0] for label in range(9)]

        options['aug_times'] = args.aug_times
        options['novel_class'] = args.novel_class
        print('ready to generate tasks')


    elif args.task == 'mixture':
        task_gen_func = generate_mixture_domain_task
        root = 'wsi_workdir/workdir/extracted_feats'
        feats = np.load(f'{root}/{args.model}/LC.npy')
        labels = np.load(f'{root}/LC_labels.npy', 'r')
        label_idxs = [np.where(labels==label)[0] for label in range(5)]

    elif args.task == 'out':
        task_gen_func = generate_out_domain_task
        options['mode']=args.mode
        root = 'wsi_workdir/workdir/extracted_feats' 
        PAIP_train_feats = np.load(f'{root}/{args.model}/PAIP_train.npy', 'r')
        PAIP_train_labels = np.load(f'{root}/PAIP_train_labels.npy', 'r')
        train_label_idxs = [np.where(PAIP_train_labels==label)[0] for label in range(3)]

        PAIP_test_feats = np.load(f'{root}/{args.model}/PAIP_test.npy', 'r')
        PAIP_test_labels = np.load(f'{root}/PAIP_test_labels.npy', 'r')
        test_label_idxs = [np.where(PAIP_test_labels==label)[0] for label in range(3)]

        data_pth = 'data/PAIP19/data'
        wsi_ids = np.sort(os.listdir(data_pth))
        PAIP_train_ids = np.array(wsi_ids[:15])
        PAIP_test_ids = np.array(wsi_ids[15:])

        data_list = 'data/PAIP19/meta/paip_train_labeled.txt'
        pos = open(data_list,'r')
        pos = pos.readlines()
        PAIP_wsi_ids = np.array([x.split('/')[0] for x in pos])

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