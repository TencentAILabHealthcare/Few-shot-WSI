import argparse
import importlib
import numpy as np
import os
import os.path as osp
import time
from tqdm import trange,tqdm
import threading


import mmcv
import torch
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import get_dist_info, init_dist, load_checkpoint

from openselfsup.utils import dist_forward_collect, nondist_forward_collect
from openselfsup.datasets import build_dataloader, build_dataset
from openselfsup.models import build_model
from openselfsup.models.utils import MultiPooling
from openselfsup.utils import get_root_logger
from torch import nn
import argparse

def nondist_forward_collect(func, data_loader, length):
    results = []
    prog_bar = mmcv.ProgressBar(len(data_loader))
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = func(**data)
        results.append(result)
        prog_bar.update()

    results_all = {}
    for k in results[0].keys():
        results_all[k] = np.concatenate(
            [batch[k].numpy() for batch in results], axis=0)
        assert results_all[k].shape[0] == length

    return results_all

def extract(model, data_loader):
    model.eval()
    func = lambda **x: model(mode='extract', **x)
    results = nondist_forward_collect(func, data_loader,
                                            len(data_loader.dataset))
    return results



def main(args):
    config_file = args.config
    
    cfg = mmcv.Config.fromfile(config_file)

    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    
    dataset = build_dataset(cfg.data.extract)
    data_loader = build_dataloader(
            dataset,
            imgs_per_gpu=cfg.data.imgs_per_gpu,
            workers_per_gpu=cfg.data.workers_per_gpu,
            dist=False,
            shuffle=False)

    cfg.model.pretrained = args.pretrained
    model = build_model(cfg.model)

    model = MMDataParallel(model, device_ids=[0])
    model.eval()
    func = lambda **x: model(mode='extract', **x)
    result_dict = extract(model, data_loader)
    features = result_dict['backbone']
    np.save(args.output, features)

 
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='extract dataset features using pretrained ')
    parser.add_argument('--pretrained', type=str, required=True, help='path to pretrained model')
    parser.add_argument('--config', type=str, required=True, help='path to data root')
    parser.add_argument('--output', type=str, required=True, help='output path')
    parser.add_argument('--start', type=int, required=False)
    args = parser.parse_args()
    main(args)
    exit()

    ## extract augmented features
    
    config_file = args.config
    
    cfg = mmcv.Config.fromfile(config_file)

    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    
    dataset = build_dataset(cfg.data.extract)
    data_loader = build_dataloader(
            dataset,
            imgs_per_gpu=cfg.data.imgs_per_gpu,
            workers_per_gpu=cfg.data.workers_per_gpu,
            dist=False,
            shuffle=False)

    cfg.model.pretrained = args.pretrained
    model = build_model(cfg.model)

    model = MMDataParallel(model, device_ids=[0])
    model.eval()
    func = lambda **x: model(mode='extract', **x)

    def extract_and_save(idxs):
        for idx in tqdm(idxs):
            result_dict = nondist_forward_collect(func, data_loader, len(data_loader.dataset))
            features = result_dict['backbone']
            np.save(f'wsi_workdir/workdir/extracted_feats/moco_v3_wo_78/NCT_aug/NCT_aug_{idx}.npy', features)
            print('saving', idx)

    extract_and_save(np.arange(args.start, args.start+25))



