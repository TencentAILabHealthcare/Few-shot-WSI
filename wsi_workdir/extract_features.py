import argparse
import numpy as np
import mmcv
import torch
from mmcv.parallel import MMDataParallel
from openselfsup.datasets import build_dataloader, build_dataset
from openselfsup.models import build_model
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
    
    result_dict = extract(model, data_loader)
    features = result_dict['backbone']
    np.save(args.output, features)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='extract dataset features using pretrained ')
    parser.add_argument('--pretrained', type=str, required=True, help='path to pretrained model')
    parser.add_argument('--config', type=str, required=True, help='path to data root')
    parser.add_argument('--output', type=str, required=True, help='output path')
    args = parser.parse_args()
    main(args)

