import argparse
import importlib
import numpy as np
import os
import os.path as osp
import time

import mmcv
import torch
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import get_dist_info, init_dist, load_checkpoint
from munkres import Munkres

from openselfsup.utils import dist_forward_collect, nondist_forward_collect
from openselfsup.datasets import build_dataloader, build_dataset
from openselfsup.models import build_model
from openselfsup.models.utils import MultiPooling
from openselfsup.utils import get_root_logger
from torch import nn
import pdb
from sklearn.metrics import confusion_matrix
import cv2
import math
import torch.nn.functional as F
from sklearn.cluster import KMeans

def get_visualization(model, data_loader):
    model.eval()
    func = lambda **x: model(mode='multi_layer_map', **x)
    prog_bar = mmcv.ProgressBar(len(data_loader))
    for data in data_loader:
        with torch.no_grad():
            func(**data)
        prog_bar.update()


def main():
    parser = argparse.ArgumentParser(description='visualize features using pretrained ')
    parser.add_argument('--model', type=str, required=True, help='which model')
    parser.add_argument('--novel_class', type=int, required=True, help='novel class')
    parser.add_argument('--config_root', 
                        type=str, required=False, 
                        default='configs/submission_visualize', 
                        help='path to config root')

    args = parser.parse_args()
    novel_dict = {
        0: '0_BACK', 1: '1_ADI', 2: '2_DEB', 3: '3_LYM', 4: '4_MUC', 5:'5_MUS', 6: '6_NORM', 7:'7_STR', 8:'8_TUM',
        9: 'LC_0_colon_aca', 10: 'LC_1_colon_benign', 11: 'LC_2_lung_aca', 12: 'LC_3_lung_benign', 13:'LC_4_lung_scc',
        14: 'PAIP_0', 15: 'PAIP_1', 16:'PAIP_2',
    }
    config_file = f'{args.config_root}/visualize_feats_NCT_{args.novel_class}.py'
    # config_file = f'{args.config_root}/visualize_feats_LC_{args.novel_class-9}.py'
    # config_file = f'{args.config_root}/visualize_feats_PAIP_{args.novel_class-14}.py'
    cfg = mmcv.Config.fromfile(config_file)
    cfg_fsp = cfg.deepcopy()
    cfg_clp = cfg.deepcopy()

    torch.backends.cudnn.benchmark = True
    
    dataset = build_dataset(cfg.data.extract)
    data_loader = build_dataloader(
            dataset,
            imgs_per_gpu=cfg.data.imgs_per_gpu,
            workers_per_gpu=cfg.data.workers_per_gpu,
            dist=False,
            shuffle=False)
    
    pretrained_roots = 'wsi_workdir/workdir/pretrained_weights'
    
    ## near-domain task
    cfg_fsp.model.pretrained = f'{pretrained_roots}/sup_wo_{args.novel_class}.pth'
    cfg_clp.model.pretrained = f'{pretrained_roots}/clp_wo_{args.novel_class}.pth'
    
    ## ablation study
    # cfg_fsp.model.pretrained = f'{pretrained_roots}/sup_wo_78.pth'
    # cfg_clp.model.pretrained = f'{pretrained_roots}/clp_wo_78.pth'

    ## mixture-/out-domain tasks
    # cfg_fsp.model.pretrained = f'{pretrained_roots}/fsp.pth'
    # cfg_clp.model.pretrained = f'{pretrained_roots}/clp.pth'
    model_fsp = build_model(cfg_fsp.model)
    model_fsp = MMDataParallel(model_fsp, device_ids=[0])

    model_clp = build_model(cfg_clp.model)
    model_clp = MMDataParallel(model_clp, device_ids=[0])


    model_fsp.eval()
    model_clp.eval()
    func_fsp = lambda **x: model_fsp(mode='forward_backbone', **x)
    func_clp = lambda **x: model_clp(mode='forward_backbone', **x)

    mean = np.array([0.485, 0.456, 0.406])*255
    std = np.array([0.229, 0.224, 0.225])*255
    img_id = 0
    num_imgs = 100
    selected_ids = np.arange(0, 100)
    for data in data_loader:
        with torch.no_grad():
            img = data['img'].cuda()
            clp_backbone_feats = func_clp(**data)
            sup_backbone_feats = func_fsp(**data)
            batch_img = img.cpu()
            for i in range(3):
                batch_img[:,i,...] = batch_img[:,i,...] * std[i] + mean[i]
            batch_img = np.uint8(batch_img).transpose(0,2,3,1)

            for bix in range(len(batch_img)):
                print(img_id)
                if img_id > num_imgs:
                    exit()
                if img_id not in selected_ids:
                    img_id+=1
                    continue
                
                clp_multi_attention, clp_multi_cluster = forward(clp_backbone_feats, batch_img, bix, with_img=True)
                sup_multi_attention, sup_multi_cluster = forward(sup_backbone_feats, batch_img, bix, with_img=False)

                white = [255,255,255]
                multi_clp_map = []
                multi_fsp_map = []
                for layer_clp, layer_fsp in zip(clp_multi_cluster, sup_multi_cluster):
                    clp_results = []
                    sup_results = []
                    for k_map_clp, k_map_fsp, k in zip(layer_clp, layer_fsp, [2,4,6]):
                        k_map_fsp = match_assignment(k_map_clp, k_map_fsp, k)
                        k_map_clp = np.reshape(k_map_clp, (int(math.sqrt(len(k_map_clp))), -1))
                        k_map_fsp = np.reshape(k_map_fsp, (int(math.sqrt(len(k_map_fsp))), -1))

                        clp_map = cv2.applyColorMap(np.uint8(k_map_clp/k * 255), cv2.COLORMAP_RAINBOW)
                        clp_map = cv2.resize(clp_map, (448,448), interpolation=cv2.INTER_NEAREST)
                        clp_map = clp_map *0.4 + batch_img[bix] * 0.6
                        clp_map = cv2.copyMakeBorder(np.uint8(clp_map),10, 10, 10, 10,cv2.BORDER_CONSTANT,value=white)

                        clp_results.append(clp_map)

                        sup_map = cv2.applyColorMap(np.uint8(k_map_fsp/k * 255), cv2.COLORMAP_RAINBOW)
                        sup_map = cv2.resize(sup_map, (448,448), interpolation=cv2.INTER_NEAREST)
                        sup_map = sup_map *0.4 + batch_img[bix] * 0.6
                        sup_map = cv2.copyMakeBorder(np.uint8(sup_map),10, 10, 10, 10,cv2.BORDER_CONSTANT,value=white)

                        sup_results.append(sup_map)

                    multi_clp_map.append(cv2.hconcat(clp_results))
                    multi_fsp_map.append(cv2.hconcat(sup_results))
                multi_clp_map = cv2.vconcat(multi_clp_map)
                multi_fsp_map = cv2.vconcat(multi_fsp_map)
                
                final_fsp = cv2.hconcat([sup_multi_attention,multi_fsp_map])
                final_fsp = cv2.copyMakeBorder(final_fsp, 0, 0, 15, 0, cv2.BORDER_CONSTANT,value=white)
                final_clp = cv2.hconcat([clp_multi_attention,multi_clp_map])
                final_img = cv2.hconcat([final_clp, final_fsp])

                # out_dir = f"wsi_workdir/workdir/plots/wo_78_attention/7_STR"
                out_dir = f"wsi_workdir/workdir/plots/attention/{novel_dict[args.novel_class]}"
                # cv2.imwrite(f'{out_dir}/{img_id}_clp.jpg', final_clp)
                # cv2.imwrite(f'{out_dir}/{img_id}_fsp.jpg', final_fsp)
                cv2.imwrite(f'{out_dir}/{img_id}.jpg', final_img)
                img_id += 1


avgpool = nn.AdaptiveAvgPool2d((1,1))

def forward(backbone_feats, batch_img, bix, with_img=True):
    size_upsample = (448, 448)
    mean = np.array([0.485, 0.456, 0.406])*255
    std = np.array([0.229, 0.224, 0.225])*255
    multi_attention = []
    multi_cluster = []
    for x in backbone_feats:
        global_x = avgpool(x).view(x.size(0), -1)
        global_x = F.normalize(global_x, p=2, dim=1)
        x = F.normalize(x, p=2, dim=1)  # B, C, H, W
        patch = x[bix].permute(1,2,0)     # H, W, C
        patch = patch.view(-1, patch.size(-1)) 
        patch_size = int(math.sqrt(patch.size(0)))
        attention_map = get_cam(global_feat=global_x[bix], 
                                local_feats=patch, 
                                img=batch_img[bix], 
                                patch_size=patch_size, 
                                size_upsample=size_upsample,
                                with_img=with_img)
        assignments_list = get_clustered_local_feats_tmp(
                    local_feats=patch, img=batch_img[bix], patch_size=patch_size, size_upsample=size_upsample
                )
        multi_attention.append(cv2.hconcat(attention_map))
        multi_cluster.append(assignments_list)
    multi_attention = cv2.vconcat(multi_attention)
    return multi_attention, multi_cluster

def get_clustered_local_feats_tmp(local_feats, img, patch_size, size_upsample=(448,448), num_clusters=[2,4,6]):
    
    local_feats = np.ascontiguousarray(local_feats.cpu().numpy())
    assignment_list = []
    for k in num_clusters:
        kmeans = KMeans(n_clusters=k, random_state=0).fit(local_feats)
        assignment_list.append(kmeans.labels_)
    return assignment_list

def get_cam(global_feat, local_feats, img, patch_size, size_upsample=(448,448), with_img=True):
    absolute_cam = (local_feats @ global_feat.unsqueeze(1)).view(-1)
    normalized_cam = absolute_cam.clone()

    absolute_cam *= 255
    absolute_cam = np.uint8(absolute_cam.view(patch_size,-1).cpu().numpy())
    absolute_cam = cv2.resize(absolute_cam, size_upsample)
    absolute_cam = cv2.applyColorMap(absolute_cam, cv2.COLORMAP_JET)

    normalized_cam = (normalized_cam - normalized_cam.min())/(normalized_cam.max() - normalized_cam.min())
    normalized_cam *= 255
    normalized_cam = np.uint8(normalized_cam.view(patch_size,-1).cpu().numpy())
    normalized_cam = cv2.resize(normalized_cam, size_upsample)
    normalized_cam = cv2.applyColorMap(normalized_cam, cv2.COLORMAP_JET)

    _img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    white = [255,255,255]
    absolute_cam = absolute_cam * 0.4 + _img * 0.6
    normalized_cam = normalized_cam * 0.4 + _img * 0.6
    src_img = cv2.copyMakeBorder(np.uint8(_img),10, 10, 10, 10,cv2.BORDER_CONSTANT,value=white)
    absolute_cam = cv2.copyMakeBorder(np.uint8(absolute_cam),10, 10, 10, 10,cv2.BORDER_CONSTANT,value=white)
    normalized_cam = cv2.copyMakeBorder(np.uint8(normalized_cam),10, 10, 10, 10,cv2.BORDER_CONSTANT,value=white)

    if with_img:
        attention_map = [src_img, absolute_cam, normalized_cam]
    else:
        attention_map = [absolute_cam, normalized_cam]
    return attention_map

def compute_cost_matrix(C, n_clusters):
    cost_matrix = np.zeros((n_clusters, n_clusters))
    for j in range(n_clusters):
        s = np.sum(C[:, j])  # number of examples in cluster i
        for i in range(n_clusters):
            t = C[i, j]
            cost_matrix[j, i] = s - t
    return cost_matrix

def get_cluster_labels_from_indices(indices):
    n_clusters = len(indices)
    cluster_labels = np.zeros(n_clusters)
    for i in range(n_clusters):
        cluster_labels[i] = indices[i][1]
    return cluster_labels

def match_assignment(target, source, n_clusters):
    cf_mat = confusion_matrix(target, source, labels=None)
    cost_matrix = compute_cost_matrix(cf_mat, n_clusters)
    indices = Munkres().compute(cost_matrix)
    kmeans_to_true_cluster_labels = get_cluster_labels_from_indices(indices)

    if np.min(source) != 0:
        source = source - np.min(source)
    matched_assignments = kmeans_to_true_cluster_labels[source]
    return matched_assignments


if __name__ == '__main__':
    main()
