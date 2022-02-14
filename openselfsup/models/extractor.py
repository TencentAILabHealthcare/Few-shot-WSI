import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import math 
from sklearn.cluster import KMeans

from openselfsup.utils import print_log

from . import builder
from .registry import MODELS
from .utils import Sobel

### For visualization.

@MODELS.register_module
class Extractor(nn.Module):
    img_id = 0
    def __init__(self,
                 backbone,
                 pretrained=None):
        super(Extractor, self).__init__()
        self.backbone = builder.build_backbone(backbone)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.init_weights(pretrained=pretrained)

    def init_weights(self, pretrained=None):
        if pretrained is not None:
            print_log('load model from: {}'.format(pretrained), logger='root')
        self.backbone.init_weights(pretrained=pretrained)

    def forward(self, img, mode='extract', **kwargs):
        if mode == 'extract':
            return self.forward_extract(img)
        elif mode == 'forward_backbone':
            return self.forward_backbone(img)
        elif mode == 'multi_layer_map':
            return self.forward_multi_layer_visulization(img)
        elif mode == 'multi_layer_map_tmp':
            return self.forward_multi_layer_visulization_tmp(img)
        else:
            raise Exception("No such mode: {}".format(mode))
    
    def forward_extract(self, img, **kwargs):
        backbone_feats = self.backbone(img)
        backbone_feats = self.avgpool(backbone_feats[-1])
        backbone_feats = backbone_feats.view(backbone_feats.size(0), -1)
        backbone_feats = F.normalize(backbone_feats, p=2, dim=1)
        return dict(backbone=backbone_feats.cpu())

    def forward_backbone(self, img, **kwargs):
        backbone_feats = self.backbone(img)
        return backbone_feats


    def forward_multi_layer_visulization(self, img, **kwargs):
        backbone_feats = self.backbone(img)
        batch_img = img.cpu()
        out_dir = 'path to saving dir'
        size_upsample = (448, 448)
        mean = np.array([0.485, 0.456, 0.406])*255
        std = np.array([0.229, 0.224, 0.225])*255
        
        for i in range(3):
            batch_img[:,i,...] = batch_img[:,i,...] * std[i] + mean[i]
        batch_img = np.uint8(batch_img).transpose(0,2,3,1)

        selected_ids = np.arange(200)

        for b in range(len(batch_img)):
            multi_resuts = []
            for x in backbone_feats:
                if self.img_id not in selected_ids: # only save these two
                    continue
                global_x = self.avgpool(x).view(x.size(0), -1)
                global_x = F.normalize(global_x, p=2, dim=1)
                x = F.normalize(x, p=2, dim=1)  # B, C, H, W
                patch = x[b].permute(1,2,0)     # H, W, C
                patch = patch.view(-1, patch.size(-1)) 

                patch_size = int(math.sqrt(patch.size(0)))
                attention_map = self.get_cam(global_feat=global_x[b], 
                                             local_feats=patch, 
                                             img=batch_img[b], 
                                             patch_size=patch_size, 
                                             size_upsample=size_upsample)
                cluster_map = self.get_clustered_local_feats(
                    local_feats=patch, img=batch_img[b], patch_size=patch_size, size_upsample=size_upsample
                )
                multi_resuts.append(cv2.hconcat([*attention_map, *cluster_map]))

            final_img = cv2.vconcat(multi_resuts)
            if self.img_id in selected_ids:
                cv2.imwrite(f'{out_dir}/{self.img_id}.jpg', final_img)
                print(f'\n saving to {out_dir}/{self.img_id}.jpg')
            self.img_id+=1
            if self.img_id > selected_ids[-1]:
                exit()

    @staticmethod
    def get_cam(global_feat, local_feats, img, patch_size, size_upsample=(448,448)):
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

        attention_map = [src_img, absolute_cam, normalized_cam]
        return attention_map

    @staticmethod
    def get_clustered_local_feats(local_feats, img, patch_size, size_upsample=(448,448), num_clusters=[2,4,6]):
        white = [255,255,255]
        _img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
        local_feats = np.ascontiguousarray(local_feats.cpu().numpy())
        cluster_results = []
        for k in num_clusters:
            kmeans = KMeans(n_clusters=k, random_state=0).fit(local_feats)
            assignments = np.reshape(kmeans.labels_, (patch_size, patch_size))
            cluster_map = cv2.applyColorMap(np.uint8(assignments/k * 255), cv2.COLORMAP_RAINBOW)
            cluster_map = cv2.resize(cluster_map, size_upsample, interpolation=cv2.INTER_NEAREST)
            cluster_result = cluster_map * 0.4 + _img * 0.6
            cluster_result = cv2.copyMakeBorder(np.uint8(cluster_result),10, 10, 10, 10,cv2.BORDER_CONSTANT,value=white )

            cluster_results.append(cluster_result)
        return cluster_results


    def forward_multi_layer_visulization_tmp(self, img, **kwargs):
        backbone_feats = self.backbone(img)
        batch_img = img.cpu()
        novel_dict = {
            0 : 'colon_aca',
            1 : 'colon_benign',
            2 : 'lung_aca',
            3 : 'lung_benign',
            4 : 'lung_scc',
        }
        size_upsample = (448, 448)
        mean = np.array([0.485, 0.456, 0.406])*255
        std = np.array([0.229, 0.224, 0.225])*255
        
        for i in range(3):
            batch_img[:,i,...] = batch_img[:,i,...] * std[i] + mean[i]
        batch_img = np.uint8(batch_img).transpose(0,2,3,1)

        # selected_ids = [62, 74, 113, 119, 154]
        selected_ids = [154]

        for b in range(len(batch_img)):
            multi_resuts = []
            print(self.img_id)
            for x in backbone_feats:
                if self.img_id not in selected_ids: # only save these two
                    continue
                global_x = self.avgpool(x).view(x.size(0), -1)
                global_x = F.normalize(global_x, p=2, dim=1)
                x = F.normalize(x, p=2, dim=1)  # B, C, H, W
                patch = x[b].permute(1,2,0)     # H, W, C
                patch = patch.view(-1, patch.size(-1)) 

                patch_size = int(math.sqrt(patch.size(0)))
                assignments_list = self.get_clustered_local_feats_tmp(
                    local_feats=patch, img=batch_img[b], patch_size=patch_size, size_upsample=size_upsample
                )
                multi_resuts.append(assignments_list)

            if self.img_id in selected_ids:
                print(self.img_id, 'now in it')
                self.img_id+=1
                return multi_resuts, batch_img[b]
            self.img_id+=1
            if self.img_id > selected_ids[-1]:
                exit()
                # break

    @staticmethod
    def get_clustered_local_feats_tmp(local_feats, img, patch_size, size_upsample=(448,448), num_clusters=[2,4,6]):
        white = [255,255,255]
        _img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
        local_feats = np.ascontiguousarray(local_feats.cpu().numpy())
        assignment_list = []
        for k in num_clusters:
            kmeans = KMeans(n_clusters=k, random_state=0).fit(local_feats)
            assignment_list.append(kmeans.labels_)
        return assignment_list