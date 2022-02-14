_base_ = '../base.py'
# model settings
model = dict(
    type='Extractor',
    pretrained=None,
    backbone=dict(
        type='ResNet',
        depth=18,
        out_indices=[4],  # 0: conv-1, x: stage-x
        norm_cfg=dict(type='BN')))
# dataset settings

data_source_cfg = dict(type='ImageList')
data_root = 'data/NCT/data'
data_list = 'data/NCT/meta/img_list.txt'

img_norm_cfg = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
extract_pipeline = [
    dict(type='Resize', size=(224, 224)),
    dict(type='ToTensor'), 
    dict(type='Normalize', **img_norm_cfg)]

data = dict(
    imgs_per_gpu=256, 
    workers_per_gpu=5,
    extract=dict(
        type='ExtractDataset',
        data_source=dict(
            list_file=data_list, root=data_root, **data_source_cfg),
        pipeline=extract_pipeline),
    )

