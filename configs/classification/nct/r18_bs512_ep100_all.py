_base_ = '../../base.py'
# model settings
model = dict(
    type='Classification',
    pretrained=None,
    backbone=dict(
        type='ResNet',
        depth=18,
        out_indices=[4],  # 0: conv-1, x: stage-x
        norm_cfg=dict(type='BN')),
    head=dict(
        type='ClsHead', with_avg_pool=True, in_channels=512,
        num_classes=9)
    )
# dataset settings
data_source_cfg = dict(type='ImageList')

data_root = 'data/NCT/data'
dataset_type = 'ClassificationDataset'
data_train_list = 'data/NCT/meta/train_labeled.txt'


img_norm_cfg = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
train_pipeline = [
    dict(type='RandomResizedCrop', size=224),
    dict(type='RandomHorizontalFlip'),
    dict(type='ToTensor'),
    dict(type='Normalize', **img_norm_cfg),
]
test_pipeline = [
    dict(type='Resize', size=224),
    dict(type='ToTensor'),
    dict(type='Normalize', **img_norm_cfg),
]
batch_size=512
data = dict(
    imgs_per_gpu=batch_size//4,  
    workers_per_gpu=5,
    train=dict(
        type=dataset_type,
        data_source=dict(
            list_file=data_train_list, 
            root=data_root,
            **data_source_cfg),
        pipeline=train_pipeline),
    )
prefetch=False

# optimizer
optimizer = dict(type='SGD', lr=0.5, momentum=0.9, weight_decay=0.)
# learning policy
lr_config = dict(policy='step', step=[30, 60, 90])
checkpoint_config = dict(interval=100)
# runtime settings
total_epochs = 100