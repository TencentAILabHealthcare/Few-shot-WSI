_base_ = '../../base.py'
# model settings
model = dict(
    type='MOCOv3',
    base_momentum=0.996,
    backbone=dict(
        type='ResNet',
        depth=18,
        in_channels=3,
        out_indices=[4],  # 0: conv-1, x: stage-x
        norm_cfg=dict(type='BN')),
    projector=dict(
        type='NonLinearNeckSimCLR',
        in_channels=512,
        hid_channels=1024,
        out_channels=256,
        num_layers=3,
        sync_bn=True,
        with_bias=False,
        with_last_bn=True,
        with_avg_pool=True),
    predictor=dict(
        type='NonLinearNeckSimCLR',
        in_channels=256,
        hid_channels=1024,
        out_channels=256,
        num_layers=2,
        sync_bn=True,
        with_bias=False,
        with_last_bn=True,
        with_avg_pool=False),
    temperature=1,
    )
# dataset settings
data_source_cfg = dict(type='ImageList')

data_train_list = 'data/NCT/meta/wo_2_train.txt'
data_train_root = 'data/NCT/data'

dataset_type = 'ContrastiveDataset'

img_norm_cfg = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

train_pipeline = [
    dict(type='RandomResizedCrop', size=224, scale=(0.2, 1.)),
    dict(type='RandomHorizontalFlip'),
    dict(
        type='RandomAppliedTrans',
        transforms=[
            dict(
                type='ColorJitter',
                brightness=0.4,
                contrast=0.4,
                saturation=0.4,
                hue=0.4)
        ],
        p=0.8),
    dict(type='RandomGrayscale', p=0.2),
    dict(
        type='RandomAppliedTrans',
        transforms=[
            dict(
                type='GaussianBlur',
                sigma_min=0.1,
                sigma_max=2.0)
        ],
        p=0.5),
    dict(type='RandomAppliedTrans',
         transforms=[dict(type='Solarization')], p=0.2)
]

test_pipeline = [
    dict(type='Resize', size=224),
]

# prefetch
prefetch = True 
if not prefetch:
    train_pipeline.extend([dict(type='ToTensor'), dict(type='Normalize', **img_norm_cfg)])

batch_size =256 
data = dict(
    imgs_per_gpu=batch_size//8,  # total 32*8=256
    #imgs_per_gpu=16,
    workers_per_gpu=5,
    train=dict(
        type=dataset_type,
        data_source=dict(
            list_file=data_train_list, root=data_train_root,
            **data_source_cfg),
        pipeline=train_pipeline,
        prefetch=prefetch),
    )
    
# additional hooks
custom_hooks = [dict(type='BYOLHook', end_momentum=1.)]

# optimizer
optimizer = dict(type='LARS', lr=0.3, weight_decay=1.5e-6, momentum=0.9,)

# lr schedule
lr_config = dict(policy='CosineAnnealing', min_lr=0.)

checkpoint_config = dict(interval=50)
# runtime settings
total_epochs = 200


