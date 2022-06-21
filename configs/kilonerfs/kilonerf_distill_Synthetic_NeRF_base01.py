_base_ = [
    # '../_base_/models/nerf.py',
    # '../_base_/schedules/adam_20w_iter.py',
    # '../_base_/default_runtime.py'
]

import os
from datetime import datetime

method = 'kilo_nerf'  # [nerf, kilo_nerf, mip_nerf]
model_type = 'multi_network'  #[single_network, multi_network]
phase = 'distill'  # [pretrain, distill, finetune]

resolution_table = dict(Chair=[13, 13, 16],
                        Drums=[16, 13, 12],
                        Ficus=[8, 11, 16],
                        Hotdog=[16, 16, 6],
                        Lego=[9, 16, 10],
                        Materials=[16, 14, 5],
                        Mic=[16, 16, 15],
                        Ship=[16, 16, 9])

# optimizer
optimizer = dict(type='Adam', lr=0.001)
optimizer_config = dict(grad_clip=None)

# max_iters = 150000
max_iters = 50000  # Hotdog only needs 50000 iterations, other scenes need  150000 iterations
lr_config = None
checkpoint_config = None
log_level = 'INFO'
log_config = dict(interval=500,
                  by_epoch=False,
                  hooks=[dict(type='TextLoggerHook')])
workflow = [('train', 500), ('val', 1)]

# hooks
# 'params' are numeric type value, 'variables' are variables in local environment
train_hooks = [
    dict(type='SaveDistillResultsHook',
         params=dict(),
         variables=dict(cfg='cfg', trainset='trainset')),
    dict(type='DistllCycleHook', params=dict(), variables=dict(cfg='cfg')),
    dict(type='OccupationHook',
         params=dict()),  # no need for open-source vision
]

# runner
train_runner = dict(type='KiloNerfDistillTrainRunner')

# runtime settings
num_gpus = 1
distributed = (num_gpus > 1)  # 是否多卡，mmcv对dp多卡支持不好，故而要么单卡要么ddp多卡
work_dir = './work_dirs/kilonerfs/Synthetic_NeRF_#DATANAME#_base01/distill'
timestamp = datetime.now().strftime('%d-%b-%H-%M')

# shared params by model and data and ...
dataset_type = 'nsvf'
datadir = 'data/nsvf/Synthetic_NeRF/#DATANAME#'
max_num_networks = 512
num_networks = max_num_networks
outputs = 'color_and_density'
alpha_distance = 0.0211
convert_density_to_alpha = True
quantile_se = 0.99
skip_final = True
tree_type = 'kdtree_longest'
test_error_metric = 'quantile_se'
equal_split_metric = 'mse'
max_error = 100000
train_batch_size = 128

# resume_from = os.path.join(work_dir, 'latest.pth')
# load_from = os.path.join(work_dir, 'latest.pth')

model = dict(
    type='StudentNerfNetwork',
    cfg=dict(
        outputs=outputs,
        test_batch_size=512,
        query_batch_size=80000,
    ),
    pretrained_kwargs=dict(
        config='./configs/kilonerfs/kilonerf_pretrain_Synthetic_NeRF_base01.py',
        checkpoint=
        './work_dirs/kilonerfs/Synthetic_NeRF_#DATANAME#_base01/pretrain/latest.pth'
    ),
    multi_network=dict(  # multi network
        type='KiloNerfMultiNetwork',
        num_networks=max_num_networks,
        alpha_rgb_initalization=
        'pass_actual_nonlinearity',  # in multi network model init
        bias_initialization_method='standard',  # in multi network model init
        direction_layer_size=32,  # in multi network model init
        hidden_layer_size=32,  # in multi network model init
        late_feed_direction=True,  # in multi network model init
        network_rng_seed=8078673,  # in multi network model init
        nonlinearity_initalization=
        'pass_actual_nonlinearity',  # in multi network model init
        num_hidden_layers=2,  # in multi network model init
        num_output_channels=4,
        refeed_position_index=None,  # in multi network model init
        use_same_initialization_for_all_networks=
        True,  # in multi network model init
        weight_initialization_method=
        'kaiming_uniform',  # in multi network model init
        embedder=dict(
            type='KiloNerfFourierEmbedder',
            num_networks=max_num_networks,  # num of networks, will be changed
            input_ch=3,
            multires=
            10,  # num_frequencies, log2 of max freq for positional encoding (3D location)
            multires_dirs=
            4,  # num_frequencies_direction, this is 'multires_views' in origin codes, log2 of max freq for positional encoding (2D direction)
        ),
    ),
    render=dict(  # render model
        type='KiloNerfSimpleRender',
        alpha_distance=alpha_distance,
        convert_density_to_alpha=convert_density_to_alpha,
    ),
)

basedata_cfg = dict(
    dataset_type=dataset_type,
    datadir=datadir,
    mode='train',
    batch_index=0,
    work_dir=work_dir,
    num_examples_per_network=1000000,
    max_num_networks=max_num_networks,
    train_batch_size=train_batch_size,
    outputs=outputs,
    is_batching=False,
)

traindata_cfg = basedata_cfg.copy()
valdata_cfg = basedata_cfg.copy()

traindata_cfg.update(dict())
valdata_cfg.update(dict(mode='val', num_examples_per_network=20000))

train_pipeline = [
    dict(
        type='ExampleSample',
        enable=True,
        train_batch_size=train_batch_size,
    ),
    dict(
        type='ToTensor',
        enable=True,
        keys=['domain_mins', 'domain_maxs'],
    ),
    dict(type='DeleteUseless', enable=True, keys=[
        'all_examples'
    ]),  # delete batch_examples after getting batch_inputs and batch_targets
]

test_pipeline = [
    dict(
        type='ToTensor',
        enable=True,
        keys=['domain_mins', 'domain_maxs'],
    ),
]

data = dict(
    train_loader=dict(batch_size=1, num_workers=4),
    train=dict(
        type='KiloNerfNodeDataset',
        cfg=traindata_cfg,
        pipeline=train_pipeline,
    ),
    val_loader=dict(batch_size=1, num_workers=0),
    val=dict(
        type='KiloNerfNodeDataset',
        cfg=valdata_cfg,
        pipeline=test_pipeline,
    ),
)
