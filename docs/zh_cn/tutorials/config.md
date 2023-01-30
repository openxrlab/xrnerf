# 教程 1: 如何编写配置文件

XRNeRF 使用 python 文件作为配置文件。其配置文件系统的设计将模块化与继承整合进来，方便用户进行各种实验。
XRNeRF 提供的所有配置文件都放置在 `$PROJECT/configs` 文件夹下。

<!-- TOC -->

- [教程 1: 如何编写配置文件](#教程-1-如何编写配置文件)
  - [配置文件组成部分](#配置文件组成部分)

<!-- TOC -->

## 配置文件组成部分
配置文件的内容在逻辑上可以分为3个部分:
* 训练
* 模型
* 数据

下面的内容将会逐部分介绍配置文件
* 训练
    训练配置部分包含了控制训练过程的各类参数，包括optimizer, hooks, runner等等
    ```python
    import os
    from datetime import datetime

    method = 'nerf' # nerf方法

    # optimizer 参数
    optimizer = dict(type='Adam', lr=5e-4, betas=(0.9, 0.999))
    optimizer_config = dict(grad_clip=None)

    max_iters = 20000 # 训练多少个iter
    lr_config = dict(policy='step', step=500 * 1000, gamma=0.1, by_epoch=False) # 学习率和衰减
    checkpoint_config = dict(interval=5000, by_epoch=False) # 保存checkpoint的间隔
    log_level = 'INFO'
    log_config = dict(interval=5000,
                    by_epoch=False,
                    hooks=[dict(type='TextLoggerHook')])
    workflow = [('train', 5000), ('val', 1)] # 循环: 每训练 5000 iters, 验证 1 iter

    # hooks
    # 'params' 是数值型参数, 'variables' 是代码运行上下面出现的变量
    train_hooks = [
        dict(type='SetValPipelineHook',
            params=dict(),
            variables=dict(valset='valset')),
        dict(type='ValidateHook',
            params=dict(save_folder='visualizations/validation')),
        dict(type='SaveSpiralHook',
            params=dict(save_folder='visualizations/spiral')),
        dict(type='PassIterHook', params=dict()),  # 将当前iter数告诉dataset
    ]
    test_hooks = [
        dict(type='SetValPipelineHook',
            params=dict(),
            variables=dict(valset='testset')),
        dict(type='TestHook', params=dict()),
    ]

    # runner
    train_runner = dict(type='NerfTrainRunner')
    test_runner = dict(type='NerfTestRunner')

    # runtime settings
    num_gpus = 1
    distributed = (num_gpus > 1)  # 是否使用 ddp
    work_dir = './work_dirs/nerfsv3/nerf_#DATANAME#_base01/' # 保存运行时产生文件的位置
    timestamp = datetime.now().strftime('%d-%b-%H-%M') # 保证每次的workspace都不同

    # some shared params by model and data, to avoid define twice
    dataset_type = 'blender'
    no_batching = True  # 每次选择1张图片来抽取射线
    no_ndc = True

    white_bkgd = True  # 渲染时背景设定为全白
    is_perturb = True  # set to 0. for no jitter, 1. for jitter
    use_viewdirs = True  # use full 5D input instead of 3D
    N_rand_per_sampler = 1024 * 4  # 在取多少根射线 在 get_item() 函数中使用
    lindisp = False  # sampling linearly in disparity rather than depth
    N_samples = 64  # 在coarse模型中输入多少根射线

    # resume_from = os.path.join(work_dir, 'latest.pth')
    # load_from = os.path.join(work_dir, 'latest.pth')

    ```

* 模型
    模型部分的配置信息，定义了网络模型结构，一个network通常由embedder, mlp 和 render组成。
    ```python
    model = dict(
        type='NerfNetwork', # network 类名字
        cfg=dict(
            phase='train',  # 'train' or 'test'
            N_importance=128,  # number of additional fine samples per ray
            is_perturb=is_perturb, # see above
            chunk=1024 * 32,  # mainly work for val, to avoid oom
            bs_data='rays_o',  # the data's shape indicates the real batch-size, this's also the num of rays
        ),
        mlp=dict(  # coarse mlp model
            type='NerfMLP', # mlp class name
            skips=[4],
            netdepth=8,  # layers in network
            netwidth=256,  # channels per layer
            netchunk=1024 * 32,  # to avoid oom
            output_ch=5,  # 5 if cfg.N_importance>0 else 4
            use_viewdirs=use_viewdirs,
            embedder=dict(
                type='BaseEmbedder', # embedder class name
                i_embed=0,  # set 0 for default positional encoding, -1 for none
                multires=10,  # log2 of max freq for positional encoding (3D location)
                multires_dirs=4,  # this is 'multires_views' in origin codes, log2 of max freq for positional encoding (2D direction)
            ),
        ),
        mlp_fine=dict(  # fine model
            type='NerfMLP',
            skips=[4],
            netdepth=8,
            netwidth=256,
            netchunk=1024 * 32,
            output_ch=5,
            use_viewdirs=use_viewdirs,
            embedder=dict(
                type='BaseEmbedder',
                i_embed=0,
                multires=10,
                multires_dirs=4,
            ),
        ),
        render=dict(
            type='NerfRender', # render cloass name
            white_bkgd=white_bkgd,  # see above
            raw_noise_std=0,  # std dev of noise added to regularize sigma_a output, 1e0 recommended
        ),
    )
    ```

* 数据
    数据部分的配置信息，定义了数据集类型，数据的处理流程，batchsize等等信息。
    ```python
    basedata_cfg = dict(
        dataset_type=dataset_type,
        datadir='data/nerf_synthetic/#DATANAME#',
        half_res=True,  # load blender synthetic data at 400x400 instead of 800x800
        testskip=
        8,  # will load 1/N images from test/val sets, useful for large datasets like deepvoxels
        white_bkgd=white_bkgd,
        is_batching=False,  # True for blender, False for llff
        mode='train',
    )

    traindata_cfg = basedata_cfg.copy()
    valdata_cfg = basedata_cfg.copy()
    testdata_cfg = basedata_cfg.copy()

    traindata_cfg.update(dict())
    valdata_cfg.update(dict(mode='val'))
    testdata_cfg.update(dict(mode='test', testskip=0))

    train_pipeline = [
        dict(type='Sample'),
        dict(type='DeleteUseless', keys=['images', 'poses', 'i_data', 'idx']),
        dict(type='ToTensor', keys=['pose', 'target_s']),
        dict(type='GetRays'),
        dict(type='SelectRays',
            sel_n=N_rand_per_sampler,
            precrop_iters=500,
            precrop_frac=0.5),  # in the first 500 iter, select rays inside center of image
        dict(type='GetViewdirs', enable=use_viewdirs),
        dict(type='ToNDC', enable=(not no_ndc)),
        dict(type='GetBounds'),
        dict(type='GetZvals', lindisp=lindisp,
            N_samples=N_samples),  # N_samples: number of coarse samples per ray
        dict(type='PerturbZvals', enable=is_perturb),
        dict(type='GetPts'),
        dict(type='DeleteUseless', keys=['pose', 'iter_n']),
    ]

    test_pipeline = [
        dict(type='ToTensor', keys=['pose']),
        dict(type='GetRays'),
        dict(type='FlattenRays'),
        dict(type='GetViewdirs', enable=use_viewdirs),
        dict(type='ToNDC', enable=(not no_ndc)),
        dict(type='GetBounds'),
        dict(type='GetZvals', lindisp=lindisp, N_samples=N_samples),
        dict(type='PerturbZvals', enable=False),  # do not perturb when test
        dict(type='GetPts'),
        dict(type='DeleteUseless', keys=['pose']),
    ]
    data = dict(
        train_loader=dict(batch_size=1, num_workers=4),
        train=dict(
            type='SceneBaseDataset',
            cfg=traindata_cfg,
            pipeline=train_pipeline,
        ),
        val_loader=dict(batch_size=1, num_workers=0),
        val=dict(
            type='SceneBaseDataset',
            cfg=valdata_cfg,
            pipeline=test_pipeline,
        ),
        test_loader=dict(batch_size=1, num_workers=0),
        test=dict(
            type='SceneBaseDataset',
            cfg=testdata_cfg,
            pipeline=test_pipeline,  # same pipeline as validation
        ),
    )
    ```
