# Tutorial 1: Learn about Configs

We use python files as configs, incorporate modular and inheritance design into our config system, which is convenient to conduct various experiments.
You can find all the provided configs under `$PROJECT/configs`.

<!-- TOC -->

- [Tutorial 1: Learn about Configs](#tutorial-1-learn-about-configs)
  - [Configuration Components](#configuration-components)

<!-- TOC -->

## Configuration Components
We can logically divide the configuration file into components:
* training
* model
* data

The fllowing content explain these configuration components one by one.
* training
    training configurations contains all paramters to control model training, include optimizer, hooks, runner and soon on.
    ```python
    import os
    from datetime import datetime

    method = 'nerf' # which nerf method

    # optimizer setting
    optimizer = dict(type='Adam', lr=5e-4, betas=(0.9, 0.999))
    optimizer_config = dict(grad_clip=None)

    max_iters = 20000 # train for how many iters
    lr_config = dict(policy='step', step=500 * 1000, gamma=0.1, by_epoch=False) # learning rate and decay
    checkpoint_config = dict(interval=5000, by_epoch=False) # when to save checkpoint
    log_level = 'INFO'
    log_config = dict(interval=5000,
                    by_epoch=False,
                    hooks=[dict(type='TextLoggerHook')])
    workflow = [('train', 5000), ('val', 1)] # loop: train 5000 iters, validate 1 iter

    # hooks
    # 'params' are numeric type value, 'variables' are variables in local environment
    train_hooks = [
        dict(type='SetValPipelineHook',
            params=dict(),
            variables=dict(valset='valset')),
        dict(type='ValidateHook',
            params=dict(save_folder='visualizations/validation')),
        dict(type='SaveSpiralHook',
            params=dict(save_folder='visualizations/spiral')),
        dict(type='PassIterHook', params=dict()),  # 将当前iter数告诉dataset
        dict(type='OccupationHook',
            params=dict()),  # no need for open-source vision
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
    distributed = (num_gpus > 1)  # whether to use ddp
    work_dir = './work_dirs/nerfsv3/nerf_#DATANAME#_base01/' # where to save ckpt, images, video, logs
    timestamp = datetime.now().strftime('%d-%b-%H-%M') # to make sure different log-files each train

    # some shared params by model and data, to avoid define twice
    dataset_type = 'blender'
    no_batching = True  # only take random rays from 1 image at a time
    no_ndc = True

    white_bkgd = True  # set to render synthetic data on a white bkgd (always use for dvoxels)
    is_perturb = True  # set to 0. for no jitter, 1. for jitter
    use_viewdirs = True  # use full 5D input instead of 3D
    N_rand_per_sampler = 1024 * 4  # how many N_rand in get_item() function
    lindisp = False  # sampling linearly in disparity rather than depth
    N_samples = 64  # number of coarse samples per ray

    # resume_from = os.path.join(work_dir, 'latest.pth')
    # load_from = os.path.join(work_dir, 'latest.pth')

    ```

* model
    define network structure, a network is usually composed of embedder, mlp and render.
    ```python
    model = dict(
        type='NerfNetwork', # network class name
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


* data
    define network structure, a network is usually composed of embedder, mlp and render.
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
