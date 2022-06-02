# XRNeRF


## Introduction
更新日志 见doc/update.log

[代码框架](https://www.processon.com/view/link/626766f11e085332a3bf6c5b)


## Environment
cd /mnt/lustre/share/spring
source deactivate 
source s0.3.4
然后安装mmcv，参考[安装教程](https://mmcv.readthedocs.io/zh_CN/latest/get_started/build.html#linux-macos-mmcv)


```pip install -e . --user```

## Train

训练blender数据集中的lego
```
srun --partition=3dmr-sensevideo --mpi=pmi2 \
--job-name=train_generator --kill-on-bad-exit=0 \
--gres=gpu:1 --ntasks-per-node=1 -n1 --cpus-per-task=8 \
python train_nerf.py --config configs/nerfsv2/nerf_lego_base01.py \
2>&1 | tee nerf_lego_base01.log
```


训练llff数据集中的fern
```
srun --partition=3dmr-sensevideo --mpi=pmi2 \
--job-name=train_generator --kill-on-bad-exit=0 \
--gres=gpu:1 --ntasks-per-node=1 -n1 --cpus-per-task=8 \
python train_nerf.py --config configs/nerfsv2/nerf_fern_base01.py \
2>&1 | tee nerf_fern_base01.log
```

## Test

训练blender数据集中的lego
```
srun --partition=3dmr-sensevideo --mpi=pmi2 \
--job-name=train_generator --kill-on-bad-exit=0 \
--gres=gpu:1 --ntasks-per-node=1 -n1 --cpus-per-task=8 \
python train_nerf.py --config configs/nerfsv2/nerf_lego_base01.py \
--test_only \
2>&1 | tee nerf_lego_base01.log
```

## Codes Check
安装
```pip install pre-commit```

检查
```pre-commit run --all-files```



