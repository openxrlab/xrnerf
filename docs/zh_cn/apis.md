# APIS
## run_nerf
input: args, 运行python文件时的命令行参数
purpose: 解析命令行参数，并根据参数训练/测试/渲染一个nerf模型

## train_nerf
input: cfg, mmcv.Config
purpose: args, 运行python文件时的命令行参数, 训练一个nerf模型

## test_nerf
input: cfg, mmcv.Config
purpose: args, 运行python文件时的命令行参数, 测试/渲染一个nerf模型

## parse_args
input: args, 运行python文件时的命令行参数
purpose: 解析命令行参数
