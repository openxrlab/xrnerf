# APIS
## run_nerf
input: args, running parameters
purpose: parse running parameters, and train, test or render a nerf model according to specified parameters

## train_nerf
input: cfg, mmcv.Config
purpose: parse running parameters, train a nerf model according to specified parameters

## test_nerf
input: cfg, mmcv.Config
purpose: parse running parameters, test or render a nerf model according to specified parameters

## parse_args
input: args, running parameters
purpose: parse running parameters, convert to a mmcv.Config
