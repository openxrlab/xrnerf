
# # model settings
# model = dict(
#     type='nerf',
#     i_embed=0, # set 0 for default positional encoding, -1 for none
#     multires=10, # log2 of max freq for positional encoding (3D location)
#     multires_views=4, # log2 of max freq for positional encoding (2D direction)
#     use_viewdirs=True, # use full 5D input instead of 3D
#     N_importance=0, # number of additional fine samples per ray
#     netdepth=8, # layers in network
#     netwidth=256, # channels per layer
#     netdepth_fine=8, # layers in fine network
#     netwidth_fine=256, # channels per layer in fine network
#     netchunk=1024*64, # number of pts sent through network in parallel, decrease if running out of memory
#     )
