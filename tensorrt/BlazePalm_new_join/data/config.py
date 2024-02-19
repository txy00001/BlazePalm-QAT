cfg_blaze = {
    'name': 'blazepalm',
    'upType': 'deconvolution',
    'min_sizes': [[16, 32], [64, 128], [64, 128],[64, 128]],
    'steps': [8, 16, 16, 16],
    'variance': [0.1, 0.2],
    'clip': False,
    'loc_weight': 1,
    'cls_weight': 6,
    'landm_weight': 0.1,
    'gpu_train': True,
    'batch_size': 48,
    'ngpu': 1,
    'epoch': 500,
    'decay1': 250,
    'decay2': 400,
    'image_size': 192,
    'pretrain': True,
    'return_layers': {'stage1': 1, 'stage2': 2, 'stage3': 3},
    'in_channel': 32,
    'out_channel': 32,
    'num_classes': 1
}


# cfg_blaze = {
#     'name': 'blazepalm',
#     'upType': 'deconvolution',
#     'min_sizes': [[16, 32], [64, 128], [64, 128],[64, 128]],
#     'steps': [8, 16, 16, 16],
#     'variance': [0.1, 0.2],
#     'clip': False,
#     'loc_weight': 1.0,
#     'gpu_train': True,
#     'batch_size': 64,
#     'ngpu': 1,
#     'epoch': 250,
#     'decay1': 190,
#     'decay2': 220,
#     'image_size': 128,
#     'pretrain': True,
#     'return_layers': {'stage1': 1, 'stage2': 2, 'stage3': 3},
#     'in_channel': 32,
#     'out_channel': 64
# }
# cfg_blaze = {
#     'name': 'blazepalm',
#     'upType': 'deconvolution',
#     'min_sizes': [[16, 32], [64, 128], [256, 512]],
#     'steps': [8, 16, 16, 16],
#     'variance': [0.1, 0.2],
#     'clip': False,
#     'loc_weight': 2.0,
#     'gpu_train': True,
#     'batch_size': 1,
#     'ngpu': 1,
#     'epoch': 250,
#     'decay1': 190,
#     'decay2': 220,
#     'image_size': 192,
#     'pretrain': True,
#     'return_layers': {'stage1': 1, 'stage2': 2, 'stage3': 3},
#     'in_channel': 32,
#     'out_channel': 64
# }