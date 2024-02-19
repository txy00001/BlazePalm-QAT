cfg_blaze = {
    'name': '  BlazePalm_QAT',
    'upType': 'deconvolution',
    'min_sizes': [[16, 32], [64, 128], [64, 128],[64, 128]],
    'steps': [8, 16, 16, 16],
    'variance': [0.1, 0.2],
    'clip': False,
    'loc_weight': 1,
    'cls_weight': 6,
    'landm_weight': 0.1,
    'gpu_train': False,
    'batch_size': 32,
    'ngpu': 1,
    'epoch': 1,
    'decay1': 1,
    'decay2': 1,
    'image_size': 192,
    'pretrain': True,
    'return_layers': {'stage1': 1, 'stage2': 2, 'stage3': 3},
    'in_channel': 32,
    'out_channel': 32,
    'num_classes': 1
}



