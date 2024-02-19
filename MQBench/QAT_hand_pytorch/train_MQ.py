from __future__ import print_function
import os
import warnings

import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
import argparse
import torch.utils.data as data

from data import PalmDatasetv1, cfg_blaze, detection_collate

from layers_QAT.modules.custom_loss2 import MultiBoxLoss as CustomLoss2
from layers_QAT.functions.prior_box import GenerateAnchor
import time
import datetime
import math
from models_QAT.blaze_palml_new_QAT import BlazePalm_QAT, ConvTF
from collections import OrderedDict
import torchvision
import torch
import torch.utils.data
from absl import logging
import logging
from mqbench.prepare_by_platform import prepare_by_platform, BackendType
from mqbench.convert_deploy import convert_deploy
from mqbench.utils.state import enable_calibration, enable_quantization
from torch import optim

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description='Retinaface Training')
parser.add_argument('--training_dataset', default=r"E:\datasets\hand_datasets\handpose_x\handpose_datasets_v1-2021-01-31\val_hand_v2",
                    help='Training dataset directory')
parser.add_argument('--network', default='BlazePalm_QAT' , help='Backbone')
parser.add_argument('--num_workers', default=0, type=int, help='Number of workers used in dataloading')
parser.add_argument('--lr', '--learning-rate', default=1e-5, type=float, help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--resume_net', default=r"E:\pytorch_QAT\BlazePalm_new_join\weights\blazepalm_Final_cmm.pth",
                    help='resume net for retraining')
parser.add_argument('--resume_epoch', default=0, type=int, help='resume iter for retraining')
parser.add_argument('--weight_decay', default=5e-4, type=float, help='Weight decay for SGD')
parser.add_argument('--gamma', default=0.1, type=float, help='Gamma update for SGD')
parser.add_argument('--save_folder', default=r'E:\pytorch_QAT\QAT_hand_pytorch\weights', help='Location to save checkpoint models')
args = parser.parse_args()

if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)
cfg = None
if args.network == "BlazePalm_QAT":
    cfg = cfg_blaze

def adjust_learning_rate(optimizer, initial_lr, gamma, epoch, step_index, iteration, epoch_size):
    """Sets the learning rate
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    warmup_epoch = -1
    if epoch <= warmup_epoch:
        lr = 1e-6 + (initial_lr - 1e-6) * iteration / (epoch_size * warmup_epoch)
    else:
        lr = initial_lr * (gamma ** (step_index))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def train(args):
    # 定义一些变量
    num_gpu = cfg['ngpu']
    batch_size = cfg['batch_size']
    max_epoch = cfg['epoch']
    gpu_train = cfg['gpu_train']

    num_workers = args.num_workers
    momentum = args.momentum
    weight_decay = args.weight_decay
    # initial_lr = args.lr
    gamma = args.gamma
    training_dataset = args.training_dataset
    save_folder = args.save_folder
    # 1. 定义模型
    net = BlazePalm_QAT()
    # dummy_input = torch.randn(1, 3, 192, 192)
    # dummy_input = dummy_input.to('cpu')

    # 2. 加载模型数据权重
    # if args.resume_net is not None:
    #     print('Loading resume network from:', args.resume_net)
    #     state_dict = torch.load(args.resume_net)
    pre_train = True
    if pre_train == True:
        state_dict = torch.load(args.resume_net)

        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            head = k[:7]
            if head == 'module.':
                name = k[7:]  # remove `module.`
            else:
                name = k
            new_state_dict[name] = v
        print(f"{len(new_state_dict)}/{len(net.state_dict().items())} params loaded")
        net.load_state_dict(new_state_dict)

    # 3. 定义anchor 生成类,取消梯度回传,并copy至cuda
    anchors_generator = GenerateAnchor(cfg)
    with torch.no_grad():
        anchors = anchors_generator.forward()
        #anchors = anchors.cuda()
        anchors = anchors.cpu()
    #print("priors.shape:", anchors.shape)  # torch.Size([16800, 4])

    #4. 是否多GPU训练
    if num_gpu > 1 and gpu_train:
        net = torch.nn.DataParallel(net).cuda()
    else:
        net = net.cuda()
    cudnn.benchmark = True
    # net = net.cpu()
    # 5. 定义优化器
    # optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=momentum, weight_decay=weight_decay)
    optimizer = optim.AdamW(net.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=weight_decay)
    # optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=weight_decay)
    # 6.  定义anchor相关的损失函数
    criterion = CustomLoss2(cfg, 0.35, True, 0, True, 7, 0.35, False)
    # 7. 设置网络为训练模型
    net.train()
    leaf_model = ConvTF####将convTF提出来
    extra_qconfig_dict = {
        'w_observer': 'MinMaxObserver',
        'a_observer': 'EMAMinMaxObserver',
        'w_fakequantize': 'LearnableFakeQuantize',
        'a_fakequantize': 'LearnableFakeQuantize',
    }
    prepare_custom_config_dict = {'extra_qconfig_dict': extra_qconfig_dict,'leaf_model': leaf_model}
    quan_net = prepare_by_platform(net, BackendType.Tensorrt, prepare_custom_config_dict)
    #print(quan_net.code)
    enable_calibration(quan_net)
    # quan_net(dummy_input)
    enable_quantization(quan_net)



    # 8. 定义数据集
    epoch = 0 + args.resume_epoch
    print('Loading Dataset...')
    dataset = PalmDatasetv1(training_dataset)
    logging.info("finish dataset load")
    epoch_size = math.ceil(len(dataset) / batch_size)
    max_iter = max_epoch * epoch_size

    stepvalues = (cfg['decay1'] * epoch_size, cfg['decay2'] * epoch_size)
    step_index = 0

    if args.resume_epoch > 0:
        start_iter = args.resume_epoch * epoch_size
    else:
        start_iter = 0

    logging.info(f"start_iter:{start_iter} max_iter:{max_iter}")
    for iteration in range(start_iter, max_iter):
        if iteration % epoch_size == 0:
            # create batch iterator
            batch_iterator = iter(data.DataLoader(dataset, batch_size, shuffle=True, num_workers=num_workers,
                                                  collate_fn=detection_collate))
            if (epoch % 2 == 0 and epoch > 0) or (epoch % 2 == 0 and epoch > cfg['decay1']):
                torch.save(quan_net.state_dict(), save_folder + cfg['name'] + '_epoch_' + str(epoch) + '.pth')
            epoch += 1

        load_t0 = time.time()
        if iteration in stepvalues:
            step_index += 1
        lr = adjust_learning_rate(optimizer, args.lr, gamma, epoch, step_index, iteration, epoch_size)

        # load train data
        images, targets = next(batch_iterator)
        # print("images.shape:",images.shape) #torch.Size([1, 3, 192, 192])
        # for target in targets:
        #     print("target.shape:",target.shape) #torch.Size([obj_num_per_image, 19])
        images = images.to(device)
        targets = [anno.cpu() for anno in targets]

        out = quan_net(images)

        optimizer.zero_grad()
        loss_l, loss_c, loss_landm = criterion(out, anchors, targets)
        loss = (loss_l + loss_c + 5 * loss_landm)

        loss.backward(retain_graph=True)
        optimizer.step()
        load_t1 = time.time()
        batch_time = load_t1 - load_t0
        eta = int(batch_time * (max_iter - iteration))
        print(
            'Epoch:{}/{} || Epochiter: {}/{} || Iter: {}/{} || Loc: {:.4f} Cla: {:.4f} Landm: {:.4f} || LR: {:.8f} || Batchtime: {:.4f} s || ETA: {}'
            .format(epoch, max_epoch, (iteration % epoch_size) + 1,
                    epoch_size, iteration + 1, max_iter, loss_l.item(), loss_c.item(), loss_landm.item(), lr,
                    batch_time, str(datetime.timedelta(seconds=eta))))

    torch.save(quan_net.state_dict(), save_folder + cfg['name'] + '_Final.pth')
    quan_net.eval()
    convert_deploy(quan_net, BackendType.Tensorrt, {'x': [1, 3, 192, 192]}, model_name='quant_model1')
    # torch.save(net.state_dict(), save_folder + 'Final_Retinaface.pth')





if __name__ == '__main__':
    train(args)
