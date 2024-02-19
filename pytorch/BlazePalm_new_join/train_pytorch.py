from __future__ import print_function

import copy
import os
import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
import argparse
import torch.utils.data as data
import logging

from torch.quantization import prepare_qat, fuse_modules
from torch.quantization.qconfig import get_default_qat_qconfig

from data import PalmDatasetv1, preproc, cfg_blaze, detection_collate
from layers.modules import MultiBoxLoss
from layers.modules.custom_loss2 import MultiBoxLoss as CustomLoss2
#from layers_QAT import MultiBoxLoss as CustomLoss2
from layers.functions.prior_box import GenerateAnchor
import time
import datetime
import math
# from models.blaze_palm import BlazePalm
from models.blaze_palml_new import BlazePalm
from collections import OrderedDict
import torchvision
import torch
import torch.utils.data
from absl import logging

# from pytorch_quantization import nn as quant_nn
# from pytorch_quantization import calib
# from pytorch_quantization.tensor_quant import QuantDescriptor

from models_QAT.QAT_net_pytorch import BlazePalm_QAT,GetKeysDict

logging.set_verbosity(logging.FATAL)  # Disable logging as they are too noisy in notebook
# from pytorch_quantization import quant_modules
# from torch.quantization import prepare_qat, get_default_qat_qconfig, convert, fuse_modules
#
# quant_nn.TensorQuantizer.use_fb_fake_quant = True
#
# quant_modules.initialize()

# os.environ['CUDA_VISIBLE_DEVICES'] = '0'

parser = argparse.ArgumentParser(description='Retinaface Training')
parser.add_argument('--training_dataset',
                    default=r'D:\Datas\train_v2',
                    help='Training dataset directory')
parser.add_argument('--network', default='BlazePalm_QAT', help='Backbone')
parser.add_argument('--num_workers', default=1, type=int, help='Number of workers used in dataloading')
parser.add_argument('--lr', '--learning-rate', default=1e-5, type=float, help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--resume_net', default=r".\weights\BlazePalm_QAT_epoch_10.pth",
                    help='resume net for retraining')
parser.add_argument('--resume_epoch', default=0, type=int, help='resume iter for retraining')
parser.add_argument('--weight_decay', default=5e-4, type=float, help='Weight decay for SGD')
parser.add_argument('--gamma', default=0.1, type=float, help='Gamma update for SGD')
parser.add_argument('--save_folder', default='./weights/', help='Location to save checkpoint models')
args = parser.parse_args()

if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)
cfg = None
if args.network == "BlazePalm_QAT":
    cfg = cfg_blaze


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
    # 1. 定义模型(加入量化算

    net = BlazePalm_QAT()

    # 2. 加载模型数据权重
    # if args.resume_net is not None:
    #     print('Loading resume network from:', args.resume_net)
    #     state_dict = torch.load(args.resume_net)
    palm_key = GetKeysDict().hand_palm_dict
    pre_train = True
    if pre_train == True:
        print('Loading resume network from:', args.resume_net)
        state_dict = torch.load(args.resume_net)
        new_state_dict = OrderedDict()
        id=0
        for k, v in state_dict.items():
            head = k[:7]
            if head == 'module.':
                name = k[7:]  # remove `module.`
            else:
                name = palm_key[id]
                # name = k
                id=id+1
            new_state_dict[name] = v
        print(f"{len(new_state_dict)}/{len(net.state_dict().items())} params loaded")
        net.load_state_dict(new_state_dict)



    # 3. 定义anchor 生成类,取消梯度回传,并copy至cuda
    anchors_generator = GenerateAnchor(cfg)
    with torch.no_grad():
        anchors = anchors_generator.forward()
        # anchors = anchors.cpu()
        anchors = anchors.cuda()
    print("priors.shape:", anchors.shape)  # torch.Size([16800, 4])

    # # 4. 是否多GPU训练
    if num_gpu > 1 and gpu_train:
        net = torch.nn.DataParallel(net).cuda()
    else:
        net = net.cuda()
    cudnn.benchmark = True

    # 5. 定义优化器
    # optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=momentum, weight_decay=weight_decay)
    optimizer = optim.AdamW(net.parameters(), lr=args.lr, betas=(0.9, 0.999), eps= 1e-8, weight_decay=weight_decay)
    # optimizer = optim.Adam(net.parameters(), lr=1e-5, weight_decay=weight_decay)
    # 6.  定义anchor相关的损失函数
    # num_classes = 1
    # criterion = CustomLoss2(num_classes, 0.55, True, 0, True, 7, 0.35, False)
    criterion = CustomLoss2(cfg, 0.35, True, 0, True, 7, 0.35, False)
    # 7. 设置网络为训练模型
    net.train()

    net.qconfig = get_default_qat_qconfig(backend='fbgemm') # qnnpack  or fbgemm

    MovingAverageMinMaxObserver = torch.quantization.observer.MovingAverageMinMaxObserver
    net.qconfig = torch.quantization.QConfig(
        activation=torch.quantization.fake_quantize.FakeQuantize.with_args(observer=MovingAverageMinMaxObserver,
                                                                           quant_min=-128, quant_max=127,
                                                                           dtype=torch.qint8,
                                                                           qscheme=torch.per_tensor_symmetric,
                                                                           reduce_range=False),
        weight=torch.quantization.fake_quantize.FakeQuantize.with_args(observer=MovingAverageMinMaxObserver,
                                                                       quant_min=-128, quant_max=127, dtype=torch.qint8,
                                                                       qscheme=torch.per_tensor_symmetric,
                                                                       reduce_range=False))

    # net.to('cpu')##将输入数据移到CPU上

    # assert net.training  # 检查是否处于训练状态
    # fuse_modules(net,inplace=True)

    # Step4：插入伪量化模块,替换为插入了伪定点算子的新模型
    prepare_qat(net, inplace=True)
    print("model with quant:",net)

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
            if (epoch % 10 == 0 and epoch > 0) or (epoch % 5 == 0 and epoch > cfg['decay1']):
                torch.save(net.state_dict(), save_folder + cfg['name'] + '_epoch_' + str(epoch) + '.pth')
            epoch += 1

        load_t0 = time.time()
        if iteration in stepvalues:
            step_index += 1
        lr = adjust_learning_rate(optimizer, args.lr, gamma, epoch, step_index, iteration, epoch_size)

        # load train data
        images, targets = next(batch_iterator)
        # print("images.shape:",images.shape) #torch.Size([1, 3, 192, 192])
        # images = images.cpu()
        # targets = [anno.cpu() for anno in targets]
        images = images.cuda()
        targets = [anno.cuda() for anno in targets]
        # forward
        out = net(images.float())
        # backprop
        optimizer.zero_grad()
        loss_l, loss_c, loss_landm = criterion(out, anchors, targets)
        loss = (loss_l + loss_c + 5 * loss_landm)
        # loss =loss_c
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
        if iteration % epoch_size == 0:
            if epoch % 1 == 0:
                export_onnx(net,"quant_pyorch_model.onnx",images[0])
                print("save success")

    torch.save(net.state_dict(), save_folder + cfg['name'] + '_2Final.pth')


def export_onnx(net, onnx_filename,image):

    dummy_input = image.float().unsqueeze(0)
    model=copy.deepcopy(net)
    model.apply(torch.quantization.enable_observer)
    model.apply(torch.quantization.enable_fake_quant)

    _ = model(dummy_input)
    for module in model.modules():
        if isinstance(module, torch.quantization.FakeQuantize):
            module.calculate_qparams()
    model.apply(torch.quantization.disable_observer)
    model.cuda()
    input_names = ["input"]
    torch.onnx.export(model, dummy_input, onnx_filename, verbose=False, training=False,input_names=input_names, opset_version=13,do_constant_folding=False)
    # torch.quantization.convert(net.eval().cpu(), inplace=False)###模型转换
    return True


def adjust_learning_rate(optimizer, initial_lr, gamma, epoch, step_index, iteration, epoch_size):
    warmup_epoch = -1
    if epoch <= warmup_epoch:
        lr = 1e-6 + (initial_lr - 1e-6) * iteration / (epoch_size * warmup_epoch)
    else:
        lr = initial_lr * (gamma ** (step_index))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


if __name__ == '__main__':
    train(args)
