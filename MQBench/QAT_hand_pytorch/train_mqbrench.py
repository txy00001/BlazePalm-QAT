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
from models_QAT.blaze_palml_new_QAT import BlazePalm_QAT
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
import mqbench



os.environ['CUDA_VISIBLE_DEVICES'] = '0'

device = torch.device( "cpu")

parser = argparse.ArgumentParser(description=' Training')
parser.add_argument('--training_dataset',default=r"E:\datasets\hand_datasets\handpose_x\handpose_datasets_v1-2021-01-31\val_hand_v2",help='Training dataset directory')
parser.add_argument('--val_dataset',default=r"E:\datasets\hand_datasets\handpose_x\handpose_datasets_v1-2021-01-31\val_hand_v2",help='val dataset directory')
parser.add_argument('--network', default='BlazePalm_QAT', help='Backbone')
parser.add_argument('--num_workers', default=4, type=int, help='Number of workers used in dataloading')
parser.add_argument('--lr', '--learning-rate', default=1e-5, type=float, help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--resume_net', default=r"E:\pytorch_QAT\BlazePalm_new_join\weights\blazepalm_Final_cmm.pth",help='resume net for retraining')
parser.add_argument('--resume_epoch_train', default=0, type=int, help='resume iter for retraining')
parser.add_argument('--resume_epoch_val', default=0, type=int, help='resume iter for retraining')
parser.add_argument('--weight_decay', default=5e-4, type=float, help='Weight decay for SGD')
parser.add_argument('--gamma', default=0.1, type=float, help='Gamma update for SGD')
parser.add_argument('--save_folder', default=r'E:\pytorch_QAT\QAT_hand_pytorch\weights', help='Location to save checkpoint models')
args = parser.parse_args()

def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger


def checkpoint_load(model, name):
    print('Restoring checkpoint: {}'.format(name))
    model.load_state_dict(torch.load(name, map_location='gpu'))
    epoch = int(os.path.splitext(os.path.basename(name))[0].split('-')[1])
    return epoch

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


if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)
cfg = None
if args.network == "BlazePalm_QAT":
    cfg = cfg_blaze

def train(model, criterion, optimizer, train_loader):
    model.train()
    for i, (image, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()


# 定义一些变量
num_gpu = cfg['ngpu']###gpu/cpu在qat（mq）的区别
train_batch_size = cfg['train_batch_size']
val_batch_size = cfg['val_batch_size']
max_epoch = cfg['train_epoch']
gpu_train = cfg['gpu_train']


num_workers = args.num_workers
weight_decay = args.weight_decay

gamma = args.gamma
training_dataset = args.training_dataset
val_dataset = args.val_dataset
save_folder = args.save_folder

# 1. 定义模型和输入
net = BlazePalm_QAT()
dummy_input = torch.randn(1, 3, 192, 192, device = torch.device("cpu"))
# 2. 加载模型数据权重
pre_train = False
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
   # anchors = anchors.cuda()

# 4. 是否多GPU训练
# if num_gpu > 1 and gpu_train:
#     net = torch.nn.DataParallel(net).cuda()
# else:
#     net = net.cuda()
# cudnn.benchmark = True
# 5. 定义优化器
optimizer = optim.AdamW(net.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=weight_decay)
# 6.  定义anchor相关的损失函数
criterion = CustomLoss2(cfg, 0.35, True, 0, True, 7, 0.35, False)

# 7. 设置数据集
#7.1训练数据集
train_epoch = 0 + args.resume_epoch_train
dataset_train = PalmDatasetv1(training_dataset)
logging.info("finish dataset load")
epoch_size_train = math.ceil(len(dataset_train) / train_batch_size)
train_dataloader = torch.utils.data.DataLoader(dataset_train, train_batch_size,shuffle=True, num_workers=0,drop_last=True,collate_fn=detection_collate)
#7.2评估数据集
val_epoch = 0 + args.resume_epoch_val
dataset_val = PalmDatasetv1(val_dataset)
epoch_size_val = math.ceil(len(dataset_val) / val_batch_size)
val_dataloader = torch.utils.data.DataLoader(dataset_val, val_batch_size,shuffle=True, num_workers=0,drop_last=True,collate_fn=detection_collate)

#8.将模型转换为QAT模型

# net.qconfig = torch.quantization.get_default_qat_qconfig('qnnpack')
# quan_net = torch.quantization.prepare_qat(net, inplace=True)
#9定义训练模型
net.train()
# net.train().to(torch.device('cpu'))

#10选择量化方法，初始化MQBench量化器
extra_qconfig_dict = {
    'w_observer': 'MinMaxObserver',
    'a_observer': 'EMAMinMaxObserver',
    'w_fakequantize': 'LearnableFakeQuantize',
    'a_fakequantize': 'LearnableFakeQuantize',
}
prepare_custom_config_dict = {'extra_qconfig_dict': extra_qconfig_dict}
quan_net = prepare_by_platform(net, BackendType.Tensorrt, prepare_custom_config_dict)
#print(quan_net)
enable_calibration(quan_net)
quan_net(dummy_input)
quantizer = enable_quantization(quan_net)

# quant_config = mqbench.QuantConfig(weight_bits=8, act_bits=8, quant_method='back-gate')
# quantizer = mqbench.QATQuantizer(quan_net, quant_config)

#11 进行QAT训练
num_epochs = 200
for epoch in range(num_epochs):
    for i, (images, targets) in enumerate(train_dataloader):

        quantizer.set_epoch(epoch)
        quantizer.update()

        # forward pass
        images = images.cuda()
        targets = [anno.cuda() for anno in targets]
        logger = get_logger('/log')
        logger.info('start training!')

        out = quan_net(images.float())
        optimizer.zero_grad()

        loss_l, loss_c, loss_landm = criterion(out, anchors, targets)
        loss = (loss_l + loss_c + 5 * loss_landm)
        loss.backward(retain_graph=True)
        optimizer.step()

        # ready the quantizer
        quantizer.ready()
        print('loss: {:.4f}'.format(loss.item()))

##12.evaluate the model
    with torch.no_grad():
        correct = 0
        total = 0
        quan_net.eval()
        for images, targets in val_dataloader:
            images = images.cuda()
            targets = [anno.cuda() for anno in targets]
            out = quan_net(images.float())
            _, predicted = torch.max(out.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
        accuracy = 100 * correct / total
        print("Epoch {0}: Test Accuracy={1}".format(epoch + 1, accuracy))

 #13.将模型转换为量化模型，并保存
        quant_model = torch.quantization.convert(quan_net.eval(), inplace=False)
        if (train_epoch % 2 == 0 and train_epoch > 0):
           torch.save(quant_model.state_dict(), save_folder + cfg['name'] + 'train_epoch_' + str(train_epoch ) + '_Final.pth')
           epoch += 1
        if (val_epoch % 2 == 0 and val_epoch > 0):
           torch.save(quant_model.state_dict(), save_folder + cfg['name'] + 'val_epoch_' + str(train_epoch ) + '_Final.pth')
           epoch += 1

#14.转为onnx输出
        quant_model.eval()
        convert_deploy(quant_model, BackendType.Tensorrt, {'x': [1, 3, 192, 192]}, model_name='quant_net.onnx')###以tensorr为后端进行输出























