from __future__ import print_function
import os
import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
import argparse
import torch.utils.data as data
import logging
from data import PalmDatasetv1, preproc,cfg_blaze,detection_collate
from layers.modules import MultiBoxLoss
from layers.modules.custom_loss2 import MultiBoxLoss as CustomLoss2
from layers.functions.prior_box import GenerateAnchor
import time
import datetime
import math
# from models.blaze_palm import BlazePalm
from models.blaze_palml_new import BlazePalm
from collections import OrderedDict
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

parser = argparse.ArgumentParser(description='Retinaface Training')
parser.add_argument('--training_dataset', default=r'E:\pytorch_QAT\BlazePalm_new_join\image', help='Training dataset directory')
parser.add_argument('--network', default='blazepalm', help='Backbone')
parser.add_argument('--num_workers', default=0, type=int, help='Number of workers used in dataloading')
parser.add_argument('--lr', '--learning-rate', default=1e-5, type=float, help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--resume_net', default=r"E:\pytorch_QAT\BlazePalm_new_join\weights\blazepalm_Final_cmm.pth", help='resume net for retraining')
parser.add_argument('--resume_epoch', default=0, type=int, help='resume iter for retraining')
parser.add_argument('--weight_decay', default=5e-4, type=float, help='Weight decay for SGD')
parser.add_argument('--gamma', default=0.1, type=float, help='Gamma update for SGD')
parser.add_argument('--save_folder', default='./weights/', help='Location to save checkpoint models')
args = parser.parse_args()

if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)
cfg = None
if args.network == "blazepalm":
    cfg = cfg_blaze

def train(args):

    #定义一些变量
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
    #1. 定义模型
    net = BlazePalm()

    #2. 加载模型数据权重
    if args.resume_net is not None:
        print('Loading resume network from:',args.resume_net)
        state_dict = torch.load(args.resume_net)
        
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            head = k[:7]
            if head == 'module.':
                name = k[7:] # remove `module.`
            else:
                name = k
            new_state_dict[name] = v
        print(f"{len(new_state_dict)}/{len(net.state_dict().items())} params loaded")
        net.load_state_dict(new_state_dict)

    #3. 定义anchor 生成类,取消梯度回传,并copy至cuda
    anchors_generator = GenerateAnchor(cfg)
    with torch.no_grad():
        anchors = anchors_generator.forward()
        anchors = anchors.cuda()
    print("priors.shape:",anchors.shape) #torch.Size([16800, 4])
 
    #4. 是否多GPU训练
    if num_gpu > 1 and gpu_train:
        net = torch.nn.DataParallel(net).cuda()
    else:
        net = net.cuda()
    cudnn.benchmark = True

    #5. 定义优化器
    # optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=momentum, weight_decay=weight_decay)
    optimizer = optim.AdamW(net.parameters(), lr=args.lr, betas=(0.9, 0.999), eps= 1e-8, weight_decay=weight_decay)
    # optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=weight_decay)
    #6.  定义anchor相关的损失函数
    num_classes = 1
    # criterion = MultiBoxLoss(num_classes, 0.55, True, 0, True, 7, 0.35, False)
    criterion = CustomLoss2(cfg, 0.35, True, 0, True, 7, 0.35, False)
    #7. 设置网络为训练模型
    net.train()

    #8. 定义数据集
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
            batch_iterator = iter(data.DataLoader(dataset, batch_size, shuffle=True, num_workers=num_workers, collate_fn=detection_collate))
            if (epoch % 10 == 0 and epoch > 0) or (epoch % 5 == 0 and epoch > cfg['decay1']):
                torch.save(net.state_dict(), save_folder + cfg['name']+ '_epoch_' + str(epoch) + '.pth')
            epoch += 1

        load_t0 = time.time()
        if iteration in stepvalues:
            step_index += 1
        lr = adjust_learning_rate(optimizer,args.lr,gamma, epoch, step_index, iteration, epoch_size)

        # load train data
        images, targets = next(batch_iterator)
        # print("images.shape:",images.shape) #torch.Size([1, 3, 192, 192])
        # for target in targets:
        #     print("target.shape:",target.shape) #torch.Size([obj_num_per_image, 19])
        images = images.cuda()
        targets = [anno.cuda() for anno in targets]
        # print("images:", images)
        # forward
        out = net(images)
        # print("out[0]:", out[0])
        # print("out[1]:", out[1])
        # backprop
        optimizer.zero_grad()
        loss_l, loss_c, loss_landm = criterion(out, anchors, targets)
        loss = (loss_l + loss_c + 5*loss_landm)
        # loss =loss_c
        loss.backward(retain_graph=True)
        optimizer.step()
        load_t1 = time.time()
        batch_time = load_t1 - load_t0
        eta = int(batch_time * (max_iter - iteration))
        print('Epoch:{}/{} || Epochiter: {}/{} || Iter: {}/{} || Loc: {:.4f} Cla: {:.4f} Landm: {:.4f} || LR: {:.8f} || Batchtime: {:.4f} s || ETA: {}'
              .format(epoch, max_epoch, (iteration % epoch_size) + 1,
              epoch_size, iteration + 1, max_iter, loss_l.item(), loss_c.item(), loss_landm.item(), lr, batch_time, str(datetime.timedelta(seconds=eta))))

    torch.save(net.state_dict(), save_folder + cfg['name'] + '_Final.pth')
    # torch.save(net.state_dict(), save_folder + 'Final_Retinaface.pth')


def adjust_learning_rate(optimizer,initial_lr, gamma, epoch, step_index, iteration, epoch_size):
    """Sets the learning rate
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    warmup_epoch = -1
    if epoch <= warmup_epoch:
        lr = 1e-6 + (initial_lr-1e-6) * iteration / (epoch_size * warmup_epoch)
    else:
        lr = initial_lr * (gamma ** (step_index))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

if __name__ == '__main__':
    train(args)
