# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy import ndimage

from layers_QAT.loss.gfocal_loss import QualityFocalLoss
from utils.box_utils import match, log_sum_exp

import math

# torch.log  and math.log is e based
class WingLoss(nn.Module):
    def __init__(self, omega=10, epsilon=2):
        super(WingLoss, self).__init__()
        self.omega = omega
        self.epsilon = epsilon

    def forward(self, pred, target):
        y = target
        y_hat = pred
        delta_y = (y - y_hat).abs()
        delta_y1 = delta_y[delta_y < self.omega]
        delta_y2 = delta_y[delta_y >= self.omega]
        loss1 = self.omega * torch.log(1 + delta_y1 / self.epsilon)
        C = self.omega - self.omega * math.log(1 + self.omega / self.epsilon)
        loss2 = delta_y2 - C
        return (loss1.sum() + loss2.sum()) / (len(loss1) + len(loss2))


class MultiBoxLoss(nn.Module):


    def __init__(self, cfg, overlap_thresh, prior_for_matching,bkg_label, neg_mining, neg_pos, neg_overlap, encode_target):
        super(MultiBoxLoss, self).__init__()
        self.use_gpu = True
        self.num_classes = cfg['num_classes']
        self.threshold = overlap_thresh
        self.background_label = bkg_label
        self.encode_target = encode_target
        self.use_prior_for_matching = prior_for_matching
        self.do_neg_mining = neg_mining
        self.negpos_ratio = neg_pos
        self.neg_overlap = neg_overlap
        self.variance = cfg['variance']
        self.loc_weight = cfg['loc_weight']
        self.cls_weight = cfg['cls_weight']
        self.landm_weight = cfg['landm_weight']
        self.landmarks_loss = WingLoss()

    def focal_loss_sigmoid(self, y_pred, y_true, alpha=0.25, gamma=2):
        y_pred = torch.sigmoid(y_pred)
        y_true = y_true.float()
        y_pred = y_pred.float()
        loss = -y_true * (1 - alpha) * ((1 - y_pred) * gamma) * torch.log(y_pred) - (1 - y_true) * alpha * (
                y_pred ** gamma) * torch.log(1 - y_pred)
        loss = loss.mean(0)
        return loss

    def forward(self, predictions, priors, targets):

        conf_data,reg_data = predictions
        reg_data = reg_data/192.0
        loc_data = reg_data[:,:,0:4]
        landm_data = reg_data[:,:,4:]

        priors = priors
        num = loc_data.size(0)
        num_priors = (priors.size(0))

        # match priors (default boxes) and ground truth boxes
        loc_t = torch.Tensor(num, num_priors, 4)
        landm_t = torch.Tensor(num, num_priors, 14)
        conf_t = torch.LongTensor(num, num_priors)
        for idx in range(num):
            truths = targets[idx][:, :4].data  # 获取当前idx图片 gt的box torch.size([hand_num,4])
            landms = targets[idx][:, 4:-1].data  # 获取当前idx图片 gt的landmarks torch.size([hand_num,14])
            labels = targets[idx][:, -1].data  # 获取当前idx图片 gt的landmarks torch.size([hand_num,1]) 是手的默认为1
            defaults = priors.data
            match(self.threshold, truths, defaults, self.variance,
                     labels, landms, loc_t, conf_t, landm_t, idx)
        if self.use_gpu:
            loc_t = loc_t.cuda()
            conf_t = conf_t.cuda()
            landm_t = landm_t.cuda()

        zeros = torch.tensor(0).cuda()
        nagative_1 = torch.tensor(-1).cuda()
        # landm Loss (Smooth L1)
        # Shape: [batch,num_priors,10]
        pos1 = conf_t > zeros
        num_pos_landm = pos1.long().sum(1, keepdim=True)
        N1 = max(num_pos_landm.data.sum().float(), 1)
        pos_idx1 = pos1.unsqueeze(pos1.dim()).expand_as(landm_data)
        landm_p = landm_data[pos_idx1].view(-1, 14)
        landm_t = landm_t[pos_idx1].view(-1, 14)
        # loss_landm = F.smooth_l1_loss(landm_p, landm_t, reduction='mean')
        loss_landm = self.landmarks_loss(landm_p, landm_t)

        pos = conf_t != zeros
        conf_t[pos] = 1

        # Localization Loss (Smooth L1)
        # Shape: [batch,num_priors,4]
        pos_idx = pos.unsqueeze(pos.dim()).expand_as(loc_data)
        loc_p = loc_data[pos_idx].view(-1, 4)
        loc_t = loc_t[pos_idx].view(-1, 4)
        loss_l = QualityFocalLoss(loc_p, loc_t, reduction='mean')

        # Focal loss
        # Sum of losses: L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N
        pos_neg = conf_t >= nagative_1
        batch_conf = conf_data[pos_neg].squeeze()
        epoch = 300
        if epoch <= 5:
           loss_c = nn.SmoothL1Loss()
        else:
           loss_c = self.focal_loss_sigmoid(batch_conf, conf_t[pos_neg], alpha=0.25, gamma=2)

        # loss_c = self.focalloss(batch_conf, conf_t[pos_neg])
        # loss_c *= self.cls_weight



        return loss_l, loss_c, loss_landm