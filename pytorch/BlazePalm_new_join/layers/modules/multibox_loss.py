import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from utils.box_utils import match, log_sum_exp
from data import cfg_blaze
GPU = cfg_blaze['gpu_train']




class MultiBoxLoss(nn.Module):
    """SSD Weighted Loss Function
    Compute Targets:
        1) Produce Confidence Target Indices by matching  ground truth boxes
           with (default) 'priorboxes' that have jaccard index > threshold parameter
           (default threshold: 0.5).
        2) Produce localization target by 'encoding' variance into offsets of ground
           truth boxes and their matched  'priorboxes'.
        3) Hard negative mining to filter the excessive number of negative examples
           that comes with using a large number of default bounding boxes.
           (default negative:positive ratio 3:1)
    Objective Loss:
        L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N
        Where, Lconf is the CrossEntropy Loss and Lloc is the SmoothL1 Loss
        weighted by α which is set to 1 by cross val.
        Args:
            c: class confidences,
            l: predicted boxes,
            g: ground truth boxes
            N: number of matched default boxes
        See: https://arxiv.org/pdf/1512.02325.pdf for more details.
    """

    def __init__(self, num_classes, overlap_thresh, prior_for_matching, bkg_label, neg_mining, neg_pos, neg_overlap, encode_target):
        super(MultiBoxLoss, self).__init__()
        self.num_classes = num_classes
        self.threshold = overlap_thresh
        self.background_label = bkg_label
        self.encode_target = encode_target
        self.use_prior_for_matching = prior_for_matching
        self.do_neg_mining = neg_mining
        self.negpos_ratio = neg_pos
        self.neg_overlap = neg_overlap
        self.variance = [0.1, 0.2]
        self.landmarks_loss = WingLoss()
        # self.x_scale = 192.0
        # self.y_scale = 192.0
        # self.w_scale = 192.0
        # self.h_scale = 192.0
        # self.num_keypoints = 7

    def focal_loss_sigmoid(self, y_pred, y_true, alpha=0.25, gamma=2):
        y_pred = torch.sigmoid(y_pred)
        y_true = y_true.float()
        y_pred = y_pred.float()
        loss = -y_true * (1 - alpha) * ((1 - y_pred) * gamma) * torch.log(y_pred) - (1 - y_true) * alpha * (
                y_pred ** gamma) * torch.log(1 - y_pred)
        loss = loss.mean(0)
        return loss

    def forward(self, predictions, priors, targets):
        """Multibox Loss
        Args:
            predictions (tuple): A tuple containing loc preds, conf preds,
            and prior boxes from SSD net.
                conf shape: torch.size(batch_size,num_priors,num_classes)
                loc shape: torch.size(batch_size,num_priors,4)
                priors shape: torch.size(num_priors,4)

            ground_truth (tensor): Ground truth boxes and labels for a batch,
                shape: [batch_size,num_objs,5] (last idx is the label).
        """

        # loc_data, conf_data, landm_data = predictions
        conf_data,reg_data = predictions
        # detection_boxs = self._decode_boxes(reg_data,priors)
        reg_data = reg_data/192.0
        loc_data = reg_data[:, :, 0:4]
        landm_data = reg_data[:, :, 4:]

        priors = priors  #生成anchor的个数
        batch_size = loc_data.size(0)
        num_priors = (priors.size(0))
        # 将生成的anchor与GT box进行匹配
        loc_t = torch.Tensor(batch_size, num_priors, 4)                         # torch.Size([batch_size, 4])
        landm_t = torch.Tensor(batch_size, num_priors, 14)    # torch.Size([batch_size, 2016, 14])
        conf_t = torch.LongTensor(batch_size, num_priors)                       # torch.Size([batch_size, 2016])
        for idx in range(batch_size): # 遍历
            truths = targets[idx][:, :4].data       #获取当前idx图片 gt的box torch.size([hand_num,4])
            landms = targets[idx][:, 4:-1].data     #获取当前idx图片 gt的landmarks torch.size([hand_num,14])
            labels = targets[idx][:, -1].data       #获取当前idx图片 gt的landmarks torch.size([hand_num,1]) 是手的默认为1
            defaults = priors.data
            match(self.threshold, truths, defaults, self.variance, labels, landms, loc_t, conf_t, landm_t, idx)

        if GPU:
            loc_t = loc_t.cuda()
            conf_t = conf_t.cuda()
            landm_t = landm_t.cuda()

        zeros = torch.tensor(0).cuda()
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

        pos = conf_t > zeros
        conf_t[pos] = 1
        # Localization Loss (Smooth L1)
        # Shape: [batch,num_priors,4]
        pos_idx = pos.unsqueeze(pos.dim()).expand_as(loc_data)
        loc_p = loc_data[pos_idx].view(-1, 4)
        loc_t = loc_t[pos_idx].view(-1, 4)
        loss_l = F.smooth_l1_loss(loc_p, loc_t, reduction='mean')
        
        # # Compute max conf across batch for hard negative mining
        conf_data_o = conf_data.clone()
        batch_conf = conf_data_o.view(-1, batch_size) #self.num_classes)
        loss_c = batch_conf.reshape(-1,1) #log_sum_exp(batch_conf) - batch_conf.gather(1, conf_t.view(-1, 1))

        # # Hard Negative Mining
        loss_c[pos.view(-1, 1)] = -1000000 # filter out pos boxes for now
        loss_c = loss_c.view(batch_size, -1)
        _, loss_idx = loss_c.sort(1, descending=True)
        _, idx_rank = loss_idx.sort(1)
        num_pos = pos.long().sum(1, keepdim=True) 
        num_neg = torch.clamp(self.negpos_ratio*num_pos, max=loss_idx.size(1)-1)
        neg = idx_rank < num_neg.expand_as(idx_rank)

        # # Confidence Loss Including Positive and Negative Examples
        pos_idx = pos.unsqueeze(2).expand_as(conf_data)
        neg_idx = neg.unsqueeze(2).expand_as(conf_data)
        # conf_pp = conf_data[pos_idx.gt(0)]  #
        # conf_pn = conf_data[neg_idx.gt(0)]  #
        conf_p = conf_data[(pos_idx+neg_idx).gt(0)]#self.num_classes)
        # conf_p00 = torch.sigmoid(conf_pp)
        # conf_p11 = torch.sigmoid(conf_pn)
        # conf_p = torch.sigmoid(conf_p)
        # targets_weighted00 = conf_t[pos.gt(0)].float()
        # targets_weighted11 = conf_t[neg.gt(0)].float()
        targets_weighted = conf_t[(pos+neg).gt(0)]
        # loss_c0 = F.binary_cross_entropy(conf_p00, targets_weighted00, reduction="mean")
        # loss_c1 = F.binary_cross_entropy(conf_p11, targets_weighted11, reduction='mean')
        # loss_c = loss_c0 + loss_c1
        # loss_c = F.binary_cross_entropy(conf_p, targets_weighted, reduction='mean')
        loss_c = self.focal_loss_sigmoid(conf_p, targets_weighted, alpha=0.25, gamma=2)

        return loss_l, loss_c, loss_landm



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