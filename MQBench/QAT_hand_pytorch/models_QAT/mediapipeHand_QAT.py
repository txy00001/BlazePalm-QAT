import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models_QAT.blazebase_QAT import BlazeLandmark, BlazeBlock


class GetKeysDict:
    def __init__(self):
        self.hand_landmark_dict = {
            # 'conv1.weight':   'model_1/model/conv2d/Conv2D_dequantize',

            'conv1.weight': 'model_1/model/conv2d/Conv2D',
            'conv1.bias': 'model_1/model/batch_normalization/FusedBatchNormV3;model_1/model/depthwise_conv2d_3/depthwise;model_1/model/conv2d/Conv2D',
            'PR_Conv1.PR_Relu.weight': 'model_1/model/p_re_lu/add;model_1/model/p_re_lu/Relu;model_1/model/p_re_lu/Neg_1;model_1/model/p_re_lu/Relu_1;model_1/model/p_re_lu/mul',
            'PR_Conv1.PR_Conv.0.weight': 'model_1/model/depthwise_conv2d/depthwise',
            'PR_Conv1.PR_Conv.0.bias': 'model_1/model/depthwise_conv2d_3/depthwise',
            'PR_Conv1.PR_Conv.1.weight': 'model_1/model/conv2d_1/Conv2D',
            'PR_Conv1.PR_Conv.1.bias': 'model_1/model/batch_normalization_1/FusedBatchNormV3;model_1/model/depthwise_conv2d_3/depthwise;model_1/model/conv2d_1/Conv2D',

            'PR_Conv2.PR_Relu.weight': 'model_1/model/p_re_lu_1/add;model_1/model/p_re_lu_1/Relu;model_1/model/p_re_lu_1/Neg_1;model_1/model/p_re_lu_1/Relu_1;model_1/model/p_re_lu_1/mul',
            'PR_Conv2.PR_Conv.0.weight': 'model_1/model/depthwise_conv2d_1/depthwise',
            'PR_Conv2.PR_Conv.0.bias': 'model_1/model/depthwise_conv2d_3/depthwise',
            'PR_Conv2.PR_Conv.1.weight': 'model_1/model/conv2d_2/Conv2D',
            'PR_Conv2.PR_Conv.1.bias': 'model_1/model/batch_normalization_2/FusedBatchNormV3;model_1/model/depthwise_conv2d_3/depthwise;model_1/model/conv2d_2/Conv2D',

            'PR_Conv3.PR_Relu.weight': 'model_1/model/p_re_lu_2/add;model_1/model/p_re_lu_2/Relu;model_1/model/p_re_lu_2/Neg_1;model_1/model/p_re_lu_2/Relu_1;model_1/model/p_re_lu_2/mul',
            'PR_Conv3.PR_Conv.0.weight': 'model_1/model/depthwise_conv2d_2/depthwise',
            'PR_Conv3.PR_Conv.0.bias': 'model_1/model/depthwise_conv2d_3/depthwise',
            'PR_Conv3.PR_Conv.1.weight': 'model_1/model/conv2d_3/Conv2D',
            'PR_Conv3.PR_Conv.1.bias': 'model_1/model/batch_normalization_3/FusedBatchNormV3;model_1/model/depthwise_conv2d_3/depthwise;model_1/model/conv2d_3/Conv2D',

            'MP_Pad_Conv1.PR_Relu.weight': 'model_1/model/p_re_lu_3/add;model_1/model/p_re_lu_3/Relu;model_1/model/p_re_lu_3/Neg_1;model_1/model/p_re_lu_3/Relu_1;model_1/model/p_re_lu_3/mul',
            'MP_Pad_Conv1.conv.0.weight': 'model_1/model/depthwise_conv2d_3/depthwise1',
            'MP_Pad_Conv1.conv.0.bias': 'model_1/model/depthwise_conv2d_3/depthwise',
            'MP_Pad_Conv1.conv.1.weight': 'model_1/model/conv2d_4/Conv2D',
            'MP_Pad_Conv1.conv.1.bias': 'model_1/model/batch_normalization_4/FusedBatchNormV3;model_1/model/depthwise_conv2d_7/depthwise;model_1/model/conv2d_4/Conv2D',

            'PR_Conv4.PR_Relu.weight': 'model_1/model/p_re_lu_4/add;model_1/model/p_re_lu_4/Relu;model_1/model/p_re_lu_4/Neg_1;model_1/model/p_re_lu_4/Relu_1;model_1/model/p_re_lu_4/mul',
            'PR_Conv4.PR_Conv.0.weight': 'model_1/model/depthwise_conv2d_4/depthwise',
            'PR_Conv4.PR_Conv.0.bias': 'model_1/model/depthwise_conv2d_7/depthwise',
            'PR_Conv4.PR_Conv.1.weight': 'model_1/model/conv2d_5/Conv2D',
            'PR_Conv4.PR_Conv.1.bias': 'model_1/model/batch_normalization_5/FusedBatchNormV3;model_1/model/depthwise_conv2d_7/depthwise;model_1/model/conv2d_5/Conv2D',

            'PR_Conv5.PR_Relu.weight': 'model_1/model/p_re_lu_5/add;model_1/model/p_re_lu_5/Relu;model_1/model/p_re_lu_5/Neg_1;model_1/model/p_re_lu_5/Relu_1;model_1/model/p_re_lu_5/mul',
            'PR_Conv5.PR_Conv.0.weight': 'model_1/model/depthwise_conv2d_5/depthwise',
            'PR_Conv5.PR_Conv.0.bias': 'model_1/model/depthwise_conv2d_7/depthwise',
            'PR_Conv5.PR_Conv.1.weight': 'model_1/model/conv2d_6/Conv2D',
            'PR_Conv5.PR_Conv.1.bias': 'model_1/model/batch_normalization_6/FusedBatchNormV3;model_1/model/depthwise_conv2d_7/depthwise;model_1/model/conv2d_6/Conv2D',

            'PR_Conv6.PR_Relu.weight': 'model_1/model/p_re_lu_6/add;model_1/model/p_re_lu_6/Relu;model_1/model/p_re_lu_6/Neg_1;model_1/model/p_re_lu_6/Relu_1;model_1/model/p_re_lu_6/mul',
            'PR_Conv6.PR_Conv.0.weight': 'model_1/model/depthwise_conv2d_6/depthwise',
            'PR_Conv6.PR_Conv.0.bias': 'model_1/model/depthwise_conv2d_7/depthwise',
            'PR_Conv6.PR_Conv.1.weight': 'model_1/model/conv2d_7/Conv2D',
            'PR_Conv6.PR_Conv.1.bias': 'model_1/model/batch_normalization_7/FusedBatchNormV3;model_1/model/depthwise_conv2d_7/depthwise;model_1/model/conv2d_7/Conv2D',

            'MP_Pad_Conv2.PR_Relu.weight': 'model_1/model/p_re_lu_7/add;model_1/model/p_re_lu_7/Relu;model_1/model/p_re_lu_7/Neg_1;model_1/model/p_re_lu_7/Relu_1;model_1/model/p_re_lu_7/mul',
            'MP_Pad_Conv2.conv.0.weight': 'model_1/model/depthwise_conv2d_7/depthwise1',
            'MP_Pad_Conv2.conv.0.bias': 'model_1/model/depthwise_conv2d_7/depthwise',
            'MP_Pad_Conv2.conv.1.weight': 'model_1/model/conv2d_8/Conv2D',
            'MP_Pad_Conv2.conv.1.bias': 'model_1/model/batch_normalization_8/FusedBatchNormV3;model_1/model/conv2d_25/Conv2D;model_1/model/conv2d_8/Conv2D',

            'PR_Conv7.PR_Relu.weight': 'model_1/model/p_re_lu_8/add;model_1/model/p_re_lu_8/Relu;model_1/model/p_re_lu_8/Neg_1;model_1/model/p_re_lu_8/Relu_1;model_1/model/p_re_lu_8/mul',
            'PR_Conv7.PR_Conv.0.weight': 'model_1/model/depthwise_conv2d_8/depthwise',
            'PR_Conv7.PR_Conv.0.bias': 'model_1/model/conv2d_25/Conv2D',
            'PR_Conv7.PR_Conv.1.weight': 'model_1/model/conv2d_9/Conv2D',
            'PR_Conv7.PR_Conv.1.bias': 'model_1/model/batch_normalization_9/FusedBatchNormV3;model_1/model/conv2d_25/Conv2D;model_1/model/conv2d_9/Conv2D',

            'PR_Conv8.PR_Relu.weight': 'model_1/model/p_re_lu_9/add;model_1/model/p_re_lu_9/Relu;model_1/model/p_re_lu_9/Neg_1;model_1/model/p_re_lu_9/Relu_1;model_1/model/p_re_lu_9/mul',
            'PR_Conv8.PR_Conv.0.weight': 'model_1/model/depthwise_conv2d_9/depthwise',
            'PR_Conv8.PR_Conv.0.bias': 'model_1/model/conv2d_25/Conv2D',
            'PR_Conv8.PR_Conv.1.weight': 'model_1/model/conv2d_10/Conv2D',
            'PR_Conv8.PR_Conv.1.bias': 'model_1/model/batch_normalization_10/FusedBatchNormV3;model_1/model/conv2d_25/Conv2D;model_1/model/conv2d_10/Conv2D',

            'PR_Conv9.PR_Relu.weight': 'model_1/model/p_re_lu_10/add;model_1/model/p_re_lu_10/Relu;model_1/model/p_re_lu_10/Neg_1;model_1/model/p_re_lu_10/Relu_1;model_1/model/p_re_lu_10/mul',
            'PR_Conv9.PR_Conv.0.weight': 'model_1/model/depthwise_conv2d_10/depthwise',
            'PR_Conv9.PR_Conv.0.bias': 'model_1/model/conv2d_25/Conv2D',
            'PR_Conv9.PR_Conv.1.weight': 'model_1/model/conv2d_11/Conv2D',
            'PR_Conv9.PR_Conv.1.bias': 'model_1/model/batch_normalization_11/FusedBatchNormV3;model_1/model/conv2d_25/Conv2D;model_1/model/conv2d_11/Conv2D',

            'PReLU1.weight': 'model_1/model/p_re_lu_11/add;model_1/model/p_re_lu_11/Relu;model_1/model/p_re_lu_11/Neg_1;model_1/model/p_re_lu_11/Relu_1;model_1/model/p_re_lu_11/mul',

            'conv2.weight': 'model_1/model/depthwise_conv2d_11/depthwise',
            'conv2.bias': 'model_1/model/conv2d_25/Conv2D',

            'conv3.weight': 'model_1/model/conv2d_12/Conv2D',
            'conv3.bias': 'model_1/model/batch_normalization_12/FusedBatchNormV3;model_1/model/conv2d_22/Conv2D;model_1/model/conv2d_12/Conv2D',

            'PR_Conv10.PR_Relu.weight': 'model_1/model/p_re_lu_12/add;model_1/model/p_re_lu_12/Relu;model_1/model/p_re_lu_12/Neg_1;model_1/model/p_re_lu_12/Relu_1;model_1/model/p_re_lu_12/mul',
            'PR_Conv10.PR_Conv.0.weight': 'model_1/model/depthwise_conv2d_12/depthwise',
            'PR_Conv10.PR_Conv.0.bias': 'model_1/model/conv2d_22/Conv2D',
            'PR_Conv10.PR_Conv.1.weight': 'model_1/model/conv2d_13/Conv2D',
            'PR_Conv10.PR_Conv.1.bias': 'model_1/model/batch_normalization_13/FusedBatchNormV3;model_1/model/conv2d_22/Conv2D;model_1/model/conv2d_13/Conv2D',

            'PR_Conv11.PR_Relu.weight': 'model_1/model/p_re_lu_13/add;model_1/model/p_re_lu_13/Relu;model_1/model/p_re_lu_13/Neg_1;model_1/model/p_re_lu_13/Relu_1;model_1/model/p_re_lu_13/mul',
            'PR_Conv11.PR_Conv.0.weight': 'model_1/model/depthwise_conv2d_13/depthwise',
            'PR_Conv11.PR_Conv.0.bias': 'model_1/model/conv2d_22/Conv2D',
            'PR_Conv11.PR_Conv.1.weight': 'model_1/model/conv2d_14/Conv2D',
            'PR_Conv11.PR_Conv.1.bias': 'model_1/model/batch_normalization_14/FusedBatchNormV3;model_1/model/conv2d_22/Conv2D;model_1/model/conv2d_14/Conv2D',

            'PR_Conv12.PR_Relu.weight': 'model_1/model/p_re_lu_14/add;model_1/model/p_re_lu_14/Relu;model_1/model/p_re_lu_14/Neg_1;model_1/model/p_re_lu_14/Relu_1;model_1/model/p_re_lu_14/mul',
            'PR_Conv12.PR_Conv.0.weight': 'model_1/model/depthwise_conv2d_14/depthwise',
            'PR_Conv12.PR_Conv.0.bias': 'model_1/model/conv2d_22/Conv2D',
            'PR_Conv12.PR_Conv.1.weight': 'model_1/model/conv2d_15/Conv2D',
            'PR_Conv12.PR_Conv.1.bias': 'model_1/model/batch_normalization_15/FusedBatchNormV3;model_1/model/conv2d_22/Conv2D;model_1/model/conv2d_15/Conv2D',

            'PReLU2.weight': 'model_1/model/p_re_lu_15/add;model_1/model/p_re_lu_15/Relu;model_1/model/p_re_lu_15/Neg_1;model_1/model/p_re_lu_15/Relu_1;model_1/model/p_re_lu_15/mul',

            'conv4.weight': 'model_1/model/depthwise_conv2d_15/depthwise',
            'conv4.bias': 'model_1/model/conv2d_22/Conv2D',

            'conv5.weight': 'model_1/model/conv2d_16/Conv2D',
            'conv5.bias': 'model_1/model/batch_normalization_16/FusedBatchNormV3;model_1/model/conv2d_22/Conv2D;model_1/model/conv2d_16/Conv2D',

            'PR_Conv13.PR_Relu.weight': 'model_1/model/p_re_lu_16/add;model_1/model/p_re_lu_16/Relu;model_1/model/p_re_lu_16/Neg_1;model_1/model/p_re_lu_16/Relu_1;model_1/model/p_re_lu_16/mul',
            'PR_Conv13.PR_Conv.0.weight': 'model_1/model/depthwise_conv2d_16/depthwise',
            'PR_Conv13.PR_Conv.0.bias': 'model_1/model/conv2d_22/Conv2D',
            'PR_Conv13.PR_Conv.1.weight': 'model_1/model/conv2d_17/Conv2D',
            'PR_Conv13.PR_Conv.1.bias': 'model_1/model/batch_normalization_17/FusedBatchNormV3;model_1/model/conv2d_22/Conv2D;model_1/model/conv2d_17/Conv2D',

            'PR_Conv14.PR_Relu.weight': 'model_1/model/p_re_lu_17/add;model_1/model/p_re_lu_17/Relu;model_1/model/p_re_lu_17/Neg_1;model_1/model/p_re_lu_17/Relu_1;model_1/model/p_re_lu_17/mul',
            'PR_Conv14.PR_Conv.0.weight': 'model_1/model/depthwise_conv2d_17/depthwise',
            'PR_Conv14.PR_Conv.0.bias': 'model_1/model/conv2d_22/Conv2D',
            'PR_Conv14.PR_Conv.1.weight': 'model_1/model/conv2d_18/Conv2D',
            'PR_Conv14.PR_Conv.1.bias': 'model_1/model/batch_normalization_18/FusedBatchNormV3;model_1/model/conv2d_22/Conv2D;model_1/model/conv2d_18/Conv2D',

            'PR_Conv15.PR_Relu.weight': 'model_1/model/p_re_lu_18/add;model_1/model/p_re_lu_18/Relu;model_1/model/p_re_lu_18/Neg_1;model_1/model/p_re_lu_18/Relu_1;model_1/model/p_re_lu_18/mul',
            'PR_Conv15.PR_Conv.0.weight': 'model_1/model/depthwise_conv2d_18/depthwise',
            'PR_Conv15.PR_Conv.0.bias': 'model_1/model/conv2d_22/Conv2D',
            'PR_Conv15.PR_Conv.1.weight': 'model_1/model/conv2d_19/Conv2D',
            'PR_Conv15.PR_Conv.1.bias': 'model_1/model/batch_normalization_19/FusedBatchNormV3;model_1/model/conv2d_22/Conv2D;model_1/model/conv2d_19/Conv2D',

            'PReLU3.weight': 'model_1/model/p_re_lu_19/add;model_1/model/p_re_lu_19/Relu;model_1/model/p_re_lu_19/Neg_1;model_1/model/p_re_lu_19/Relu_1;model_1/model/p_re_lu_19/mul',
            'conv6.weight': 'model_1/model/conv2d_20/Conv2D',
            'conv6.bias': 'model_1/model/batch_normalization_20/FusedBatchNormV3;model_1/model/conv2d_20/BiasAdd/ReadVariableOp/resource;model_1/model/conv2d_20/BiasAdd',
            'PReLU4.weight': 'model_1/model/p_re_lu_20/add;model_1/model/p_re_lu_20/Relu;model_1/model/p_re_lu_20/Neg_1;model_1/model/p_re_lu_20/Relu_1;model_1/model/p_re_lu_20/mul',
            'conv7.weight': 'model_1/model/depthwise_conv2d_19/depthwise',
            'conv7.bias': 'model_1/model/conv2d_22/Conv2D',
            'conv8.weight': 'model_1/model/conv2d_21/Conv2D',
            'conv8.bias': 'model_1/model/batch_normalization_21/FusedBatchNormV3;model_1/model/conv2d_22/Conv2D;model_1/model/conv2d_21/Conv2D',

            ###
            'PR_Conv16.PR_Relu.weight': 'model_1/model/p_re_lu_21/add;model_1/model/p_re_lu_21/Relu;model_1/model/p_re_lu_21/Neg_1;model_1/model/p_re_lu_21/Relu_1;model_1/model/p_re_lu_21/mul',
            'PR_Conv16.PR_Conv.0.weight': 'model_1/model/depthwise_conv2d_20/depthwise',
            'PR_Conv16.PR_Conv.0.bias': 'model_1/model/conv2d_22/Conv2D',
            'PR_Conv16.PR_Conv.1.weight': 'model_1/model/conv2d_22/Conv2D1',
            'PR_Conv16.PR_Conv.1.bias': 'model_1/model/batch_normalization_22/FusedBatchNormV3;model_1/model/conv2d_22/Conv2D',
            ###
            #
            'PReLU5.weight': 'model_1/model/p_re_lu_22/add;model_1/model/p_re_lu_22/Relu;model_1/model/p_re_lu_22/Neg_1;model_1/model/p_re_lu_22/Relu_1;model_1/model/p_re_lu_22/mul',
            #
            'conv9.weight': 'model_1/model/conv2d_23/Conv2D',
            #
            'conv9.bias': 'model_1/model/batch_normalization_23/FusedBatchNormV3;model_1/model/conv2d_23/BiasAdd/ReadVariableOp/resource;model_1/model/conv2d_23/BiasAdd',
            #
            'PReLU6.weight': 'model_1/model/p_re_lu_23/add;model_1/model/p_re_lu_23/Relu;model_1/model/p_re_lu_23/Neg_1;model_1/model/p_re_lu_23/Relu_1;model_1/model/p_re_lu_23/mul',
            #
            'conv10.weight': 'model_1/model/depthwise_conv2d_21/depthwise',
            #
            'conv10.bias': 'model_1/model/conv2d_25/Conv2D',
            #
            'conv11.weight': 'model_1/model/conv2d_24/Conv2D',
            #
            'conv11.bias': 'model_1/model/batch_normalization_24/FusedBatchNormV3;model_1/model/conv2d_25/Conv2D;model_1/model/conv2d_24/Conv2D',

            'PR_Conv17.PR_Relu.weight': 'model_1/model/p_re_lu_24/add;model_1/model/p_re_lu_24/Relu;model_1/model/p_re_lu_24/Neg_1;model_1/model/p_re_lu_24/Relu_1;model_1/model/p_re_lu_24/mul',
            'PR_Conv17.PR_Conv.0.weight': 'model_1/model/depthwise_conv2d_22/depthwise',
            'PR_Conv17.PR_Conv.0.bias': 'model_1/model/conv2d_25/Conv2D',
            'PR_Conv17.PR_Conv.1.weight': 'model_1/model/conv2d_25/Conv2D1',
            'PR_Conv17.PR_Conv.1.bias': 'model_1/model/batch_normalization_25/FusedBatchNormV3;model_1/model/conv2d_25/Conv2D',

            'PReLU7.weight': 'model_1/model/p_re_lu_25/add;model_1/model/p_re_lu_25/Relu;model_1/model/p_re_lu_25/Neg_1;model_1/model/p_re_lu_25/Relu_1;model_1/model/p_re_lu_25/mul',

            'conv12.weight': 'model_1/model/regressor_palm_16_NO_PRUNING/Conv2D',
            'conv12.bias': 'model_1/model/regressor_palm_16_NO_PRUNING/BiasAdd;model_1/model/regressor_palm_16_NO_PRUNING/Conv2D;model_1/model/regressor_palm_16_NO_PRUNING/BiasAdd/ReadVariableOp/resource',

            'conv13.weight': 'model_1/model/classifier_palm_16_NO_PRUNING/Conv2D',
            'conv13.bias': 'model_1/model/classifier_palm_16_NO_PRUNING/BiasAdd;model_1/model/classifier_palm_16_NO_PRUNING/Conv2D;model_1/model/classifier_palm_16_NO_PRUNING/BiasAdd/ReadVariableOp/resource',

            'conv14.weight': 'model_1/model/regressor_palm_8_NO_PRUNING/Conv2D',
            'conv14.bias': 'model_1/model/regressor_palm_8_NO_PRUNING/BiasAdd;model_1/model/regressor_palm_8_NO_PRUNING/Conv2D;model_1/model/regressor_palm_8_NO_PRUNING/BiasAdd/ReadVariableOp/resource',

            'conv15.weight': 'model_1/model/classifier_palm_8_NO_PRUNING/Conv2D',
            'conv15.bias': 'model_1/model/classifier_palm_8_NO_PRUNING/BiasAdd;model_1/model/classifier_palm_8_NO_PRUNING/Conv2D;model_1/model/classifier_palm_8_NO_PRUNING/BiasAdd/ReadVariableOp/resource'
        }


class PR_Conv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.PR_Relu = nn.PReLU(in_channels)
        self.PR_Conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, groups=in_channels, kernel_size=5, padding=2, stride=1, bias=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, bias=True),
        )

    def forward(self, x):
        y = self.PR_Relu(x)
        return y + self.PR_Conv(y)


class MP_Pad_Conv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.channel_pad = in_channels
        self.PR_Relu = nn.PReLU(in_channels)
        self.MP_Pad = nn.MaxPool2d(2)

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, groups=in_channels, kernel_size=5, stride=2, bias=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=1,  stride=1, bias=True),
        )

    def forward(self, x):
        x = self.PR_Relu(x)
        y = self.MP_Pad(x)
        y = F.pad(y, (0, 0, 0, 0, 0, self.channel_pad))
        z = F.pad(x, (1, 2, 1, 2))
        return self.conv(z) + y


IMAGE_NORMALIZE = False
NO_DECODE = False
NUM_CLASSES = 1
NUM_BOXES = 2016
NUM_COORDS = 18
X_SCALE = 192.0
Y_SCALE = 192.0
W_SCALE = 192.0
H_SCALE = 192.0

SSD_ANCHOR_OPTIONS = {
    "num_layers": 4,
    "min_scale": 0.1484375,
    "max_scale": 0.75,
    "input_size_height": 192,
    "input_size_width": 192,
    "anchor_offset_x": 0.5,
    "anchor_offset_y": 0.5,
    "strides": [8, 16, 16, 16],
    "aspect_ratios": [1.0],
    "reduce_boxes_in_lowest_layer": False,
    "interpolated_scale_aspect_ratio": 1.0,
    "fixed_anchor_size": True,
}


# From https://github.com/google/mediapipe/blob/master/mediapipe/models/palm_detection.tflite
class HandLandmarks(BlazeLandmark):
    def __init__(self):
        super(HandLandmarks, self).__init__()
        self.resolution = 192
        self._define_layers()

    def load_weights(self, path):
        # aa=torch.load(path, map_location=torch.device('cpu'))
        self.load_state_dict(torch.load(path, map_location=torch.device('cpu')), False)
        # self.load_state_dict(path)
        self.eval()

    def _define_layers(self):

        self.conv1 = nn.Conv2d(3, 32, 5, padding=0, stride=2, bias=True)
        self.PR_Conv1 = PR_Conv(32, 32)
        self.PR_Conv2 = PR_Conv(32, 32)
        self.PR_Conv3 = PR_Conv(32, 32)
        self.MP_Pad_Conv1 = MP_Pad_Conv(32, 64)
        self.PR_Conv4 = PR_Conv(64, 64)
        self.PR_Conv5 = PR_Conv(64, 64)
        self.PR_Conv6 = PR_Conv(64, 64)
        self.MP_Pad_Conv2 = MP_Pad_Conv(64, 128)
        self.PR_Conv7 = PR_Conv(128, 128)
        self.PR_Conv8 = PR_Conv(128, 128)
        self.PR_Conv9 = PR_Conv(128, 128)

        self.PReLU1 = nn.PReLU(128)
        self.MaxPool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(128, 128, groups=128, kernel_size=5, stride=2, bias=True)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=1, stride=1, bias=True)

        self.PR_Conv10 = PR_Conv(256, 256)
        self.PR_Conv11 = PR_Conv(256, 256)
        self.PR_Conv12 = PR_Conv(256, 256)

        self.PReLU2 = nn.PReLU(256)
        self.MaxPool2 = nn.MaxPool2d(2)
        self.conv4 = nn.Conv2d(256, 256, groups=256, kernel_size=5, padding=0, stride=2, bias=True)
        self.conv5 = nn.Conv2d(256, 256, kernel_size=1, stride=1, bias=True)

        self.PR_Conv13 = PR_Conv(256, 256)
        self.PR_Conv14 = PR_Conv(256, 256)
        self.PR_Conv15 = PR_Conv(256, 256)

        self.PReLU3 = nn.PReLU(256)
        self.conv6 = nn.Conv2d(256, 256, kernel_size=1, stride=1, bias=True)
        self.PReLU4 = nn.PReLU(256)

        self.conv7 = nn.Conv2d(256, 256, groups=256, kernel_size=5, padding=2, stride=1, bias=True)
        self.conv8 = nn.Conv2d(256, 256, kernel_size=1, stride=1, bias=True)

        self.PR_Conv16 = PR_Conv(256, 256)

        self.PReLU5 = nn.PReLU(256)
        self.conv9 = nn.Conv2d(256, 128, kernel_size=1, stride=1, bias=True)
        self.PReLU6 = nn.PReLU(128)

        self.conv10 = nn.Conv2d(128, 128, groups=128, kernel_size=5, padding=2, stride=1, bias=True)
        self.conv11 = nn.Conv2d(128, 128, kernel_size=1, stride=1, bias=True)
        self.PR_Conv17 = PR_Conv(128, 128)
        self.PReLU7 = nn.PReLU(128)

        self.conv12 = nn.Conv2d(256, 108, kernel_size=1, padding=0, stride=1, bias=True)
        self.conv13 = nn.Conv2d(256, 6, kernel_size=1, padding=0, stride=1, bias=True)
        self.conv14 = nn.Conv2d(128, 36, kernel_size=1, padding=0, stride=1, bias=True)
        self.conv15 = nn.Conv2d(128, 2, kernel_size=1, padding=0, stride=1, bias=True)
        self.Rsize = nn.Upsample(scale_factor=2, mode='linear', align_corners=False)
        
    def forward(self, x):

        x = F.pad(x, (1, 2, 1, 2), "constant", 0)
        y = self.conv1(x)
        y = self.PR_Conv1(y)
        y = self.PR_Conv2(y)
        y = self.PR_Conv3(y)
        y = self.MP_Pad_Conv1(y)
        y = self.PR_Conv4(y)
        y = self.PR_Conv5(y)
        y = self.PR_Conv6(y)
        y = self.MP_Pad_Conv2(y)
        y = self.PR_Conv7(y)
        y = self.PR_Conv8(y)
        y = self.PR_Conv9(y)

        y = self.PReLU1(y)
        branch1 = y
        y = self.MaxPool1(y)
        y = F.pad(y, (0, 0, 0, 0, 0, 128))
        z = F.pad(branch1, (1, 2, 1, 2))
        temp = self.conv2(z)
        temp = self.conv3(temp)
        y = y + temp

        y = self.PR_Conv10(y)
        y = self.PR_Conv11(y)
        y = self.PR_Conv12(y)

        y = self.PReLU2(y)
        branch2 = y
        y = self.MaxPool2(y)
        z = F.pad(branch2, (1, 2, 1, 2))
        temp = self.conv4(z)
        temp = self.conv5(temp)
        y = y + temp

        y = self.PR_Conv13(y)
        y = self.PR_Conv14(y)
        y = self.PR_Conv15(y)

        y = self.PReLU3(y)
        y = F.interpolate(y, scale_factor=2, mode='bilinear', align_corners=False)
        # y = self.Rsize(y)
        y = self.conv5(y)
        y = self.PReLU4(y)
        y = y + branch2

        branch2 = y
        y = self.conv7(y)
        y = self.conv8(y)
        y = y + branch2

        y = self.PR_Conv16(y)

        branch2 = self.PReLU5(y)
        branch3 = branch2
        y = F.interpolate(branch3, scale_factor=2, mode='bilinear', align_corners=False)
        y = self.conv9(y)
        y = self.PReLU6(y)
        y = branch1 + y

        branch1 = y
        y = self.conv10(y)
        y = self.conv11(y)
        y = branch1 + y

        y = self.PR_Conv17(y)

        branch1 = self.PReLU7(y)
        branch4 = branch1

        branch3 = self.conv12(branch3)
        branch3 = branch3.transpose(1, 2).transpose(2, 3)
        branch3 = branch3.reshape(1, -1, 18)

        branch2 = self.conv13(branch2)
        branch2 = branch2.transpose(1, 2).transpose(2, 3)
        branch2 = branch2.reshape(1, -1, 1)

        branch1 = self.conv14(branch1)
        branch1 = branch1.transpose(1, 2).transpose(2, 3)
        branch1 = branch1.reshape(1, -1, 18)

        branch4 = self.conv15(branch4)
        branch4 = branch4.transpose(1, 2).transpose(2, 3)
        branch4 = branch4.reshape(1, -1, 1)

        identity = torch.cat([branch3, branch1], dim=1)
        identity_1 = torch.cat([branch2, branch4], dim=1)

        return identity_1, identity

# from torchstat import stat
# if __name__ == '__main__':
#     model = HandLandmarks()
    # stat(model, (3, 224, 224))
    # dummy_input = Variable(torch.randn(1, 3, 224, 224))
    # model.forward(dummy_input)
