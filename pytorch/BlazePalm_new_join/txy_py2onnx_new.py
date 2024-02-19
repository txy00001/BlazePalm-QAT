import torch



# from networks.mobilenet_small import MobileFaceNet
# from networks.esnet1 import ESNet
from models.blaze_palml_new import BlazePalm
from models_QAT.QAT_net_pytorch import BlazePalm_QAT

input = torch.rand((1, 3, 192, 192))
net = BlazePalm_QAT()
# pthfile = '/new_face_class/checkpoints/2023-01-03_16-46/checkpoint-000015.pth'
pthfile = r'weights/  BlazePalm_QAT_8.pth'
loaded_model = torch.load(pthfile, map_location='cpu')
# try:
#   loaded_model.eval()
# except AttributeError as error:
#   print(error)


# print(net)
# 获取参数的数量和计算FLOPs
# from thop import profile
# flops, params = profile(net, inputs=(input, ))
# print('FLOPs = ' + str(flops/1000**3) + 'G')
# print('Params = ' + str(params/1000**2) + 'M')
# # exit()
# # ckpt = torch.load("./model/mid/045_ms1m.ckpt", map_location=torch.device('cpu'))
# # net.load_state_dict(ckpt['net_state_dict'])

net.eval()
# cal_similary(net)

input_names = ["input"]
output_names = ['output']

# torch.onnx.export(model, (dummy_input1, dummy_input2, dummy_input3), "C3AE.onnx", verbose=True, input_names=input_names, output_names=output_names)
torch.onnx.export(net, input, "./new.onnx", verbose=True, input_names=input_names,opset_version = 13,
               output_names=output_names)

print("ok!")


######
import nn
# import torch
# import torchvision
# from pytorch_quantization import nn as quant_nn
# from pytorch_quantization import quant_modules
# from torch import optim
#
#
# from models_QAT.blaze_palml_new_QAT import BlazePalm_QAT
#
# quant_nn.TensorQuantizer.use_fb_fake_quant = True
#
# quant_modules.initialize()
# model = BlazePalm_QAT()
# # models = resnet18
# model.train()
# dummy_input = torch.randn(1, 3, 192, 192, device='cpu')
#
#
# loss =  model(dummy_input).sum()
# loss.backward()
# model.eval()
#
# input_names = ["input1"]
# output_names = ["output1"]
# # torch.onnx.export(model, (dummy_input1, dummy_input2, dummy_input3), "C3AE.onnx", verbose=True, input_names=input_names, output_names=output_names)
# torch.onnx.export(model, dummy_input, "./1.onnx", verbose=True, input_names=input_names,opset_version=13,
#                output_names=output_names)
# print("ok")