import torch
from models_QAT.blaze_palml_new_QAT import BlazePalm_QAT

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input = torch.rand((1, 3, 192, 192))
net = BlazePalm_QAT()
pthfile = r'E:\pytorch_QAT\QAT_hand_pytorch\onnx\  BlazePalm_QAT_epoch_1.pth'
loaded_model = torch.load(pthfile, map_location='cpu')
# try:
#   loaded_model.eval()
# except AttributeError as error:
#   print(error)

net.load_state_dict(loaded_model)
# print(net)
# 获取参数的数量和计算FLOPs
# from thop import profile
# flops, params = profile(net, inputs=(input, ))
# print('FLOPs = ' + str(flops/1000**3) + 'G')
# print('Params = ' + str(params/1000**2) + 'M')
# exit()
# ckpt = torch.load("./model/mid/045_ms1m.ckpt", map_location=torch.device('cpu'))
# net.load_state_dict(ckpt['net_state_dict'])
net.eval()
# cal_similary(net)

input_names = ["input1"]
output_names = ["output1"]
# torch.onnx.export(model, (dummy_input1, dummy_input2, dummy_input3), "C3AE.onnx", verbose=True, input_names=input_names, output_names=output_names)
torch.onnx.export(net, input, "./new.onnx", verbose=True, input_names=input_names,
               output_names=output_names)

print("ok!")