import torch
import unittest

#from test.version import GITHUB_RES




from mqbench.prepare_by_platform import prepare_by_platform, BackendType
from mqbench.convert_deploy import convert_deploy
from mqbench.utils.state import enable_calibration, enable_quantization
#from resnet import resnet18
from models_QAT.blaze_palml_new_QAT import BlazePalm_QAT



model_to_quantize = BlazePalm_QAT()
dummy_input = torch.randn(1, 3, 192, 192, device = torch.device("cpu"))
model_to_quantize.train()
extra_qconfig_dict = {
    'w_observer': 'MinMaxObserver',
    'a_observer': 'EMAMinMaxObserver',
    'w_fakequantize': 'FixedFakeQuantize',
    'a_fakequantize': 'FixedFakeQuantize',
}
prepare_custom_config_dict = {'extra_qconfig_dict': extra_qconfig_dict}
model_prepared = prepare_by_platform(model_to_quantize, BackendType.Tensorrt, prepare_custom_config_dict)
enable_calibration(model_prepared)
model_prepared(dummy_input)
enable_quantization(model_prepared)


loss = model_prepared(dummy_input).sum()
loss.backward()
model_prepared.eval()
convert_deploy(model_prepared, BackendType.Tensorrt, {'x': [1, 3, 192, 192]}, model_name='fixed.onnx')