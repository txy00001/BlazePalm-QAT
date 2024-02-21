# 基于三个框架下的针对重构BlazePalm的QAT量化
首先对谷歌的MediaPipe-BlazePalm进行了基于pytorch的重构；其次将官方权重进行了提取，和重构模型进行了适配；最后将官方权重作为预训练模型在自己的数据集上进行了进一步训练评估；
完成了基于不同框架的QAT量化


## 基于mqbench的qat量化说明
```
QAT_hand_pytorch:基于原模型与MQ的结合  
demo.py：是一个简单的resnet18+mq的全流程样例；
train_mqbrench.py:没有加叶子节点，train_MQ加了叶子节点
```

## 基于pytorch的qat量化说明
```
test.py/train.py是浮点模型的测试/训练代码；test_new_qat为量化模型的测试，train_pytorch为量化训练,blaze_palml_new.py为浮点模型；QAT_net_pytorch为量化模型
```
## 基于tensorrt的qat量化框架说明
```
test.py/train.py是浮点模型的测试/训练代码；test_new_qat为量化模型的测试，train_pytorch为量化训练,blaze_palml_new.py为浮点模型；QAT_net_pytorch为量化模型
```






# 代码运行说明

1. 图片推理
```
python test.py --mode image --source images/
```

2. 视频推理
```
python test.py --mode video --source videos/hand_test.mp4
```

2. 训练
```
python train.py
```
## 训练数据集
链接：https://pan.baidu.com/s/1rNg8Fo7L0Lf2vj4HAtAltg?pwd=1234 
提取码：1234
其他后续补充
