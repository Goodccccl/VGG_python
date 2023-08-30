import os.path as osp
import numpy as np
import onnx
import onnxruntime as ort
import torch
import torchvision

test_arr = np.random.randn(1, 3, 48, 48).astype(np.float32)

model = torch.load(r'F:\Workprojects\TongFu_Bump\runs\savedModel_25_accuracy_0.962478518486023.pth').cuda().eval()
print('pytorch result:', model(torch.from_numpy(test_arr).cuda()))

model_onnx = onnx.load(r'F:\Workprojects\TongFu_Bump\MyVGG.onnx')
onnx.checker.check_model(model_onnx)

ort_session = ort.InferenceSession(r'F:\Workprojects\TongFu_Bump\MyVGG.onnx')
outputs = ort_session.run(None, {'input': test_arr})
print('onnx_result:', outputs)