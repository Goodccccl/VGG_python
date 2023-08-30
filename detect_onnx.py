
import os, sys
import onnxruntime
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

class_names = ['error', 'normal']

onnx_path = r'F:\Workprojects\TongFu_Bump\MyVGG.onnx'
image_path = r'F:/Workprojects/TongFu_Bump/test/NG_bump_1_L0.tif_2023_3_8_10_5_12_445_178_s1_0.907_s2_0.855_s3_0.836_s4_0.456_rErr_1.190.bmp'

img_size = 48
transform = transforms.Compose([transforms.Resize((img_size, img_size)),
                                    transforms.ToTensor(),
                                    # transforms.ToTensor()])
                                    transforms.Normalize(mean=(0.3150, 0.0619, 0.1087), std=(0.2419, 0.0771, 0.2342))])
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
origin_img = Image.open(image_path).convert('RGB')
img = transform(origin_img)
img = img.to(device)
img = img.unsqueeze(0)
onet_session = onnxruntime.InferenceSession(onnx_path)
inputs = {onet_session.get_inputs()[0].name:to_numpy(img)}
outs = onet_session.run(None, inputs)
preds = outs[0]
print(preds)