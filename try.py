import torch
import torchvision
import cv2
import model

img = cv2.imread(r'F:\1.jpg')
print(img)
print(img.shape)
img = torch.Tensor(img)
print(img)
print(img.shape)
img2 = img.half()
print(img2)
print(img2.shape)