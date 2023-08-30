import os
import random
import shutil
import time

import cv2
import torch.optim
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Subset, random_split
from torchvision.transforms import transforms

import data
import model

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


if __name__ == '__main__':
    # 数据加载
    img_size = 48

    anomalyThreshold = 0.5

    transform = transforms.Compose([transforms.Resize((img_size, img_size)),
                                    transforms.ToTensor(),
                                    # transforms.ToTensor()])
                                    transforms.Normalize(mean=(0.3150, 0.0619, 0.1087), std=(0.2419, 0.0771, 0.2342))])
                                    # transforms.Normalize(mean=(0.28, 0.168, 0.102), std=(0.234, 0.149, 0.233))])  #

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print("Using device:{} to detect!".format(device))

    root_dir = r"F:\Workprojects\TongFu_Bump\test"
    # root_dir = r"F:\Workprojects\TongFu_Bump\data\test_data\NG"
    # root_dir = r"F:\Workprojects\TongFu_Bump\data\test_data\OK"
    # test_NG = r'F:\Workprojects\TongFu_Bump\data\test_NG'
    # test_OK = r'F:\Workprojects\TongFu_Bump\data\test_OK'

    # 创建模型
    myModel = torch.load(r"F:\Workprojects\TongFu_Bump\runs\savedModel_299_accuracy_0.9217391014099121.pth")

    myModel.to(device)

    myModel.eval()
    total_test_loss = 0
    total_accuracy = 0

    test_lists = os.listdir(root_dir)

    nums = int(len(test_lists))

    accuracy_nums = 0
    for i in range(nums):
        start_time = time.time()
        img_name = test_lists[i]
        img_path = os.path.join(root_dir, img_name)
        original_img = Image.open(img_path).convert('RGB')
        # original_img = cv2.imread(img_path)
        # original_img = Image.fromarray(original_img)
        img = transform(original_img)
        # a = []
        # for c in range(3):
        #     for m in range(48):
        #         for n in range(48):
        #             a.append(img[c][m][n])
        #             print(img[c][m][n])
        # print(len(a))


        img = img.to(device)
        img = img.unsqueeze(0)

        outputs = myModel(img)
        outputs = torch.squeeze(outputs)
        end_time = time.time()
        cost_time = end_time - start_time
        if outputs < anomalyThreshold:
            print('当前测试样{}为error,得分为:{}   Took {} second.'.format(img_name, outputs, cost_time))
            # shutil.copy(root_dir + '/' + img_name, test_NG + '/' + img_name)
        else:
            print('当前测试样{}为normal,得分为:{}   Took {} second.'.format(img_name, outputs, cost_time))
            # shutil.copy(root_dir + '/' + img_name, test_OK + '/' + img_name)

