'''
@Github: https://github.com/liuzehao
@Blog: https://blog.csdn.net/liu506039293

训练之前进行图像增强
'''
import numpy as np
import imgaug.augmenters as iaa
import os
import cv2
import pandas as pd
import matplotlib.pyplot as plt
import imgaug as ia

def augData(root_path,output_path,batch,epoch):
    files = os.listdir(root_path)

    new_names = []
    a = (np.arange(0, len(files)))
    for j in range(epoch):
        print("epoch:{}".format(j))
        images = []
        random_img_names = []
        for mm in range(batch):
            index = np.random.choice(a)
            random_img_names.append(files[index])
            images.append(cv2.imread(os.path.join(root_path,files[index])))


        images = np.array(images)

        sometimes = lambda aug: iaa.Sometimes(1, aug)

        seq = iaa.Sequential(
            [
                sometimes(iaa.CoarseDropout(p=0.2, per_channel=True)),
                # sometimes(iaa.CoarseSaltAndPepper(p=0.2)),
            ],
            random_order=True
        )

        images_aug = seq(images=images)

        for i in range(len(images_aug)):
            print("---------BATCH_SIZE:{}".format(i))

            new_name = "AUG_{}_{}_{}.bmp".format(random_img_names[i][:-4], j,i)  # "Aug_data"
            new_names.append(new_name)

            cv2.imwrite(os.path.join(output_path,new_name),images_aug[i])

    if len(new_names) != len(set(new_names)):
        print("数据名称有重复")


if __name__=="__main__":
    root_path=r"D:\Desktop\Project_Python\My_Project\10_tongfu\DataSet1026\normal"
    output_path=r"D:\Desktop\Project_Python\My_Project\10_tongfu\DataSet1026\error"
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    batch=20    # 一次增强batch个数据
    epoch=10    #增强epoch轮
    augData(root_path,output_path, batch, epoch)



