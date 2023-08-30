import os
from torch.utils.data import Dataset
from PIL import Image


class MyData(Dataset):
    def __init__(self, root_dir, image_dirs, transform):
        assert isinstance(image_dirs, list)
        self.labels = image_dirs
        self.root_dir = root_dir

        self.image_list = []
        self.image_paths = []
        self.label_list = []
        for imgDir in image_dirs:
            path = os.path.join(root_dir, imgDir)
            imgDir_list = os.listdir(path)
            self.image_list.extend(imgDir_list)
            self.image_paths.extend([path] * len(imgDir_list))

            self.label_list.extend([self.labels.index(imgDir)] * len(imgDir_list))

        self.transform = transform

        # self.image_list.sort()
        # self.label_list.sort()

    def __getitem__(self, idx):
        # if idx>=len(self.image_list):
        #     print(idx)
        #     idx=0
        img_name = self.image_list[idx]
        img_item_path = os.path.join(self.image_paths[idx], img_name)
        img = Image.open(img_item_path)
        img = self.transform(img)
        # 文件夹名称是label
        label = float(self.label_list[idx])

        # label单独放在文件夹中
        # label_name = self.label_list[idx]
        # label_item_path = os.path.join(self.root_dir, self.label_dir, label_name)
        # with open(label_item_path) as f:
        #     label=f.readline()

        return img, label, img_name

    def __len__(self):
        return len(self.image_list)
