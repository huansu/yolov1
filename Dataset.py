# 测试VOC2007数据集的读取
import cv2
import numpy as np
import os
import pandas as pd
import torch
import albumentations as A

from PIL import Image, ImageFile
from torch.utils.data import Dataset, DataLoader


ImageFile.LOAD_TRUNCATED_IMAGES = True

class YOLODataset(Dataset):
    def __init__(
        self,
        csv_file,
        img_dir,
        label_dir,
        image_size=416,
        transforms=None,
    ):
        self.annotations = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.image_size = image_size
        self.transform = transforms


    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        label_path = os.path.join(self.label_dir, self.annotations.iloc[index, 1])
        # 改变labels信息排列
        bboxes = np.roll(np.loadtxt(fname=label_path, delimiter=" ", ndmin=2), 4, axis=1).tolist()
        img_path = os.path.join(self.img_dir, self.annotations.iloc[index, 0])
        image = cv2.imread(img_path)

        # 图像格式转换
        if self.transform:
            image = self.transform(image=image)['image']
        bboxes = np.array(bboxes)
        return image, bboxes # [3,416,416] (0/1，x, y, w, h, classes)等


def test():
    transforms = A.Compose(
        [
            # 保持宽高比缩放图片，使最大边等于max_size,即IMAGE_SIZE
            A.LongestMaxSize(max_size=416),
            # padding填充图像
            A.PadIfNeeded(
                min_height=416, min_width=416, border_mode=cv2.BORDER_CONSTANT
            ),
            A.Normalize(mean=[0, 0, 0], std=[1, 1, 1], max_pixel_value=255, ),
            #ToTensorV2(),
        ],
        # bounding box参数检查
        #bbox_params=A.BboxParams(format="pascal_voc", min_visibility=0.4, label_fields=[]),
    )
    dataset = YOLODataset(
        csv_file="./VOC2007/train.csv",
        img_dir="./VOC2007/JPEGImages/",
        label_dir="./VOC2007/Labels/",
        transform=transforms,
    )
    loader = DataLoader(dataset=dataset, batch_size=1, shuffle=False)
    for batch_idx, (x, y) in enumerate(loader):         # iamge 和 三个不同的特征图
        print(y)

if __name__ == '__main__':
    test()