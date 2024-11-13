import os
from osgeo import gdal
import torch
from module.image import *
import time
from torch.utils.data import Dataset



class CustomDataset(Dataset):
    def __init__(self, image_dir, label_dir, transform=None):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(image_dir) if f.endswith('.tif')]
        
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.image_files[idx])
        label_path = os.path.join(self.label_dir, self.image_files[idx])
        image = ImageProcess(filepath=image_path)
        # 读取影像元信息
        image_info = image.read_img_info()
        # 读取影像矩阵
        image_data = image.read_img_data()
        label = ImageProcess(filepath=label_path)
        # 读取影像元信息
        label_info = label.read_img_info()
        # 读取影像矩阵
        label_data = label.read_img_data()
        label_data = label_data.reshape(1, label_data.shape[0], label_data.shape[1])

        # (C, H, W) -> (H, W, C)
        image_data = image_data.transpose(1, 2, 0)
        label_data = label_data.transpose(1, 2, 0)
        
        if self.transform:
            augmented = self.transform(image=image_data, mask=label_data)
            image_data = augmented['image']
            label_data = augmented['mask']
        
        # (H, W, C) -> (C, H, W)
        image_data = image_data.transpose(2, 0, 1)
        label_data = label_data.transpose(2, 0, 1)[0]
        
        # 对齐模型参数
        image_data = image_data.astype('float32')
        
        return image_data, label_data
    
class TimeTracker:
    def __init__(self):
        self.start_time = None
        self.epoch_times = []

    def start_epoch(self):
        self.start_time = time.time()

    def end_epoch(self):
        if self.start_time is not None:
            epoch_time = time.time() - self.start_time
            self.epoch_times.append(epoch_time)
            self.start_time = None
            return epoch_time
        else:
            raise ValueError("Epoch timer was not started.")

    def total_time(self):
        return sum(self.epoch_times)
