import os
from osgeo import gdal
import torch
from module.image import *
import time

class CustomDataset(torch.utils.data.Dataset):
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
        
        if self.transform:
            image_data = self.transform(image_data)
            label_data = self.transform(label_data)
        
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
