import torch
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import os

import torch.nn as nn
import torch.optim as optim
from Unet import ConvNeXtV2  # Assuming your model is defined in Unet.py
from Dataloader import *
import argparse
from torch.utils.tensorboard import SummaryWriter
import random
import numpy as np
from prettytable import PrettyTable
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
from Accuracy import *


def parse_args():
    parser = argparse.ArgumentParser(description='Train ConvNeXtV2-Unet model')
    parser.add_argument('--model_name', type=str, default=r'ConvNeXtV2-Unet', help='model name')
    parser.add_argument('--classes', type=int, default=3, help='Number of classes in the dataset')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for optimizer')
    parser.add_argument('--num_epochs', type=int, default=40, help='Number of epochs to train')
    parser.add_argument('--train_val_split', type=float, default=0.8, help='Train-validation split ratio')
    parser.add_argument('--image_dir', type=str, default=r'E:\2024Work\CCFA_基于航片的玉米异常检测\submission\data\img', help='Directory for input images')
    parser.add_argument('--label_dir', type=str, default=r'E:\2024Work\CCFA_基于航片的玉米异常检测\submission\data\label', help='Directory for labels')
    parser.add_argument('--checkpoint_dir', type=str, default=r'E:\2024Work\CCFA_基于航片的玉米异常检测\ConvNeXtV2-Unet\checkpoint', help='Directory for checkpoints')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    return parser.parse_args()

def set_seed(seed):
    """
    设置随机种子以确保实验的可重复性
    """
    torch.manual_seed(seed)  # 设置CPU生成随机数的种子
    torch.cuda.manual_seed(seed)  # 为当前GPU设置随机种子
    torch.cuda.manual_seed_all(seed)  # 为所有GPU设置随机种子
    np.random.seed(seed)  # 设置numpy生成随机数的种子
    random.seed(seed)  # 设置Python生成随机数的种子

def run():
    args = parse_args()
    set_seed(args.seed)

    # Hyperparameters
    model_name = args.model_name
    classes = args.classes
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    num_epochs = args.num_epochs
    train_val_split = args.train_val_split
    image_dir = args.image_dir
    label_dir = args.label_dir
    checkpoint_dir = args.checkpoint_dir
    

    # 数据增强
    transform = transforms.Compose([
        transforms.ToTensor(),  # 转换为张量
        transforms.RandomHorizontalFlip(),  # 随机水平翻转
        transforms.RandomVerticalFlip(),  # 随机垂直翻转
        transforms.RandomRotation(90),  # 随机旋转90度
        transforms.RandomResizedCrop(512, scale=(0.8, 1.0)),  # 随机裁剪并调整大小到512
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # 随机调整图像颜色
        transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),  # 随机高斯模糊
    ])

    # Create dataset
    dataset = CustomDataset(image_dir=image_dir, label_dir=label_dir, transform=None) #不进行aug
    image_1,_ = dataset[0]
    channels,height,width = image_1.shape
    image_size_str = f"{height}_{width}"
    train_size = int(train_val_split * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    print(f"Train size: {train_size}, Validation size: {val_size}")

    # Check if GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize model, loss function, and optimizer
    # model = ConvNeXtV2(in_chans=4, out_channel=classes, depths=[3, 3, 9, 3], dims=[96, 192, 384, 768]).to(device)
    model = ConvNeXtV2(in_chans=4, out_channel=classes, depths=[2, 2, 6, 2], dims=[48, 96, 192, 384]).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    log_dir = os.path.join(checkpoint_dir, f'logs/{model_name}')
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
            # TensorBoard setup
    writer = SummaryWriter(log_dir=log_dir)
    # Training loop
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        print(f"{'-'*30}Epoch {epoch} training started{'-'*30}")
        for images, masks in tqdm(train_loader):
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks.long())
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader)}")

        # Validation
        print(f"{'-'*30}Epoch {epoch} validation started{'-'*30}")
        model.eval()

        # Initialize confusion matrix
        conf_matrix = np.zeros((classes, classes), dtype=int)
        with torch.no_grad():
            for images, masks in tqdm(val_loader):
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                _, preds = torch.max(outputs, 1)
                # Update confusion matrix
                conf_matrix += confusion_matrix(masks.cpu().numpy().flatten().astype(np.int64), preds.cpu().numpy().flatten().astype(np.int64), labels=list(range(classes)))
                # i += 1
                # if i == 5:
                #     break

        # Define class names
        class_names = ['Background', 'Corn', 'Affected corn']

        # Calculate metrics for each class
        class_acc = np.round(conf_matrix.diagonal() / conf_matrix.sum(axis=1), 4)
        class_f1 = np.round(2 * conf_matrix.diagonal() / (conf_matrix.sum(axis=1) + conf_matrix.sum(axis=0)), 4)
        class_iou = np.round(conf_matrix.diagonal() / (conf_matrix.sum(axis=1) + conf_matrix.sum(axis=0) - conf_matrix.diagonal()), 4)

        # Calculate average metrics
        avg_acc = round(np.mean(class_acc), 4)
        avg_f1 = round(np.mean(class_f1), 4)
        avg_iou = round(np.mean(class_iou), 4)

        # Display results using PrettyTable
        table = PrettyTable()
        table.field_names = ["Class", "Accuracy", "F1 Score", "IOU"]
        kappa = round(cal_kappa(conf_matrix), 4)  
        for i, class_name in enumerate(class_names):
            table.add_row([class_name, class_acc[i], class_f1[i], class_iou[i]])

        print(table)
        print(f"Validation AVG Accuracy: {avg_acc}, F1 Score: {avg_f1}, IOU: {avg_iou}, Kappa: {kappa}")


        model_path = os.path.join(log_dir, f"{model_name}_{epoch}_{batch_size}_{image_size_str}.pth")
        torch.save(model.state_dict(), model_path)
        if epoch == 0 or avg_acc > best_acc:
            print(f"New Record! Validation Accuracy: {avg_acc}, F1 Score: {avg_f1}, IOU: {avg_iou}, Kappa: {kappa}")
            model_best_path = os.path.join(checkpoint_dir, f"{model_name}_best_{batch_size}_{image_size_str}.pth")
            best_acc = avg_acc
            torch.save(model.state_dict(), model_best_path)
   

        # Log the loss
        writer.add_scalar('Loss/train', running_loss/len(train_loader), epoch)
        writer.add_scalar('Accuracy/val', avg_acc, epoch)
        writer.add_scalar('F1/val', avg_f1, epoch)
        writer.add_scalar('IOU/val', avg_iou, epoch)
        writer.add_scalar('Kappa/val', kappa, epoch)

        image_save_dir = os.path.join(log_dir, 'images')
        if not os.path.exists(image_save_dir):
            os.makedirs(image_save_dir)

        for i in range(10):
            # Save original image (first 3 channels)
            origin_path = os.path.join(image_save_dir, f'epoch_{epoch}_original_image_{i}.png')
            origin_image = save_image(images[i][:3], origin_path)
            writer.add_image(f'Original/epoch_{epoch}_image_{i}', origin_image, epoch, dataformats='HWC')

            # Save label image (convert to RGB)
            label = masks[i].cpu().numpy()
            label_rgb = np.zeros((label.shape[0], label.shape[1], 3), dtype=np.uint8)
            unique_values = np.unique(label)
            for val in unique_values:
                if val == 0:
                    label_rgb[label == val] = [0, 0, 0]  # Background
                elif val == 1:
                    label_rgb[label == val] = [0, 255, 0]  # Corn
                elif val == 2:
                    label_rgb[label == val] = [255, 0, 0]  # Affected corn
            label_path = os.path.join(image_save_dir, f'epoch_{epoch}_label_image_{i}.png')
            plt.imsave(label_path, label_rgb)
            writer.add_image(f'Label/epoch_{epoch}_image_{i}', label_rgb, epoch, dataformats='HWC')

            # Save predict_maps as heatmaps
            predict_maps = outputs[i].cpu().detach()
            for j in range(predict_maps.shape[0]):
                predict_map_path = os.path.join(image_save_dir, f'epoch_{epoch}_predict_maps_{i}_pred{class_names[j]}.png')
                save_heatmap(predict_maps[j], predict_map_path)
                # Log predict_maps to TensorBoard
                writer.add_image(f'Predict_Maps/epoch_{epoch}_image_{i}_feature_{j}', predict_maps[j], epoch, dataformats='HW')
        writer.close()
        

if __name__ == "__main__":
    run()