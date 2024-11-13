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
import logging
import albumentations as A
from albumentations.pytorch import ToTensorV2


def parse_args():
    parser = argparse.ArgumentParser(description='Train ConvNeXtV2-Unet model')
    parser.add_argument('--model_name', type=str, default=r'ConvNeXtV2-Unet', help='model name')
    parser.add_argument('--classes', type=int, default=8, help='Number of classes in the dataset')
    parser.add_argument('--class_names', type=list, default=['Background', 'Wheat', 'Corn', 'Sunflower', 'Watermelon', 
    'Tomato', 'Onion', 'Zucchini'], help='Class names')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=0.0002, help='Learning rate for optimizer')
    parser.add_argument('--num_epochs', type=int, default=40, help='Number of epochs to train')
    parser.add_argument('--train_val_split', type=float, default=0.7, help='Train-validation split ratio')
    parser.add_argument('--image_dir', type=str, default=r'E:\2024Work\CCFA_基于航片的玉米异常检测\submission\data\img', help='Directory for input images')
    parser.add_argument('--label_dir', type=str, default=r'E:\2024Work\CCFA_基于航片的玉米异常检测\submission\data\label', help='Directory for labels')
    parser.add_argument('--checkpoint_dir', type=str, default=r'E:\2024Work\CCFA_基于航片的玉米异常检测\ConvNeXtV2-Unet\checkpoint', help='Directory for checkpoints')
    parser.add_argument('--pretrained_weights', type=str, default=r'E:\2024Work\CCFA_基于航片的玉米异常检测\ConvNeXtV2-Unet\checkpoint\pretrain\convnextv2_femto_1k_224_fcmae.pt', help='Path to pretrained weights')
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

def setup_logging(log_dir):
    log_file = os.path.join(log_dir, 'training.log')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

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

    # Create checkpoint directory and logging directory
    log_dir = os.path.join(checkpoint_dir, f'logs/{model_name}')
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    setup_logging(log_dir)

    # Data augmentation
    transform = A.Compose([
            A.HorizontalFlip(p=0.5),# 随机水平翻转
            A.VerticalFlip(p=0.5), # 随机垂直翻转
            A.GaussianBlur(blur_limit=(5, 9), p=0.2), # 高斯模糊
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.2),  # 随机增加高斯噪声
            A.OneOf([
                A.MotionBlur(p=0.2),   # 使用随机大小的内核将运动模糊应用于输入图像。
                A.MedianBlur(blur_limit=3, p=0.1),    # 中值滤波
                A.Blur(blur_limit=3, p=0.1),   # 使用随机大小的内核模糊输入图像
            ]), 
            A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=0.2),# 随机应用仿射变换：平移，缩放和旋转输入
            A.RandomBrightnessContrast(p=0.2),   # 随机明亮对比度
        ])
    
    # Create dataset
    dataset = CustomDataset(image_dir=image_dir, label_dir=label_dir, transform=transform) 
    image_1,_ = dataset[0]
    channels,height,width = image_1.shape
    image_size_str = f"{height}_{width}"
    train_size = int(train_val_split * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    

    # Check if GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize model, loss function, and optimizer
    # model = ConvNeXtV2(in_chans=4, out_channel=classes, depths=[3, 3, 9, 3], dims=[96, 192, 384, 768]).to(device)
    model = ConvNeXtV2(in_chans=4, out_channel=classes, depths=[2, 2, 6, 2], dims=[48, 96, 192, 384]).to(device)

    logging.info(f"Train size: {train_size}, Validation size: {val_size}")
    logging.info(f"Model: {model_name}, Classes: {classes}, Batch size: {batch_size}, Learning rate: {learning_rate}, Epochs: {num_epochs}")
    logging.info(f"Image size: {height}x{width}")
    logging.info(f"Class names: {args.class_names}")
    logging.info(f"Pretrained weights path: {args.pretrained_weights}")
    logging.info(f"Random seed: {args.seed}")

    # Load pretrained weights if available
    # if args.pretrained_weights:
    #     if os.path.isfile(args.pretrained_weights):
    #         print(f"Loading pretrained weights from {args.pretrained_weights}")
    #         model.load_state_dict(torch.load(args.pretrained_weights, map_location=device))
    #     else:
    #         print(f"Pretrained weights file not found at {args.pretrained_weights}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)


    # TensorBoard setup
    writer = SummaryWriter(log_dir=log_dir)
    # Training loop
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        logging.info(f"{'-'*30}Epoch {epoch} training started{'-'*30}")
        for images, masks in tqdm(train_loader):
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks.long())
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader)}")
        logging.info(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader)}")

        # Validation
        print(f"{'-'*30}Epoch {epoch} validation started{'-'*30}")
        logging.info(f"{'-'*30}Epoch {epoch} validation started{'-'*30}")
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
        # ['Background', 'Corn', 'Affected corn']
        class_names = args.class_names

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
        logging.info(table)
        print(f"Validation AVG Accuracy: {avg_acc}, F1 Score: {avg_f1}, IOU: {avg_iou}, Kappa: {kappa}")
        logging.info(f"Validation AVG Accuracy: {avg_acc}, F1 Score: {avg_f1}, IOU: {avg_iou}, Kappa: {kappa}")


        model_path = os.path.join(log_dir, f"{model_name}_{epoch}_{batch_size}_{image_size_str}.pth")
        torch.save(model.state_dict(), model_path)
        if epoch == 0 or avg_acc > best_acc:
            print(f"New Record! Validation Accuracy: {avg_acc}, F1 Score: {avg_f1}, IOU: {avg_iou}, Kappa: {kappa}")
            model_best_path = os.path.join(log_dir, f"{model_name}_best_{batch_size}_{image_size_str}.pth")
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
            writer.add_image(f'Original/image_{i}', origin_image, epoch, dataformats='HWC')

            # Save label image (convert to RGB)
            label = masks[i].cpu().numpy()
            label_rgb = np.zeros((label.shape[0], label.shape[1], 3), dtype=np.uint8)
            unique_values = np.unique(label)
            for val in unique_values:
                if val == 0:
                    label_rgb[label == val] = [0, 0, 0]  # Background
                else:
                    label_rgb[label == val] = [random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)]
            label_path = os.path.join(image_save_dir, f'epoch_{epoch}_label_image_{i}.png')
            plt.imsave(label_path, label_rgb)
            writer.add_image(f'Label/image_{i}', label_rgb, epoch, dataformats='HWC')

            # Save predict_maps as heatmaps
            predict_maps = outputs[i].cpu().detach()
            for j in range(predict_maps.shape[0]):
                predict_map_path = os.path.join(image_save_dir, f'epoch_{epoch}_predict_maps_{i}_pred{class_names[j]}.png')
                save_heatmap(predict_maps[j], predict_map_path)
                # Log predict_maps to TensorBoard
                writer.add_image(f'Predict_Maps/image_{i}_feature_{j}', predict_maps[j], epoch, dataformats='HW')
        writer.close()
        

if __name__ == "__main__":
    run()