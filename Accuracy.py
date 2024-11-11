import numpy as np
from module.image import *
import matplotlib.pyplot as plt
import seaborn as sns
import torchvision
from prettytable import PrettyTable
from sklearn.metrics import confusion_matrix
def cal_kappa(confusion_matrix):
    """
    计算多分类混淆矩阵的kappa系数
    """
    # 计算总样本数
    if confusion_matrix.dtype != np.int64:
        confusion_matrix = confusion_matrix.astype(np.int64)
    N = np.sum(confusion_matrix)  
    # 计算对角线元素之和（实际一致性）
    sum_po = np.trace(confusion_matrix)
    
    # 计算每行和每列的和
    row_sums = np.sum(confusion_matrix, axis=1)
    col_sums = np.sum(confusion_matrix, axis=0)
    
    # 计算期望一致性
    sum_pe = np.sum(row_sums * col_sums) / (N ** 2)
    
    po = sum_po / N  # 实际一致性
    pe = sum_pe  # 期望一致性
    kappa = (po - pe) / (1 - pe)  # Kappa系数

    return kappa


def save_image(tensor, path):
    grid = torchvision.utils.make_grid(tensor)
    ndarr = grid.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
    plt.imsave(path, ndarr)
    return ndarr

def save_heatmap(tensor, path):
    heatmap = tensor.cpu().numpy()
    sns.heatmap(heatmap, cmap='viridis')
    plt.savefig(path)
    plt.close()

def evaluate_classification(pre_path, gt_path):
    """
    计算分类结果的混淆矩阵、IOU、F1和ACC
    """
    # Read Tif
    _,_,pre_img = read_single_band(pre_path)
    _,_,gt_img = read_single_band(gt_path)

    # Define class names
    class_names = ['Background', 'Corn', 'Affected corn']
    num_classes = len(class_names)
    
    # 计算混淆矩阵
    conf_matrix = confusion_matrix(pre_img.ravel(), gt_img.ravel(), labels=list(range(num_classes)))

    # Calculate metrics for each class
    class_acc = np.zeros(num_classes)
    class_f1 = np.zeros(num_classes)
    class_iou = np.zeros(num_classes)
    
    for i in range(num_classes):
        tp = conf_matrix[i, i]
        fn = conf_matrix[i, :].sum() - tp
        fp = conf_matrix[:, i].sum() - tp
        tn = conf_matrix.sum() - (tp + fn + fp)
        
        class_acc[i] = tp / (tp + fn) if (tp + fn) > 0 else 0
        class_f1[i] = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0
        class_iou[i] = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0

    # Calculate average metrics
    avg_acc = round(np.mean(class_acc), 4)
    avg_f1 = round(np.mean(class_f1), 4)
    avg_iou = round(np.mean(class_iou), 4)

    # Display results using PrettyTable
    table = PrettyTable()
    table.field_names = ["Class", "Accuracy", "F1 Score", "IOU"]
    kappa = round(cal_kappa(conf_matrix), 4)  
    for i, class_name in enumerate(class_names):
        table.add_row([class_name, round(class_acc[i], 4), round(class_f1[i], 4), round(class_iou[i], 4)])

    print(table)
    print(f"Validation Avg Accuracy: {avg_acc}, F1 Score: {avg_f1}, IOU: {avg_iou}, Kappa: {kappa}")
    

if __name__ == '__main__':

    pre_path = r'E:\2024Work\CCFA_基于航片的玉米异常检测\ConvNeXtV2-Unet\predictions.tif'
    gt_path = r'E:\2024Work\CCFA_基于航片的玉米异常检测\CCFBDCI\CCF大数据与计算智能大赛数据集\standard.tif'
    evaluate_classification(pre_path, gt_path)