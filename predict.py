import torch
import numpy as np
from osgeo import gdal
from module.image import *
from Unet import ConvNeXtV2
from tqdm import tqdm


def load_model(model_path):
    model = ConvNeXtV2(4, 3,[2, 2, 6, 2],[48, 96, 192, 384])
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def predict_on_patches(model, patches, device, batch_size=4):
    predictions = []
    with torch.no_grad():
        for i in tqdm(range(0, len(patches), batch_size)):
            batch_patches = patches[i:i+batch_size]
            input_tensors = [torch.from_numpy(patch['patch']).unsqueeze(0).float() for patch in batch_patches]
            input_batch = torch.cat(input_tensors).to(device)
            output_batch = model(input_batch)
            output_batch = torch.argmax(output_batch, dim=1)
            for output in output_batch:
                predictions.append(output.squeeze(0).cpu().numpy())
    return predictions

def get_image_patches(image_array, patch_size):
    patches = []
    c, h, w = image_array.shape
    padded_image = np.pad(image_array, ((0, 0), (0, patch_size - h % patch_size), (0, patch_size - w % patch_size)), mode='constant')
    padded_h, padded_w = padded_image.shape[1], padded_image.shape[2]
    
    for i in range(0, padded_h, patch_size):
        for j in range(0, padded_w, patch_size):
            patch = padded_image[:, i:i+patch_size, j:j+patch_size]
            patches.append({'patch': patch, 'position': (i, j)})
    
    return patches, h, w

def main(image_path, model_path, patch_size, output_path):
    # Load the model
    model = load_model(model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Read the image and convert to array
    # image_info = [img_bands, img_width, img_height, img_geotrans, img_proj,img_nodata]
    _,image_info,image_array = read_multi_bands(image_path)
    # Get image patches
    patches, original_h, original_w = get_image_patches(image_array, patch_size)

    # Predict on patches
    predictions = predict_on_patches(model, patches, device)

    # Combine predictions into a single array
    prediction_array = np.zeros((original_h, original_w), dtype=np.int8)
    for i, patch in enumerate(patches):
        x, y = patch['position']
        # 输出结果与原始影像大小一致，去除padding部分
        x_end = min(x + patch_size, original_h)
        y_end = min(y + patch_size, original_w)
        prediction_array[x:x_end, y:y_end] = predictions[i][:x_end-x, :y_end-y]

    # Save the prediction array as a new TIFF file
    driver = gdal.GetDriverByName('GTiff')
    out_ds = driver.Create(output_path, prediction_array.shape[1], prediction_array.shape[0], 1, gdal.GDT_Byte)
    out_ds.SetGeoTransform(image_info[3])
    out_ds.SetProjection(image_info[4])
    out_ds.GetRasterBand(1).WriteArray(prediction_array)
    out_ds.FlushCache()

if __name__ == "__main__":
    image_path = r'E:\2024Work\CCFA_基于航片的玉米异常检测\CCFBDCI\CCF大数据与计算智能大赛数据集\result.tif'
    model_path = r'E:\2024Work\CCFA_基于航片的玉米异常检测\ConvNeXtV2-Unet\checkpoint\ConvNeXtV2-Unet_best_4_512_512.pth'
    output_path = r'E:\2024Work\CCFA_基于航片的玉米异常检测\ConvNeXtV2-Unet\predictions.tif'
    patch_size = 512  # Adjust patch size as needed
    main(image_path, model_path, patch_size, output_path)
