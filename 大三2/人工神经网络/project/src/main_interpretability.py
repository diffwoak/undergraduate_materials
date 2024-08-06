# CAM
# 计算单个图片的热力图
# 获取卷积层的输入输出特征需要修改 inference 中的钩子函数

# import os
# import numpy as np
# import data_utils.transform as tr
# from config import INIT_TRAINER
# from torchvision import transforms
# from converter.common_utils import hdf5_reader
# from analysis.analysis_tools import calculate_CAMs, save_heatmap

# import torch.nn as nn
# import torch


# current_fold = "fold1"
# image = "Black_Footed_Albatross_0001_796111"
# image_set = "001.Black_footed_Albatross"

# features_dir = f'./analysis/mid_feature/v1.0/{current_fold}/'
# weight_path = f'./analysis/result/v1.0/{current_fold}_fc_weight.npy'
# images_dir = f'./datasets/CUB_200_2011/CUB_200_2011/images/{image_set}/'


# # 默认的hook获取池化层的输入和输出，池化层的输入'feature_in'即为最后一层卷积层的输出
# weight = np.load(weight_path)

# features = hdf5_reader(f'./analysis/mid_feature/v1.0/{current_fold}/{image}','feature_in')

# print(features.shape)
# print(weight.shape)

# # (256, 28, 28) 第三层输出特征
# # (512, 14, 14) 第四层输出特征
# # (200, 512)


# # 定义一个1x1卷积层
# conv1x1 = nn.Conv2d(features.shape[0], 512, kernel_size=1)
# features = torch.tensor(features, dtype=torch.float32)
# features = conv1x1(features).cpu().detach().numpy()


# # 线性层的权重
# img_path = f'./datasets/CUB_200_2011/CUB_200_2011/images/{image_set}/{image}.jpg'
# # 对应的原始图像路径

# transformer = transforms.Compose([
#     tr.ToCVImage(),
#     tr.Resize((448, 448), 'linear'),
# #     tr.RandomResizedCrop(size=INIT_TRAINER['image_size'], scale=(1.0, 1.0)),
#     tr.ToTensor(),
#     tr.Normalize(INIT_TRAINER['train_mean'], INIT_TRAINER['train_std']),
#     tr.ToArray(),
# ])

# classes = 200 # 总类别数
# class_idx = 0 # 模型预测类别，也可以从最终结果的csv里面批量读取
# cam_path = f'./analysis/result/v1.0/{current_fold}/'

# # 确保cam_path目录存在
# if not os.path.exists(cam_path):
#     os.makedirs(cam_path, exist_ok=True)

# cams = calculate_CAMs(features, weight, range(classes))  
# save_heatmap(cams, img_path, class_idx, cam_path, transform=transformer)



#------------------------------------------------------------------------------

# CAM
# 计算单个 image_set 的热力图


# import os
# import numpy as np
# import data_utils.transform as tr
# from config import INIT_TRAINER
# from torchvision import transforms
# from converter.common_utils import hdf5_reader
# from analysis.analysis_tools import calculate_CAMs, save_heatmap

# import math
# from PIL import Image

# current_fold = "fold1"
# image_set = "001.Black_footed_Albatross"
# features_dir = f'./analysis/mid_feature/v1.0/{current_fold}/'
# weight_path = f'./analysis/result/v1.0/{current_fold}_fc_weight.npy'
# images_dir = f'./datasets/CUB_200_2011/CUB_200_2011/images/{image_set}/'

# weight = np.load(weight_path)
# # 线性层的权重

# transformer = transforms.Compose([
#     tr.ToCVImage(),
#     tr.Resize((448, 448), 'linear'),
#     tr.ToTensor(),
#     tr.Normalize(INIT_TRAINER['train_mean'], INIT_TRAINER['train_std']),
#     tr.ToArray(),
# ])

# classes = 200  # 总类别数
# cam_path = f'./analysis/result/v1.0/{current_fold}/{image_set}/'

# # 确保cam_path目录存在
# if not os.path.exists(cam_path):
#     os.makedirs(cam_path, exist_ok=True)

# # 获取image_set的文件名后缀
# image_set_suffix = image_set.split('.')[-1].lower()
# print(image_set)

# for image_name in os.listdir(features_dir):
#     if image_name.lower().startswith(image_set_suffix):
#         print(image_name)
        
#         features = hdf5_reader(os.path.join(features_dir, image_name), 'feature_in')
#         img_path = os.path.join(images_dir, f'{image_name}.jpg')
        
#         class_idx = 0  # 模型预测类别，可以根据需要修改或从结果文件中读取
        
#         cams = calculate_CAMs(features, weight, range(classes))
#         save_heatmap(cams, img_path, class_idx, cam_path, transform=transformer)


# def create_square_canvas(image_paths, save_path, grid_size=5):
#     images = [Image.open(img_path) for img_path in image_paths[:grid_size*grid_size]]
#     num_images = len(images)
    
#     if num_images == 0:
#         print("No images to process.")
#         return
    
#     # 取第一张图片的大小
#     img_width, img_height = images[0].size
    
#     # 计算画布大小
#     canvas_width = grid_size * img_width
#     canvas_height = grid_size * img_height
    
#     # 创建一个白色背景的画布
#     canvas = Image.new('RGB', (canvas_width, canvas_height), (255, 255, 255))
    
#     # 将每张图片按正方形布局粘贴到画布上
#     for idx, img in enumerate(images):
#         x_offset = (idx % grid_size) * img_width
#         y_offset = (idx // grid_size) * img_height
#         canvas.paste(img, (x_offset, y_offset))
    
#     # 保存拼接后的图片
#     canvas.save(save_path)

# # 获取cam_path目录下所有的CAM图片路径
# cam_images = [os.path.join(cam_path, img) for img in os.listdir(cam_path) 
#               if img.endswith('.jpg') and img != 'merged.jpg']

# # 只取前25张图片
# cam_images = cam_images[:25]

# merged_image_path = os.path.join(cam_path, 'merged.jpg')
# create_square_canvas(cam_images, merged_image_path, grid_size=5)


#------------------------------------------------------------------------------

# CAM 
# 计算所有图片的热力图


# import os
# import numpy as np
# import data_utils.transform as tr
# from config import INIT_TRAINER
# from torchvision import transforms
# from converter.common_utils import hdf5_reader
# from analysis.analysis_tools import calculate_CAMs, save_heatmap

# import math
# from PIL import Image


# def create_square_canvas(image_paths, save_path, grid_size=5):
#     images = [Image.open(img_path) for img_path in image_paths[:grid_size*grid_size]]
#     num_images = len(images)
    
#     if num_images == 0:
#         print("No images to process.")
#         return
    
#     # 取第一张图片的大小
#     img_width, img_height = images[0].size
    
#     # 计算画布大小
#     canvas_width = grid_size * img_width
#     canvas_height = grid_size * img_height
    
#     # 创建一个白色背景的画布
#     canvas = Image.new('RGB', (canvas_width, canvas_height), (255, 255, 255))
    
#     # 将每张图片按正方形布局粘贴到画布上
#     for idx, img in enumerate(images):
#         x_offset = (idx % grid_size) * img_width
#         y_offset = (idx // grid_size) * img_height
#         canvas.paste(img, (x_offset, y_offset))
    
#     # 保存拼接后的图片
#     canvas.save(save_path)
    

# current_fold = "fold1"
# features_dir = f'./analysis/mid_feature/v1.0/{current_fold}/'
# weight_path = f'./analysis/result/v1.0/{current_fold}_fc_weight.npy'
# images_set_dir = f'./datasets/CUB_200_2011/CUB_200_2011/images/'

# weight = np.load(weight_path)
# # 线性层的权重

# transformer = transforms.Compose([
#     tr.ToCVImage(),
#     tr.Resize((448, 448), 'linear'),
#     tr.ToTensor(),
#     tr.Normalize(INIT_TRAINER['train_mean'], INIT_TRAINER['train_std']),
#     tr.ToArray(),
# ])

# classes = 200  # 总类别数

# for images_set in os.listdir(images_set_dir):
#     print(images_set)
#     images_dir = f'./datasets/CUB_200_2011/CUB_200_2011/images/{images_set}/'
#     cam_path = f'./analysis/result/v1.0/{current_fold}/{images_set}/'

#     # 确保cam_path目录存在
#     if not os.path.exists(cam_path):
#         os.makedirs(cam_path, exist_ok=True)

#     # 获取images_set的文件名后缀
#     images_set_suffix = images_set.split('.')[-1].lower()

#     for image_name in os.listdir(features_dir):
#         if image_name.lower().startswith(images_set_suffix):

#             features = hdf5_reader(os.path.join(features_dir, image_name), 'feature_in')
#             img_path = os.path.join(images_dir, f'{image_name}.jpg')

#             class_idx = 0  # 模型预测类别，可以根据需要修改或从结果文件中读取

#             cams = calculate_CAMs(features, weight, range(classes))
#             save_heatmap(cams, img_path, class_idx, cam_path, transform=transformer)



#     # 获取cam_path目录下所有的CAM图片路径
#     cam_images = [os.path.join(cam_path, img) for img in os.listdir(cam_path) 
#                   if img.endswith('.jpg') and img != 'merged.jpg']

#     # 只取前25张图片
#     cam_images = cam_images[:25]

#     merged_image_path = os.path.join(cam_path, 'merged.jpg')
#     create_square_canvas(cam_images, merged_image_path, grid_size=5)



#------------------------------------------------------------------------------

# Grad-CAM 
# 计算单个 image_set 的热力图 

import os
import numpy as np
import data_utils.transform as tr
from config import INIT_TRAINER
from torchvision import transforms
from converter.common_utils import hdf5_reader
from analysis.analysis_tools import calculate_CAMs, save_heatmap


import torch.nn as nn
import torch
import cv2


import math
from PIL import Image


# 计算 Grad-CAM 的热力图
def calculate_gradcam(features, grads):
    # 计算每个通道的平均梯度
    pooled_gradients = torch.mean(grads, dim=[1, 2])
    
    # 将 pooled_gradients 乘以相应的特征图
    for i in range(features.shape[0]):
        features[i, :, :] *= pooled_gradients[i]

    heatmap=torch.mean(features, dim=0)
    
#     print(heatmap.shape)
        
    # 对热力图进行ReLU操作
    heatmap = np.maximum(heatmap, 0)

    # 标准化热力图
    heatmap /= torch.max(heatmap)
    
    return heatmap.numpy()


# 将 heatmap 叠加到原始图像上
def save_gradcam_heatmap(cams, img_path, cam_path, transform = None):
    img_name = img_path.split("/")[-1]
    img = cv2.imread(img_path)
    if transform is not None:
        img = transform(img)
    h, w, _ = img.shape
    heatmap = cv2.resize(cams, (w, h))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)  
    result = heatmap * 0.3 + img * 0.5
    if not os.path.exists(cam_path):
        os.mkdir(cam_path)
        
    cv2.imwrite(cam_path + img_name + '_' + 'pred_' + '.jpg', result) 

    
def create_square_canvas(image_paths, save_path, grid_size=5):
    images = [Image.open(img_path) for img_path in image_paths[:grid_size*grid_size]]
    num_images = len(images)
    
    if num_images == 0:
        print("No images to process.")
        return
    
    # 取第一张图片的大小
    img_width, img_height = images[0].size
    
    # 计算画布大小
    canvas_width = grid_size * img_width
    canvas_height = grid_size * img_height
    
    # 创建一个白色背景的画布
    canvas = Image.new('RGB', (canvas_width, canvas_height), (255, 255, 255))
    
    # 将每张图片按正方形布局粘贴到画布上
    for idx, img in enumerate(images):
        x_offset = (idx % grid_size) * img_width
        y_offset = (idx // grid_size) * img_height
        canvas.paste(img, (x_offset, y_offset))
    
    # 保存拼接后的图片
    canvas.save(save_path)


current_fold = "fold1"
# image = "Black_Footed_Albatross_0001_796111"
# image = "Black_Footed_Albatross_0002_55"
image_set = "001.Black_footed_Albatross"


features_dir = f'./analysis/mid_feature/v1.0/{current_fold}/'
grad_dir = f'./analysis/grad/v1.0/{current_fold}/'
images_dir = f'./datasets/CUB_200_2011/CUB_200_2011/images/{image_set}/'

transformer = transforms.Compose([
    tr.ToCVImage(),
    tr.Resize((448, 448), 'linear'),
#     tr.RandomResizedCrop(size=INIT_TRAINER['image_size'], scale=(1.0, 1.0)),
    tr.ToTensor(),
    tr.Normalize(INIT_TRAINER['train_mean'], INIT_TRAINER['train_std']),
    tr.ToArray(),
])


cam_path = f'./analysis/result/v1.0/GradCAM/{current_fold}/{image_set}/'
if not os.path.exists(cam_path):
    os.makedirs(cam_path, exist_ok=True)


# 获取image_set的文件名后缀
image_set_suffix = image_set.split('.')[-1].lower()
print(image_set)
    
    
for image_name in os.listdir(features_dir):
    if image_name.lower().startswith(image_set_suffix):
        print(image_name)
    
        img_path = f'./datasets/CUB_200_2011/CUB_200_2011/images/{image_set}/{image_name}.jpg'
        features = hdf5_reader(f'./analysis/mid_feature/v1.0/{current_fold}/{image_name}','feature_out')
        grads = hdf5_reader(f'./analysis/grad/v1.0/{current_fold}/{image_name}','grad_out')

        # 将 features 和 grads 转换为 tensor
        features_tensor = torch.tensor(features, dtype=torch.float32)
        grads_tensor = torch.tensor(grads, dtype=torch.float32)

        # 计算热力图
        heatmap = calculate_gradcam(features_tensor, grads_tensor)

        # 保存叠加结果
        save_gradcam_heatmap(heatmap, img_path, cam_path, transform=transformer)

# 获取cam_path目录下所有的CAM图片路径
cam_images = [os.path.join(cam_path, img) for img in os.listdir(cam_path) 
              if img.endswith('.jpg') and img != 'merged.jpg']

# 只取前25张图片
cam_images = cam_images[:25]

merged_image_path = os.path.join(cam_path, 'merged.jpg')
create_square_canvas(cam_images, merged_image_path, grid_size=5)