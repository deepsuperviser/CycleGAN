import numpy as np
from PIL import Image
import os
import torch


img_path = 'paint_dog'
resize_path = 'paint_dog_augment'
'''列举每张图像的名称并将源图像的大小改为256
for i in os.listdir(img_path):
    im = Image.open(os.path.join(img_path, i))
    out = im.resize((128, 128))
    out.save(os.path.join(resize_path, i))
'''
data_mat = np.zeros((1, 128, 128, 3))  # <class 'numpy.ndarray'> (256, 256, 3)
for i in os.listdir(resize_path):
    im = Image.open(os.path.join(resize_path, i))
    arr = np.expand_dims(np.asanyarray(im), axis=0)  # print(arr.shape) (1, 256, 256, 3)
    data_mat = np.concatenate([data_mat, arr], axis=0)
print(type(data_mat), data_mat.shape)  # <class 'numpy.ndarray'> (129, 256, 256, 3)
data_mat = data_mat[1:]
print(type(data_mat), data_mat.shape)  # <class 'numpy.ndarray'> (128, 256, 256, 3)
data_mat = np.transpose(data_mat, (0, 3, 1, 2))
print(type(data_mat), data_mat.shape)  # <class 'numpy.ndarray'> (128, 3, 256, 256)
torch_data = torch.from_numpy(data_mat)
print(torch_data.shape)  # torch.Size([128, 3, 256, 256])
torch.save(torch_data, 'paint_dog_data.pt')
'''读取的图片转为数组
im = Image.open('dog_painting/.jpg')
a = np.asanyarray(im)
print(type(a), a.shape)
'''

