import numpy as np
from PIL import Image
import os
import torch


img_path = 'real_dog'
resize_path = 'real_dog_augment'
'''
for i in os.listdir(img_path):
    im = Image.open(os.path.join(img_path, i))
    out = im.resize((128, 128))
    out.save(os.path.join(resize_path, i))
'''
data_mat = np.zeros((1, 128, 128, 3))
for i in os.listdir(resize_path):
    im = Image.open(os.path.join(resize_path, i))
    arr = np.expand_dims(np.asanyarray(im), axis=0)  # print(arr.shape) (1, 256, 256, 3)
    data_mat = np.concatenate([data_mat, arr], axis=0)
print(type(data_mat), data_mat.shape)  # <class 'numpy.ndarray'> (132, 256, 256, 3)
data_mat = data_mat[1:]
print(type(data_mat), data_mat.shape)  # <class 'numpy.ndarray'> (131, 256, 256, 3)
data_mat = np.transpose(data_mat, (0, 3, 1, 2))
print(type(data_mat), data_mat.shape)  # <class 'numpy.ndarray'> (131, 3, 256, 256)
torch_data = torch.from_numpy(data_mat)
print(torch_data.shape)  # torch.Size([131, 3, 256, 256])
torch.save(torch_data, 'real_dog_data.pt')
