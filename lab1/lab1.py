import torch
from torch.nn.functional import conv2d
import numpy as np

def convolution2d(input_data, weight_tensor, bias = None,
                 stride = 1, padding = (0, 0), dilation = 1):
    if not isinstance(input_data, np.ndarray) or not isinstance(weight_tensor, np.ndarray):
        raise TypeError("Convolution2d takes tensors as numpy array with shape (n, m)")
    padding = padding[0] if isinstance(padding, tuple) else 0

    image_height, image_width = input_data.shape
    weight_height, weight_width = weight_tensor.shape
    H_out = int((image_height - dilation * (weight_height - 1) - 1 + 2 * padding) / stride) + 1
    W_out = int((image_width - dilation * (weight_width - 1) - 1 + 2 * padding) / stride) + 1

    input_data = np.pad(input_data, padding, mode='constant') if padding > 0 else input_data
    result = np.array([
        [np.sum(input_data[y * stride:y * stride + weight_height,
                x * stride:x * stride + weight_width] * weight_tensor)
            for x in range(W_out)] for y in range(H_out)
    ])

    if bias:
        result += bias

    return result