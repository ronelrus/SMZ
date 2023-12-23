from torch.nn.functional import conv2d
from lab1 import convolution2d
from numpy import allclose
import torch

if __name__ == "__main__":
    test = torch.randn(1,1, 5, 5)
    weights = torch.randn(1, 1, 1, 1)
    assert allclose(
        convolution2d(test[0,0].numpy(), weights[0, 0].numpy()),
        conv2d(test, weights)[0, 0]
    )

    test = torch.randn(1,1, 5, 5)
    weights = torch.randn(1, 1, 1, 1)
    assert allclose(
        convolution2d(test[0,0].numpy(), weights[0, 0].numpy()),
        conv2d(test, weights)[0, 0]
    )

    test = torch.randn(1,1, 5, 5)
    weights = torch.randn(1, 1, 1, 1)
    assert allclose(
        convolution2d(test[0,0].numpy(), weights[0, 0].numpy(), stride=2),
        conv2d(test, weights, stride=2)[0, 0]
    )

    test = torch.randn(1,1, 5, 5)
    weights = torch.randn(1, 1, 1, 1)
    assert allclose(
        convolution2d(test[0,0].numpy(), weights[0, 0].numpy(), dilation=3),
        conv2d(test, weights, dilation=3)[0, 0]
    )

    test = torch.randn(1,1, 5, 5)
    weights = torch.randn(1, 1, 1, 1)
    assert allclose(
        convolution2d(test[0,0].numpy(), weights[0, 0].numpy(), dilation=3, stride=2),
        conv2d(test, weights, dilation=3, stride=2)[0, 0]
    )
    print("Ok")