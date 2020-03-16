import torch
import os
import math

def lerp(a, b, l):
    return (1 - l) * a + l * b


def rgb2lum(image):
    image = 0.27 * image[:, 0, :, :] + 0.67 * image[:, 1, :, :] + 0.06 * image[:, 2, :, :]
    return image[:, None, :, :]

def atanh(x):
    return 0.5*math.log((1+x)/(1-x))


def tanh01(x):
    return torch.tanh(x) * 0.5 + 0.5


def tanh_range(l, r, initial=None):

    def get_activation(left, right, initial):

        def activation(x):
            if initial is not None:
                bias = atanh(2 * (initial - left) / (right - left) - 1)
            else:
                bias = 0
            return tanh01(x + bias) * (right - left) + left

        return activation

    return get_activation(l, r, initial)

# https://github.com/pytorch/pytorch/issues/229
def flip(x, dim):
    indices = [slice(None)] * x.dim()
    indices[dim] = torch.arange(x.size(dim) - 1, -1, -1,
                                dtype=torch.long, device=x.device)
    return x[tuple(indices)]


# the file helper
def check_dir(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)
