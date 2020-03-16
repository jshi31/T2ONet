import os
import json
import pdb
from shutil import copyfile

import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import minimize
import kornia

from utils.operator_utils import tanh_range, rgb2lum, lerp
from utils.visual_utils import tensor2img
from pyutils.edgeconnect.src.config import Config
from pyutils.edgeconnect.src.edge_connect import EdgeConnect


# Cache for Inpainting model
global InpaintModel
InpaintModel = None
device = 'cuda' if torch.cuda.is_available() else 'cpu'


class Operator(nn.Module):
    """
    Given operator choice, predict the operator parameter and execute the operator
    net: must be rgb image as input.
    img passed into Operator should be RGB and in [0, 1]
    """
    def __init__(self, cfg):
        super(Operator, self).__init__()
        self.cfg = cfg
        self.is_discrete = cfg.discrete_param
        self.channels = 2 * cfg.hidden_size
        # Specified in child classes
        self.num_op_param = None
        self.short_name = None
        self.op_param = None
        self.param_sample_flag = False

    def setup(self):
        """must be called by child class"""
        output_dim = self.get_num_op_param()
        self.fc1 = nn.Linear(self.channels, self.cfg.operator_fc_dim)
        # self.bn1 = nn.BatchNorm1d(self.cfg.operator_fc_dim)
        # self.ln1 = nn.LayerNorm(self.cfg.operator_fc_dim)
        self.lrelu = nn.LeakyReLU(inplace=True)
        if not self.cfg.discrete_param:
            self.fc2 = nn.Linear(self.cfg.operator_fc_dim, output_dim)
        else:
            # TODO: currently for discrete operation only support 1 operator parameter
            self.fc2 = nn.Linear(self.cfg.operator_fc_dim, self.cfg.discrete_step)
        self.dist, self.ub, self.lb, self.initial = self.get_param_noise_distribution()

    def get_param_noise(self, bs):
        noise = self.dist.sample([bs])
        noise = (F.relu(noise) * (self.ub - self.initial) + F.relu(-noise) * (self.initial - self.lb)) / 3 * self.cfg.param_noise_factor
        return noise

    def set_param_sample_flag(self, flag):
        self.param_sample_flag = flag

    def get_short_name(self):
        assert self.short_name
        return self.short_name

    def get_num_op_param(self):
        assert self.num_op_param is not None, 'Must specify the number of parameter'
        return self.num_op_param

    def extract_parameters(self, features):
        """
        get the filter parameter
        :param features:
        :return: param
        """
        features = self.fc1(features)
        # features = self.bn1(features)
        # features = self.ln1(features)
        features = self.lrelu(features)
        features = self.fc2(features)
        if self.is_discrete:
            param = self.op_param_classifier(features)
        else:
            param = self.op_param_regressor(features)
        return param

    def sample_parameters(self, probs):
        """
        :param probs: (bs, n_cls)
        :return:
        """
        dist = torch.distributions.Categorical(probs=probs)  # better initialize with logits (bs, n_cls)
        ind = dist.sample()  # (bs,)
        return ind


    # implement in child class
    def op_param_regressor(self, features):
        raise NotImplementedError

    def op_param_classifier(self, features):
        raise NotImplementedError

    # process the image inside the mask
    def process(self, img, param):
        raise NotImplementedError

    # Apply the whole filter with masking
    def execute(self, img, mask=None, features=None, specified_param=None, has_noise=False):
        assert (features is None) ^ (specified_param is None)
        if features is not None:
            param = self.extract_parameters(features) # (bs, n_param)
        else:
            param = specified_param
        if has_noise:
            param_noise = self.get_param_noise(img.shape[0]).to(img.device)
            param = param + param_noise
            param = torch.clamp(param, self.lb, self.ub)
        self.param = param
        if mask is None: mask = torch.ones_like(img)

        self.mask = mask # self.mask is only used for inpainting op

        # TODO: process might need to input image feature, for the generation task
        output = self.process(img, param) # process whole image
        output = output * mask + img * (1 - mask) # combine masked part and not masked part
        output = torch.clamp(output, 0, 1)
        return output

    def param_loss_fn(self):
        """
        :return: function
        """
        raise NotImplementedError

    def get_param(self):
        return self.op_param

    def visualize_op(self, img):
        raise NotImplementedError


    def discretize(self, start, end, num):
        """ discretize a continuous range
        two types of input:
        1. start == 0
        2. start == -end
        :param start:int
        :param end:int
        :param num:int
        :return:cates ndarray
        """
        if start == 0:
            cates = np.delete(np.linspace(start, end, num + 1), 0)
        elif start == -end:
            cates = np.delete(np.linspace(start, end, num + 1), num / 2)
        else:
            assert False, 'Error: the discretize condition is not satisfied!'
        return cates.astype(np.float32)

    def select_param_ind(self, features):
        self.log_prob = F.log_softmax(features, 1) # (bs, 10)
        # explore
        param_probs = torch.exp(self.log_prob)  # (bs, 10)
        param_probs = param_probs * (1 - self.cfg.explore_prob) + self.cfg.explore_prob * 1.0 / self.cfg.discrete_step
        param_probs = param_probs / (param_probs.sum(1, keepdim=True) + 1e-30)

        if self.param_sample_flag:
            ind = self.sample_parameters(param_probs)
        else:
            _, ind = torch.max(self.log_prob, dim=1) # (bs,)
        return ind

    def get_param_range(self):
        raise NotImplementedError

    def get_param_noise_distribution(self):
        ub, lb, initial = self.get_param_range()
        dist = torch.distributions.normal.Normal(torch.zeros(self.num_op_param), torch.ones(self.num_op_param))
        return dist, ub, lb, initial


class ExposureOperator(Operator):
    def __init__(self, cfg):
        super(ExposureOperator, self).__init__(cfg)
        self.short_name = 'exposure'
        self.num_op_param = 1
        self.setup()

    def op_param_regressor(self, features):

        bnd = self.cfg.exposure_range
        return tanh_range(-bnd, bnd, initial=0)(features)

    def op_param_classifier(self, features):
        """
        It is unnatural to return two values
        :param features:
        :return:
        """
        cand_params = torch.from_numpy(self.discretize(-self.cfg.exposure_range, self.cfg.exposure_range, self.cfg.discrete_step)).unsqueeze(0).to(features.device) # (1, 10)
        ind = self.select_param_ind(features) # (bs,)
        param = torch.index_select(cand_params, 1, ind) # (1, 1)
        return param, ind

    def process(self, img, param):
        return img * torch.exp(param.unsqueeze(-1).unsqueeze(-1) * np.log(2))

    def param_loss_fn(self):
        return F.mse_loss

    def get_param_range(self):
        ub = self.cfg.exposure_range
        lb = -self.cfg.exposure_range
        initial = 0
        return ub, lb, initial

    def visualize_op(self):
        pass

class ContrastOperator(Operator):
    def __init__(self, cfg):
        super(ContrastOperator, self).__init__(cfg)
        self.short_name = 'contrast'
        self.num_op_param = 1
        self.setup()

    def op_param_regressor(self, features):
        return torch.tanh(features)

    def op_param_classifier(self, features):
        cand_params = torch.from_numpy(self.discretize(-1, 1, self.cfg.discrete_step)).unsqueeze(0).to(features.device) # (1, 10)
        ind = self.select_param_ind(features) # (bs,)
        param = torch.index_select(cand_params, 1, ind) # (1, 1)
        return param, ind

    def process(self, img, param):
        luminance = torch.min(torch.max(rgb2lum(img), torch.tensor(0.0, device=img.device)),
                              torch.tensor(1.0, device=img.device))
        contrast_lum = -torch.cos(np.pi * luminance) * 0.5 + 0.5
        contrast_image = img / (luminance + 1e-6) * contrast_lum
        return lerp(img, contrast_image, param.unsqueeze(-1).unsqueeze(-1))

    def param_loss_fn(self):
        return F.mse_loss

    def get_param_range(self):
        ub = 1
        lb = -1
        initial = 0
        return ub, lb, initial

    def visualize_op(self):
        pass

class BrightnessOperator(Operator):
    def __init__(self, cfg):
        super(BrightnessOperator, self).__init__(cfg)
        self.short_name = 'brightness'
        self.num_op_param = 1
        self.setup()

    def op_param_regressor(self, features):
        bnd = self.cfg.brightness_range
        # return features
        return tanh_range(-bnd, bnd, initial=0)(features)

    def op_param_classifier(self, features):
        cand_params = torch.from_numpy(self.discretize(-self.cfg.brightness_range, self.cfg.brightness_range, self.cfg.discrete_step)).unsqueeze(0).to(features.device) # (1, 10)
        ind = self.select_param_ind(features) # (bs,)
        param = torch.index_select(cand_params, 1, ind) # (bs, 1)
        return param, ind

    def process(self, img, param):
        hsv = kornia.rgb_to_hsv(img)
        h, s, v = torch.chunk(hsv, chunks=3, dim=1)
        v_out = (v*(1 + param.unsqueeze(-1).unsqueeze(-1))).clamp(0, 1)
        hsv_out = torch.cat([h, s, v_out], dim=1)
        out = kornia.hsv_to_rgb(hsv_out)
        return out

    def param_loss_fn(self):
        return F.mse_loss

    def get_param_range(self):
        ub = self.cfg.brightness_range
        lb = -self.cfg.brightness_range
        initial = 0
        return ub, lb, initial

    def visualize_op(self, img):
        pass


class BNWOperator(Operator):

    def __init__(self, cfg):
        super(BNWOperator, self).__init__(cfg)
        self.short_name = 'black&white'
        self.num_op_param = 1
        self.setup()

    def op_param_regressor(self, features):
        return torch.sigmoid(features)

    def op_param_classifier(self, features):
        cand_params = torch.from_numpy(self.discretize(-1, 1, self.cfg.discrete_step)).unsqueeze(0).to(features.device) # (1, 10)
        ind = self.select_param_ind(features) # (bs,)
        param = torch.index_select(cand_params, 1, ind) # (bs, 1)
        return param, ind

    def process(self, img, param):
        luminance = rgb2lum(img)
        return lerp(img, luminance, param.unsqueeze(-1).unsqueeze(-1))

    def param_loss_fn(self):
        return F.mse_loss

    def get_param_range(self):
        ub = 1
        lb = 0
        initial = (ub + lb) / 2
        return ub, lb, initial

    def visualize_op(self, img):
        pass


class SharpnessOperator(Operator):
    def __init__(self, cfg):
        super(SharpnessOperator, self).__init__(cfg)
        self.short_name = 'sharpness'
        self.num_op_param = 1
        self.setup()
        self.kernel = torch.tensor([[[[0, -1, 0], [-1, 4, -1], [0, -1, 0]]]], dtype=torch.float)

    def op_param_regressor(self, features):
        # scope [0, 1] looks fine
        # return features
        return torch.sigmoid(features) * self.cfg.sharpness_range

    def op_param_classifier(self, features):
        cand_params = torch.from_numpy(self.discretize(0, 1, self.cfg.discrete_step)).unsqueeze(0).to(features.device) # (1, 10)
        ind = self.select_param_ind(features)  # (bs,)
        param = torch.index_select(cand_params, 1, ind)  # (bs, 1)
        return param, ind

    def process(self, img, param):
        img_r, img_g, img_b = img.split([1, 1, 1], 1)
        delta_r = F.conv2d(img_r, self.kernel.to(img.device), padding=1)
        delta_g = F.conv2d(img_g, self.kernel.to(img.device), padding=1)
        delta_b = F.conv2d(img_b, self.kernel.to(img.device), padding=1)
        delta = torch.cat((delta_r, delta_g, delta_b), 1)
        out = img + param.unsqueeze(-1).unsqueeze(-1) * delta
        return out

    def param_loss_fn(self):
        return F.mse_loss

    def get_param_range(self):
        ub = self.cfg.sharpness_range
        lb = 0
        initial = ub / 2
        return ub, lb, initial

    def visualize_op(self, img):
        pass


class BlurOperator(Operator):
    def __init__(self, cfg):
        super(BlurOperator, self).__init__(cfg)
        self.short_name = 'blur'
        self.num_op_param = 1
        self.setup()
        # TODO: the list is just not let the model.cuda() affect the gaussian filter. It should be deleted in the future
        self.gaussian_filter = [get_gaussian_kernel(3, 2, 1).to(device)]

    def op_param_regressor(self, features):
        # scope [0, 1] looks fine
        return torch.sigmoid(features)

    def op_param_classifier(self, features):
        cand_params = torch.from_numpy(self.discretize(0, 1, self.cfg.discrete_step)).unsqueeze(0).to(features.device) # (1, 10)
        ind = self.select_param_ind(features) # (bs,)
        param = torch.index_select(cand_params, 1, ind) # (bs, 1)
        return param, ind

    def process(self, img, param):
        img_r, img_g, img_b = img.split([1, 1, 1], 1)
        blur_r = self.gaussian_filter[0](img_r)
        blur_g = self.gaussian_filter[0](img_g)
        blur_b = self.gaussian_filter[0](img_b)
        blur_img = torch.cat((blur_r, blur_g, blur_b), 1)
        out = lerp(img, blur_img, param.unsqueeze(-1).unsqueeze(-1))
        return out

    def param_loss_fn(self):
        return F.mse_loss

    def get_param_range(self):
        ub = 1
        lb = 0
        initial = (ub + lb) / 2
        return ub, lb, initial

    def visualize_op(self, img):
        pass


class HueOperator(Operator):
    def __init__(self, cfg):
        super(HueOperator, self).__init__(cfg)
        self.short_name = 'hue_'
        self.num_op_param = 1
        self.setup()

    def op_param_regressor(self, features):
        return features
        return torch.sigmoid(features)

    def op_param_classifier(self, features):
        cand_params = torch.from_numpy(self.discretize(-1, 1, self.cfg.discrete_step)).unsqueeze(0).to(features.device) # (1, 10)
        ind = self.select_param_ind(features) # (bs,)
        param = torch.index_select(cand_params, 1, ind) # (bs, 1)
        return param, ind

    def process(self, img, param):
        hsv = kornia.rgb_to_hsv(img)
        hue, sat, value = hsv.split([1, 1, 1], 1)
        # out_hsv = torch.cat((param[:, 0:1].expand_as(value),
        #                      param[:, 1:2].expand_as(value), value), 1)
        out_hsv = torch.cat((param.expand_as(value),
                             sat, value), 1)
        out = kornia.hsv_to_rgb(out_hsv)
        return out

    def param_loss_fn(self):
        return F.mse_loss

    def get_param_range(self):
        ub = 1
        lb = 0
        initial = (ub + lb) / 2
        return ub, lb, initial

    def visualize_op(self, img):
        pass


class SaturationOperator(Operator):
    def __init__(self, cfg):
        super(SaturationOperator, self).__init__(cfg)
        self.short_name = 'saturation'
        self.num_op_param = 1
        self.setup()

    def op_param_regressor(self, features):
        # return tanh_range(self.cfg.saturation_range[0], self.cfg.saturation_range[1], initial=0)(features)
        # return features
        param = torch.tanh(F.relu(features)) * self.cfg.saturation_range[1] + torch.tanh(F.relu(-features)) * self.cfg.saturation_range[0]
        return param

    def op_param_classifier(self, features):
        cand_params = torch.from_numpy(self.discretize(-0.5, 0.5, self.cfg.discrete_step)).unsqueeze(0).to(features.device) # (1, 10)
        ind = self.select_param_ind(features) # (bs,)
        param = torch.index_select(cand_params, 1, ind) # (bs, 1)
        return param, ind

    def process(self, img, param):
        hsv = kornia.rgb_to_hsv(img)
        h, s, v = torch.chunk(hsv, chunks=3, dim=1)
        s_out = (s*(1 + param.unsqueeze(-1).unsqueeze(-1))).clamp(0, 1)
        hsv_out = torch.cat([h, s_out, v], dim=1)
        out = kornia.hsv_to_rgb(hsv_out)
        return out

    def param_loss_fn(self):
        return F.mse_loss

    def get_param_range(self):
        ub = self.cfg.saturation_range[1]
        lb = self.cfg.saturation_range[0]
        initial = 0
        return ub, lb, initial

    def visualize_op(self, img):
        pass


class WhiteOperator(Operator):
    def __init__(self, cfg):
        super(WhiteOperator, self).__init__(cfg)
        self.short_name = 'color_bg'
        self.num_op_param = 1
        self.setup()

    def op_param_regressor(self, features):
        return torch.sigmoid(features)

    def op_param_classifier(self, features):
        cand_params = torch.from_numpy(self.discretize(-1, 1, self.cfg.discrete_step)).unsqueeze(0).to(features.device) # (1, 10)
        ind = self.select_param_ind(features) # (bs,)
        param = torch.index_select(cand_params, 1, ind) # (bs, 1)
        return param, ind

    def process(self, img, param):
        out = torch.ones_like(img)
        return out

    def param_loss_fn(self):
        return F.mse_loss

    def get_param_range(self):
        ub = 1
        lb = 0
        initial = (ub + lb) / 2
        return ub, lb, initial

    def visualize_op(self, img):
        pass


class ImprovedWhiteBalanceOperator(Operator):

    def __init__(self, cfg):
        super(ImprovedWhiteBalanceOperator, self).__init__(cfg)
        self.short_name = 'whitebalance'
        self.num_op_param = 3
        self.setup()

    def op_param_regressor(self, features):
        log_wb_range = 0.5
        mask = torch.tensor([0, 1, 1], dtype=torch.float).view(1, 3).to(features.device)
        features = features * mask
        color_scaling = torch.exp(tanh_range(-log_wb_range, log_wb_range)(features))

        # There will be no division by zero here unless the WB range lower bound is 0
        # normalize by luminance
        color_scaling = color_scaling * 1.0 / (
                                       1e-5 + 0.27 * color_scaling[:, 0] + 0.67 * color_scaling[:, 1] +
                                       0.06 * color_scaling[:, 2]).unsqueeze(1)
        return color_scaling

    def process(self, img, param):
        return img * param.unsqueeze(-1).unsqueeze(-1)

    def get_param_range(self):
        ub = 1.8
        lb = 0.4
        initial = (lb + ub) / 2
        return ub, lb, initial

class ToneOperator(Operator):

    def __init__(self, cfg):
        super(ToneOperator, self).__init__(cfg)
        self.curve_steps = cfg.curve_steps
        self.short_name = 'tone'
        self.num_op_param = cfg.curve_steps
        self.setup()

    def op_param_regressor(self, features):
        return features
        # tone_curve = tanh_range(self.cfg.tone_curve_range[0], self.cfg.tone_curve_range[1])(features)
        # return tone_curve

    def process(self, img, param):
        """
        :param img: (bs, 3, h, w)
        :param param: (bs, n)
        :return: img: (bs, 3, h, w)
        """
        # img = tf.minimum(img, 1.0)
        tone_curve = param.view(-1, 1, self.cfg.curve_steps, 1, 1)
        tone_curve_sum = tone_curve.sum(2) + 1e-10
        total_img = torch.zeros_like(img)
        for i in range(self.cfg.curve_steps):
            total_img = total_img + torch.clamp(img - 1.0 * i / self.cfg.curve_steps, 0, 1.0 / self.cfg.curve_steps) \
                           * tone_curve[:, :, i, :, :]
        img = total_img * self.cfg.curve_steps / tone_curve_sum
        return img

    def get_param_range(self):
        ub = self.cfg.tone_curve_range[1]
        lb = self.cfg.tone_curve_range[0]
        initial = (ub + lb) / 2
        return ub, lb, initial

class ColorOperator(Operator):

    def __init__(self, cfg):
        super(ColorOperator, self).__init__(cfg)
        self.curve_steps = cfg.curve_steps
        self.short_name = 'hue'
        self.num_op_param = 3 * cfg.curve_steps
        self.setup()

    def op_param_regressor(self, features):
        return features
        # color_curve = tanh_range(self.cfg.color_curve_range[0], self.cfg.color_curve_range[1], initial=1)(features)
        # return color_curve

    def process(self, img, param):
        color_curve = param.view(-1, 3, self.cfg.curve_steps, 1, 1)
        # There will be no division by zero here unless the color filter range lower bound is 0
        color_curve_sum = color_curve.sum(2) + 1e-10
        total_img = torch.zeros_like(img)
        for i in range(self.cfg.curve_steps):
            total_img += torch.clamp(img - 1.0 * i / self.cfg.curve_steps, 0, 1.0 / self.cfg.curve_steps) * \
                           color_curve[:, :, i, :, :]
        total_img *= self.cfg.curve_steps / color_curve_sum
        return total_img

    def get_param_range(self):
        ub = self.cfg.color_curve_range[1]
        lb = self.cfg.color_curve_range[0]
        initial = (ub + lb) / 2
        return ub, lb, initial


class InpaintOperator(Operator):
    def __init__(self, cfg):
        super(InpaintOperator, self).__init__(cfg)
        self.short_name = 'inpaint_obj'
        self.num_op_param = 1
        self.setup()
        cur_dir = os.path.dirname(os.path.realpath(__file__))
        checkpoint = os.path.join(cur_dir, 'pyutils/edge-connect/checkpoints/places2')
        config_path = os.path.join(checkpoint, 'config.yml.example')
        if not os.path.exists(config_path):
            copyfile(os.path.join(cur_dir, './pyutils/edge-connect/config.yml.example'), config_path)
        self.config = Config(config_path)
        self.config.MODE = 2
        self.config.MODEL = 3
        # data loading
        if torch.cuda.is_available():
            self.config.DEVICE = torch.device("cuda")
            torch.backends.cudnn.benchmark = True   # cudnn auto-tuner
        else:
            self.config.DEVICE = torch.device("cpu")
        # conditional load InpaintModel
        global InpaintModel
        if InpaintModel is None:
            InpaintModel = EdgeConnect(self.config)
            InpaintModel.load()
        self.model = InpaintModel

    def op_param_regressor(self, features):
        # pseudo parameter output
        bs, _ = features.shape
        cuda_check = features.is_cuda
        if cuda_check:
            get_cuda_device = features.get_device()
        return torch.zeros((bs, self.num_op_param), requires_grad=True, device='cuda:{}'.format(get_cuda_device))

    def op_param_classifier(self, features):
        # pseudo parameter output
        self.log_prob = F.log_softmax(features, 1) # (bs, 10)
        bs, _ = features.shape
        cuda_check = features.is_cuda
        if cuda_check:
            get_cuda_device = features.get_device()
        return torch.zeros((bs, self.num_op_param), dtype=torch.float, requires_grad=True, device='cuda:{}'.format(get_cuda_device)), \
               torch.zeros((bs,), dtype=torch.float, device='cuda:{}'.format(get_cuda_device))

    def param_loss_fn(self):
        def psudo_loss_fn(pred, tgt):
            return 0
        return psudo_loss_fn

    def get_param_range(self):
        ub = 0
        lb = 0
        initial = 0
        return ub, lb, initial

    def process(self, img, param):
        out = self.model.test(img, self.mask)
        return out


def get_gaussian_kernel(kernel_size=3, sigma=2, channels=3):
    # Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
    x_coord = torch.arange(kernel_size)
    x_grid = x_coord.repeat(kernel_size).view(kernel_size, kernel_size)
    y_grid = x_grid.t()
    xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()

    mean = (kernel_size - 1) / 2.
    variance = sigma ** 2.

    # Calculate the 2-dimensional gaussian kernel which is
    # the product of two gaussian distributions for two different
    # variables (in this case called x and y)
    gaussian_kernel = (1. / (2. * np.pi * variance)) * \
                      torch.exp(
                          -torch.sum((xy_grid - mean) ** 2., dim=-1) / \
                          (2 * variance)
                      )

    # Make sure sum of values in gaussian kernel equals 1.
    gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

    # Reshape to 2d depthwise convolutional weight
    gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
    gaussian_kernel = gaussian_kernel.repeat(channels, 1, 1, 1)

    gaussian_filter = nn.Conv2d(in_channels=channels, out_channels=channels,
                                kernel_size=kernel_size, groups=channels, padding=1, bias=False)

    gaussian_filter.weight.data = gaussian_kernel
    gaussian_filter.weight.requires_grad = False

    return gaussian_filter


def get_color_from_name(name):
    """
    get hue and saturation from color name
    :param name: color name
    :return: h, s
    """
    name_hsv_map = json.load(open('data/color/name_hsv_map.json'))
    colors = []
    for key in name_hsv_map.keys():
        if name in key:
            colors.append(key)

    # h, s, v = name_hsv_map[colors[0]] # choose the first matched color by default
    return [name_hsv_map[color][:2] for color in colors], colors


#################################
### the separate unit testing ###
#################################
"""img: torch tensor (1, 3, h, w)"""

def test_exposure(img, mask, cfg):
    E_op = ExposureOperator(cfg)
    for alpha in np.arange(-3.5, 3.6, 0.5):
        param = torch.tensor([[alpha]], dtype=torch.float) # [-3.5, 3.5]
        out = E_op.execute(img, mask=mask, specified_param=param)
        out = tensor2img(out)
        cv2.imshow('exposure {:.2f} img'.format(alpha), out)
        cv2.waitKey(0)
        # cv2.imwrite('output/operator_analysis/exposure/{:.2f}.jpg'.format(alpha), out)


def test_contrast(img, mask, cfg):
    Ct_op = ContrastOperator(cfg)
    for alpha in np.linspace(0, 1, num=11):
        param = torch.tensor([[alpha]], dtype=torch.float) # [-1, 1]
        out = Ct_op.execute(img, mask=mask, specified_param=param)
        out = tensor2img(out)
        cv2.imshow('contrast {:.2f} img'.format(alpha), out)
        cv2.waitKey(0)
        # cv2.imwrite('output/operator_analysis/contrast/{:.2f}.jpg'.format(alpha), out)


def test_brightness(img, mask, cfg):
    Br_op = BrightnessOperator(cfg)
    for alpha in np.linspace(-0.3, 0.3, num=11):
        param = torch.tensor([[alpha]], dtype=torch.float) # [-1, 1]
        out = Br_op.execute(img, mask=mask, specified_param=param)
        out = tensor2img(out)
        cv2.imshow('brightness {:.2f} img'.format(alpha), out)
        cv2.waitKey(0)
        # cv2.imwrite('output/operator_analysis/brightness/{:.2f}.jpg'.format(alpha), out)

def test_saturation(img, mask, cfg):
    Sat_op = SaturationOperator(cfg)
    for alpha in np.linspace(-0.2, 0.8, num=21):
        param = torch.tensor([[alpha]], dtype=torch.float) # [-1, 1]
        out = Sat_op.execute(img, mask=mask, specified_param=param)
        out = tensor2img(out)
        cv2.imshow('brightness {:.2f} img'.format(alpha), out)
        cv2.waitKey(0)
        # save_dir = os.path.join(cfg.operator_analysis_dir, 'saturation')
        # os.makedirs(save_dir, exist_ok=True)
        # cv2.imwrite(os.path.join(save_dir, 'saturation {:.2f} img.jpg'.format(alpha)), out)

def test_bnw(img, mask, cfg):
    BW_op = BNWOperator(cfg)
    param = torch.tensor([[0.9]], dtype=torch.float) # [-1, 1]
    out = BW_op.execute(img, mask=mask, specified_param=param)
    out = tensor2img(out)
    cv2.imshow('black&white img', out)
    cv2.waitKey(0)


def test_sharpness(img, mask, cfg):
    Sh_op = SharpnessOperator(cfg)
    for alpha in np.linspace(0, 2, num=11):
        param = torch.tensor([[alpha]], dtype=torch.float) # [-1, 1]
        out = Sh_op.execute(img, mask=mask, specified_param=param)
        out = tensor2img(out)
        cv2.imshow('sharpness {:.2f} img'.format(alpha), out)
        # cv2.waitKey(0)
        # cv2.imwrite('output/operator_analysis/sharpness/{:.2f}.jpg'.format(alpha), out)

def test_tone(img, mask, cfg):
    Tone_op = ToneOperator(cfg)
    alpha = np.random.rand(1, cfg.curve_steps)
    param = torch.tensor(alpha, dtype=torch.float) #
    out = Tone_op.execute(img, mask=mask, specified_param=param)
    out = tensor2img(out)
    img = tensor2img(img)
    cv2.imshow('ori img', img)
    cv2.imshow('tone {:.2f} img'.format(alpha[0, 0]), out)
    # cv2.waitKey(0)
    # cv2.imwrite('output/operator_analysis/sharpness/{:.2f}.jpg'.format(alpha), out)


def test_color(img, mask, cfg):
    Color_op = ColorOperator(cfg)
    alpha = np.random.rand(1, cfg.curve_steps * 3)
    param = torch.tensor(alpha, dtype=torch.float) #
    out = Color_op.execute(img, mask=mask, specified_param=param)
    out = tensor2img(out)
    img = tensor2img(img)
    cv2.imshow('ori img', img)
    cv2.imshow('tone {:.2f} img'.format(alpha[0, 0]), out)
    # cv2.waitKey(0)
    # cv2.imwrite('output/operator_analysis/sharpness/{:.2f}.jpg'.format(alpha), out)


def test_blur(img, mask, cfg):
    Blr_op = BlurOperator(cfg)
    for alpha in np.linspace(0, 1, num=11):
        param = torch.tensor([[alpha]], dtype=torch.float) # [-1, 1]
        out = Blr_op.execute(img, mask=mask, specified_param=param)
        out = tensor2img(out)
        cv2.imshow('blur {:.2f} img'.format(alpha), out)
        cv2.waitKey(0)



def test_hue(img, mask, cfg):
    # firstly from color name to get the color value
    Hue_op = HueOperator(cfg)
    for alpha in np.linspace(0, 1, num=11):
        param = torch.tensor([[alpha]], dtype=torch.float) # [-1, 1]
        out = Hue_op.execute(img, mask=mask, specified_param=param)
        out = tensor2img(out)
        save_dir = os.path.join(cfg.operator_analysis_dir, 'hue')
        os.makedirs(save_dir, exist_ok=True)
        cv2.imwrite(os.path.join(save_dir, 'hue {:.2f} img.jpg'.format(alpha)), out)

    # firstly from color name to get the color value
    root_color_name = 'brown'
    color_hs, color_names = get_color_from_name(root_color_name)
    Cl_op = ColorOperator(cfg)
    for i in range(len(color_names)):
        color_name = color_names[i]
        h, s = color_hs[i]
        param = torch.tensor([[h]], dtype=torch.float) # [-1, 1]
        out = Cl_op.execute(img, mask=mask, specified_param=param)
        out = tensor2img(out)
        cv2.imshow('color {} img'.format(color_name), out)
        cv2.waitKey(0)
        # if not os.path.isdir('output/operator_analysis/color/{}'.format(root_color_name)):
        #     os.makedirs('output/operator_analysis/color/{}'.format(root_color_name))
        # cv2.imwrite('output/operator_analysis/color/{}/{}_h.jpg'.format(root_color_name, color_name), out)

def test_white(img, mask, cfg):
    White_op = WhiteOperator(cfg)
    for alpha in np.linspace(0, 1, num=11):
        param = torch.tensor([[alpha]], dtype=torch.float) # [-1, 1]
        out = White_op.execute(img, mask=mask, specified_param=param)
        out = tensor2img(out)
        # cv2.imshow('white {:.2f} img'.format(alpha), out)
        # cv2.waitKey(0)
        save_dir = os.path.join(cfg.operator_analysis_dir, 'color_bg')
        os.makedirs(save_dir, exist_ok=True)
        cv2.imwrite(os.path.join(save_dir, 'white {:.2f} img.jpg'.format(alpha)), out)

def test_white_balance(img, mask, cfg):
    WB_op = ImprovedWhiteBalanceOperator(cfg)
    for i in range(10):
        features = (np.random.rand(1, 3) - 0.5) * 3
        features = torch.tensor(features, dtype=torch.float) # [-1, 1]

        log_wb_range = 0.5
        channel_mask = torch.tensor([0, 1, 1], dtype=torch.float).view(1, 3).to(device)
        features = features * channel_mask
        color_scaling = torch.exp(tanh_range(-log_wb_range, log_wb_range)(features))
        # There will be no division by zero here unless the WB range lower bound is 0
        # normalize by luminance
        color_scaling = color_scaling * 1.0 / (
                1e-5 + 0.27 * color_scaling[:, 0] + 0.67 * color_scaling[:, 1] +
                0.06 * color_scaling[:, 2]).unsqueeze(1)
        out = WB_op.execute(img, mask=mask, specified_param=color_scaling)
        img_show = tensor2img(img)
        cv2.imshow('ori img', img_show)
        out = tensor2img(out)
        cv2.imshow('white balance {:.2f} img'.format(color_scaling[0, 0]), out)
        cv2.waitKey(0)

def test_inpaint(img, mask, cfg):
    Inp_op = InpaintOperator(cfg)
    out = Inp_op.execute(img, mask=mask, specified_param=0)
    out = tensor2img(out)
    # cv2.imshow('inpaint img', out)
    # cv2.waitKey(0)
    save_dir = os.path.join(cfg.operator_analysis_dir, 'inpaint_obj')
    os.makedirs(save_dir, exist_ok=True)
    cv2.imwrite(os.path.join(save_dir, 'inpaint img.jpg'.format()), out)


#################################
### reverse operation testing ###
#################################
"""reverse operation input img, parameter, return the reverse parameter
param: img: torch tensor (1, 3, h, w)
param: param: list of parameter [param1, param2, ...]
"""
rev_ops_dict = {'brightness': 'brightness', 'blur': 'sharpness', 'sharpness': 'blur', 'contrast': 'contrast'}

def get_param_naive(img, out, mask, param0, opname, cfg):
    """ standard way to estimate the parameter from img to out
    :param img: image before operation [tensor] (1, 3, h, w)
    :param out: image before operation [tensor] (1, 3, h, w)
    :param mask: mask [tensor] (1, 3, h, w]
    :param param: initial parameter that out -> img
    :param opname: operator name that out -> img
    :param cfg:
    :return: param: predicted best parameter
    :return success_flag: boolean indicating whether it is successful
    """
    SpecificOperator = eval(opname.capitalize() + 'Operator')
    Spc_op = SpecificOperator(cfg)

    def func(param):
        param = torch.tensor([param], dtype=torch.float)
        pred_out = Spc_op.execute(img, mask=mask, specified_param=param)
        err = (pred_out - out).norm()
        return err

    res = minimize(func, param0, method='Nelder-Mead')
    param = list(res.x)
    success_flag = res.success
    return param, success_flag


def apply_operator(img, mask, param, opname, cfg):
    """simplified function for applying an operation"""
    SpecificOperator = eval(opname.capitalize() + 'Operator')
    Spc_op = SpecificOperator(cfg)
    param = torch.tensor([param], dtype=torch.float)  # [-1, 1]
    out = Spc_op.execute(img, mask=mask, specified_param=param)
    return out


def get_rev_param0(param, opname):
    """get the initial parameter for reverse operator"""
    if opname in ['brightness', 'contrast']:
        func = lambda param: [-param[0]]
    elif opname in ['blur', 'sharpness']:
        func = lambda param: param
    else:
        raise NameError
    return func(param)


def get_reverse(img, out, mask, param, opname, cfg):
    """
    the reverse of forward oprator
    :param img: input iamge
    :param out: output image
    :param mask: mask
    :param param: list
    :param opname: string
    :param cfg:
    :return: rev_param: [list] param for reverse operation
    :return: rev_opname: [string] name for reverse operation
    """
    rev_opname = rev_ops_dict[opname]
    rev_param0 = get_rev_param0(param, opname)
    rev_param, success_flag = get_param_naive(out, img, mask, rev_param0, rev_opname, cfg)
    assert success_flag, 'the optimization for reverse operation failed!'
    return rev_param, rev_opname


def test_reverse(img, mask, param, opname, cfg):
    """
    the general reverse operation
    :param img: input image
    :param mask: if global, just let mask=None
    :param param: list
    :param opname: string
    :param cfg:
    :return:
    """

    out = apply_operator(img, mask, param, opname, cfg)
    rev_param, rev_opname = get_reverse(img, out, mask, param, opname, cfg)
    rev_img = apply_operator(out, mask, rev_param, rev_opname, cfg)

    # show img
    cv2.imshow('ori img', tensor2img(img))
    cv2.waitKey(0)
    cv2.imshow('out {} img'.format(opname), tensor2img(out))
    cv2.waitKey(0)
    cv2.imshow('back {} img'.format(rev_opname), tensor2img(rev_img))
    cv2.waitKey(0)

    return rev_param

