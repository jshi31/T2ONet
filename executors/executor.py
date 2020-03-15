import os
import sys
import pdb

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


from operators2 import BrightnessOperator, SharpnessOperator, ContrastOperator, InpaintOperator, WhiteOperator, SaturationOperator, ToneOperator, ColorOperator


class Executor(nn.Module):
    def __init__(self, opt):
        super(Executor, self).__init__()
        self.opt = opt
        self._register_operators(opt)
        self.name_list = [op.short_name for op in self.ops]

    def _register_operators(self, opt):
        self.brightness_op = BrightnessOperator(opt)
        self.sharpness_op = SharpnessOperator(opt)
        self.color_op = ColorOperator(opt)
        self.contrast_op = ContrastOperator(opt)
        self.inpaint_op = InpaintOperator(opt)
        self.white_op = WhiteOperator(opt)
        self.saturation_op = SaturationOperator(opt)
        self.tone_op = ToneOperator(opt)
        self.ops = [self.brightness_op, self.contrast_op, self.saturation_op, self.color_op, self.inpaint_op, self.tone_op, self.sharpness_op, self.white_op]


    def execute(self, img, op_ind, mask, features=None, specified_param=None, has_noise=False):
        """
        execute the operator with only one operator
        :param img: (bs, 3, h, w)
        :param op_ind: int
        :param features: (bs, d)
        :param mask: (bs, 1, h, w)
        :return out: output image (bs, 3, h, w)
        :return param: (bs, param_len)
        """
        # self.Op = self.ops[op_ind]
        if op_ind < 0:  # consider the case where operation index is less than 0
            bs = img.shape[0]
            return img, torch.zeros(bs, 24, dtype=torch.float).to(img.device)

        Op = self.ops[op_ind]

        # Just debug
        if specified_param is not None:
            out = Op.execute(img, mask=mask, features=None, specified_param=specified_param, has_noise=has_noise)
        else:
            out = Op.execute(img, mask=mask, features=features, has_noise=has_noise)
        return out, Op.param

    def get_param_bnd(self, op_ind):
        Op = self.ops[op_ind]
        return Op.get_param_range()

    def get_param_num(self, op_ind):
        Op = self.ops[op_ind]
        return Op.num_op_param
