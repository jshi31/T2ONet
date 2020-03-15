import numpy as np

import torch

from core.utils.FID.inception import InceptionV3
from core.utils.FID.fid_score import get_activation, calculate_frechet_distance
from core.utils.ssim import SSIM

device = 'cuda' if torch.cuda.is_available() else 'cpu'

test_txts = ['increase the brightness', 'decrease the brightness', 'enhance the color', 'decrease the color', 'improve contrast', 'reduce contrast', 'increase saturation', 'reduce saturation', 'increase the brightness a little', 'increase the brightness a lot']

class ImageEvaluator(object):
    def __init__(self):
        self.reset()
        self.ssim = SSIM(window_size=11)
        # init FID
        block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
        self.model = InceptionV3([block_idx])
        self.model.cuda()


    def reset(self):
        self.itr = 0
        # reset L1
        self.avg_out_L1 = 0 # L1 distance of output
        self.avg_in_L1 = 0 # L1 distance of input
        # reset SSIM
        self.avg_out_SSIM = 0
        self.avg_in_SSIM = 0
        # reset FID
        self.arrs_in = []
        self.arrs_out = []
        self.arrs_gt = []


    def update(self, input, output, gt):
        """
        torch tensor
        :param input: (1, 3, h, w)
        :param output: (1, 3, h, w)
        :param gt: (1, 3, h, w)
        :return:
        """
        self.itr += 1
        self.L1(input, output, gt)
        self.SSIM(input, output, gt)
        self.FID(input, output, gt)

    def L1(self, input, output, gt):
        in_L1 = torch.abs(input - gt).mean().item()
        out_L1 = torch.abs(output - gt).mean().item()
        self.avg_in_L1 = self.avg_in_L1 * (1 - 1/self.itr) + in_L1/self.itr
        self.avg_out_L1 = self.avg_out_L1 * (1 - 1 / self.itr) + out_L1 / self.itr

    def SSIM(self, input, output, gt):
        in_SSIM = self.ssim(input, gt)
        out_SSIM = self.ssim(output, gt)
        self.avg_in_SSIM = self.avg_in_SSIM * (1 - 1/self.itr) + in_SSIM/self.itr
        self.avg_out_SSIM = self.avg_out_SSIM * (1 - 1/self.itr) + out_SSIM/self.itr

    def FID(self, input, output, gt):
        arr_in = get_activation(input, self.model)
        arr_out = get_activation(output, self.model)
        arr_gt = get_activation(gt, self.model)
        self.arrs_in.append(arr_in)
        self.arrs_out.append(arr_out)
        self.arrs_gt.append(arr_gt)

    def calc_fid(self):
        arrs_in = np.concatenate(self.arrs_in, 0)
        arrs_out = np.concatenate(self.arrs_out, 0)
        arrs_gt = np.concatenate(self.arrs_gt, 0)
        # get statistic
        mu_in = np.mean(arrs_in, axis=0)
        mu_out = np.mean(arrs_out, axis=0)
        mu_gt = np.mean(arrs_gt, axis=0)
        sigma_in = np.cov(arrs_in, rowvar=False)
        sigma_out = np.cov(arrs_out, rowvar=False)
        sigma_gt = np.cov(arrs_gt, rowvar=False)
        # get final output
        fid_in = calculate_frechet_distance(mu_in, sigma_in, mu_gt, sigma_gt)
        fid_out = calculate_frechet_distance(mu_out, sigma_out, mu_gt, sigma_gt)
        return fid_in, fid_out

    def eval(self):
        print('input L1 dist {:.4f}, output L1 dist {:.4f}'.format(self.avg_in_L1, self.avg_out_L1))
        print('input SSIM {:.4f}, output SSIM {:.4f}'.format(self.avg_in_SSIM, self.avg_out_SSIM))
        fid_in, fid_out = self.calc_fid()
        print('input FID {:.4f}, output FID {:.4f}'.format(fid_in, fid_out))


