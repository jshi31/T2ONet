import cv2
import pdb
import numpy as np
import torch

def load_train_img(img_path, img_size):
    img = cv2.imread(img_path)
    # resize
    img = cv2.resize(img, (img_size, img_size))
    # bgr->rgb
    img = img[:, :, ::-1].astype(np.float32)
    # to tensor
    img = torch.from_numpy(img.transpose(2, 0, 1))/255
    return img


def load_infer_img(img_path, img_size=None):
    """
    :param img_path: img path
    :param img_size: None or tuple (h, w)
    :return: img
    """
    img = cv2.imread(img_path)
    # resize
    if img_size is not None:
        img = cv2.resize(img, (img_size[1], img_size[0]))
    # bgr->rgb
    img = img[:, :, ::-1].astype(np.float32)
    # to tensor
    img = torch.from_numpy(img.transpose(2, 0, 1))/255
    return img

def load_infer_img_short_size_bounded(img_path, short_size=600):
    img = cv2.imread(img_path)
    h, w, _ = img.shape
    min_size = min(h, w)
    ratio = short_size / min_size
    h_new = int(np.round(h*ratio))
    w_new = int(np.round(w*ratio))
    # resize
    img = cv2.resize(img, (w_new, h_new))
    # bgr->rgb
    img = img[:, :, ::-1].astype(np.float32)
    # to tensor
    img = torch.from_numpy(img.transpose(2, 0, 1))/255
    return img

