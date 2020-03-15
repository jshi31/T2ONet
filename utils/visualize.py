import os
import pdb
import hashlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import cv2
import numpy as np

#################################
### functions for web display ###
#################################

def save_img(img, name):
    """
    save the image and name
    :param img: (3, h ,w)
    :return:
    """
    img = img.transpose((1, 2, 0)) * 255
    # RGB2BGR for cv2 saving
    img = img.astype(np.uint8)[:, :, ::-1]
    cv2.imwrite(name, img)


def name_encode(name, trunc_num=4):
    sha1 = hashlib.sha1()
    sha1.update(name.encode('utf-8'))
    res = sha1.hexdigest()
    return res[:trunc_num]


def update_web_row_s(webpage, img_x, img_y, param, iter, operators, img_dir, rewards=None, isGT=False):
    """ update single row of the data and save the image result with supervision
    :param webpage:
    :param img_x: (1, 3, h, w)
    :param img_y: (1, valid_op_len, h, w)
    :param param: (1, valid_op_len, 1)
    :param iter: [int]
    :param operators: list of predicted operators
    :param: rewards [dict]: keys: 'rewards', 'operator_rewards', 'image_rewards', values (1, valid_op_len)
    :param isGT: [bool]
    """
    param = param[0]
    if rewards is not None:
        pass
    descs = ['input']
    descs += operators
    imgs_name = []
    img_name = '{:08d}{}.jpg'.format(iter, name_encode(descs[0] + str(np.random.rand())))
    save_img(img_x[0], os.path.join(img_dir, img_name))
    imgs_name.append(img_name)
    gt_str = 'gt:' if isGT else 'pred:'
    for i in range(1, len(descs)):
        reward_str = 'r:{:.2f} img_r:{:.2f} op_r:{:.2f}'. \
            format(rewards['rewards'][0][i-1], rewards['image_rewards'][0][i-1],
                   rewards['operator_rewards'][0][i-1]) if rewards is not None else ''
        param_str = ' (' + ', '.join(['{:.2f}'.format(v) for v in param[i - 1]]) + ') ' + reward_str
        descs[i] = descs[i] + param_str
        img_name = '{:08d}{}.jpg'.format(iter, name_encode(','.join(descs[:i]) + gt_str + str(np.random.rand())))
        save_img(img_y[0][i - 1], os.path.join(img_dir, img_name))
        imgs_name.append(img_name)
    webpage.add_header(gt_str)
    webpage.add_images(imgs_name, descs, imgs_name, width=256)

def update_web_row_u(webpage, img_x, img_y, iter, img_dir, isGT=False):
    """ update single row of the data and save the image result unsupervisely
    :param webpage:
    :param img_x: (1, 3, h, w)
    :param img_y: (1, 1, 3, h, w)
    :param iter: [int]
    :param img_dir: image root directory
    :param isGT: [bool]
    """
    descs = ['input']
    descs += ['output']
    imgs_name = []
    img_name = '{:08d}{}.jpg'.format(iter, name_encode(descs[0] + str(np.random.rand())))
    save_img(img_x[0], os.path.join(img_dir, img_name))
    imgs_name.append(img_name)
    gt_str = 'gt:' if isGT else 'pred:'
    for i in range(1, len(descs)):
        img_name = '{:08d}{}.jpg'.format(iter, name_encode(','.join(descs[:i]) + gt_str + str(np.random.rand())))
        try:
            save_img(img_y[0][i - 1], os.path.join(img_dir, img_name))
        except:
            pdb.set_trace()
        imgs_name.append(img_name)
    webpage.add_header(gt_str)
    webpage.add_images(imgs_name, descs, imgs_name, width=256)


def update_web_row_sm(webpage, img_x, img_y, iter, operators, img_dir, isGT=False):
    """update single row of the data and save the image result in semi-supervisely
    :param webpage:
    :param img_x: (1, 3, h, w)
    :param img_y: (1, 1, 3, h, w)
    :param iter: [int]
    :param operators: list of predicted operators (gt_op_len)
    :param img_dir: [str]
    :param isGT: [bool]
    """
    descs = ['input']
    descs += [';'.join(operators)]
    imgs_name = []
    img_name = '{:08d}{}.jpg'.format(iter, name_encode(descs[0] + str(np.random.rand())))
    save_img(img_x[0], os.path.join(img_dir, img_name))
    imgs_name.append(img_name)
    gt_str = 'gt:' if isGT else 'pred:'
    for i in range(1, len(descs)):
        img_name = '{:08d}{}.jpg'.format(iter, name_encode(','.join(descs[:i]) + gt_str + str(np.random.rand())))
        save_img(img_y[0][i - 1], os.path.join(img_dir, img_name))
        imgs_name.append(img_name)
    webpage.add_header(gt_str)
    webpage.add_images(imgs_name, descs, imgs_name, width=256)



def update_web_row_attn(webpage, attns, cmd, ops, img_dir):
    """
    update the visualization of the attention
    :param webpage:
    :param attns: (1, op_len, sent_len), sent_len (<START> ... <END>), op_len(...<END>)
    :param cmd: list of word
    :param ops: list of word
    :param img_dir: image directory
    """
    attns = attns.cpu().numpy()[0] # (op_len, sent_len)
    op_len, cmd_len = attns.shape
    cmd = (['<START>'] + cmd + ['<END>'])[:cmd_len]
    ops = (ops + ['<END>'])[:op_len]
    save_name = name_encode(', '.join(cmd), 10) + '.png'
    save_path = os.path.join(img_dir, save_name)
    showAttention(cmd, ops, attns, save_path)
    webpage.add_images([save_name], ['attention'], [save_name], width=512)



# Visualize attention
def showAttention(input_sentence, output_words, attentions, save_path):
    """
    visualize attention
    :param input_sentence: list of token
    :param output_words: list of token
    :param attentions: (output_len, input_len) ndarray
    :return:
    """
    # Set up figure with colorbar
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(attentions, cmap='bone')
    fig.colorbar(cax)

    # Set up axes
    ax.set_xticklabels([''] + input_sentence, rotation=90)
    ax.set_yticklabels([''] + output_words)

    # Show label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.savefig(save_path)


def idx2op(x, vocab):
    """
    x symbol: no matter with or not with start symbol.
    :param x:
    :param vocab:
    :return:
    """
    x = x[0]
    start = np.where(x == 1)[0]
    start = start[0] + 1 if len(start) > 0 else 0
    end = np.where(x == 2)[0]
    end = end[0] if len(end) > 0 else len(x)
    x = x[start:end]
    ops = [vocab[i] for i in x]
    return ops

def update_web(webpage, req, y, img_x, img_y, img_pred, param_gt, param_pred, iter, symbol, vocab, img_dir, supervise, rewards=None, attns=None):
    """
    update one data
    :param: x (1, max_seq_len)
    :param: y (1, max_op_len)
    :param: img_x (1, 3, h, w) \in [0, 1]
    :param: img_y (1, gt_op_len, 3, h, w) \in [0, 1]. gt_op_len = 1 if not fully supervised, otherwise valid_op_len
    :param: img_pred (1, valid_op_len, 3, h, w) \in [0, 1]
    :param: param_gt (1, gt_op_len, 1) if fully supervised, otherwise None.
    :param: param_pred (1, valid_op_len, 1)
    :param: symbol (1, valid_op_len)
    :param: vocab [dict]
    :param: img_dir [str]
    :param: supervise [int]: 0: no gt_op, no gt_param; 1: has gt_op, no gt_param; 2: has gt_op, has gt_param
    :param: rewards [dict]: keys: 'rewards', 'operator_rewards', 'image_rewards', values (1, valid_op_len)
    """
    operators_pred = idx2op(symbol, vocab)
    webpage.add_header('iter {:5d}: {}'.format(iter, req))
    update_web_row_s(webpage, img_x, img_pred, param_pred, iter, operators_pred, img_dir, rewards=rewards,
                     isGT=False)
    if supervise == 2:
        operators = idx2op(y, vocab)
        update_web_row_s(webpage, img_x, img_y, param_gt, iter, operators, img_dir, isGT=True)
    elif supervise == 1:
        operators = idx2op(y, vocab)
        update_web_row_sm(webpage, img_x, img_y, iter, operators, img_dir, isGT=True)
    elif supervise == 0:
        update_web_row_u(webpage, img_x, img_y, iter, img_dir, isGT=True)
    if attns is not None:
        update_web_row_attn(webpage, attns, req.split(), operators_pred, img_dir)


def update_web_single(webpage, req, img_x, img_y, img_pred, iter, img_dir):
    """
    update one data
    :param: x (1, max_seq_len)
    :param: y (1, max_op_len)
    :param: img_x (1, 3, h, w) \in [0, 1]
    :param: img_y (1, gt_op_len, 3, h, w) \in [0, 1]. gt_op_len = 1 if not fully supervised, otherwise valid_op_len
    :param: img_pred (1, valid_op_len, 3, h, w) \in [0, 1]
    :param: param_gt (1, gt_op_len, 1) if fully supervised, otherwise None.
    :param: param_pred (1, valid_op_len, 1)
    :param: symbol (1, valid_op_len)
    :param: vocab [dict]
    :param: img_dir [str]
    :param: supervise [int]: 0: no gt_op, no gt_param; 1: has gt_op, no gt_param; 2: has gt_op, has gt_param
    :param: rewards [dict]: keys: 'rewards', 'operator_rewards', 'image_rewards', values (1, valid_op_len)
    """
    webpage.add_header('iter {:5d}: {}'.format(iter, req))
    update_web_row_u(webpage, img_x, img_pred, iter, img_dir, isGT=False)
    update_web_row_u(webpage, img_x, img_y, iter, img_dir, isGT=True)
