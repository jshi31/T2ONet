# -*- coding: utf-8 -*-
# @Time  : 6/15/20 12:40 PM
# @Author: Jing Shi
# @Email : j.shi@rochester.edu

import os
import sys
import shutil
sys.path.append('')
import time
import string
import pdb
import json

import numpy as np
import cv2
import torch
from options.fiveK_train_options import TrainOptions
from models.actor import Actor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def tensor2img(tensor):
    """ transform tensor to BGR image
    :param tensor:
    :return: BGR image
    """
    out = tensor.squeeze(0).permute(1, 2, 0) * 255
    # RGB2BGR for cv2 saving
    out = out.cpu().numpy().astype(np.uint8)[:, :, ::-1]
    return out


def load_img(img_path):
    img = cv2.imread(img_path)
    # bgr->rgb
    img = img[:, :, ::-1].astype(np.float32)
    # to tensor
    img = torch.from_numpy(img.transpose(2, 0, 1))/255
    img = img.unsqueeze(0)
    return img


def parse_sent(desc):
    table = str.maketrans('', '', string.punctuation)
    # tokenize
    desc = desc.split()
    # convert to lower case
    desc = [word.lower() for word in desc]
    # remove punctuation from each token
    desc = [w.translate(table) for w in desc]
    # remove hanging 's' and 'a'
    desc = [word for word in desc if len(word) > 1]
    # remove tokens with numbers in them
    tokens = [word for word in desc if word.isalpha()]
    return tokens


def load_vocab(vocab_dir, session):
    """load vocabulary from files under vocab_dir"""
    with open(os.path.join(vocab_dir, 'FiveK_vocabs_sess_{}.json'.format(session))) as f:
        vocab = json.load(f)
    with open(os.path.join(vocab_dir, 'FiveK_operator_vocabs_sess_{}.json'.format(session))) as f:
        op_vocab = json.load(f)
    vocab2id = {token: i for i, token in enumerate(vocab)}
    id2vocab = {i: token for i, token in enumerate(vocab)}
    op_vocab2id = {token: i for i, token in enumerate(op_vocab)}
    id2op_vocab = {i: token for i, token in enumerate(op_vocab)}
    return vocab2id, id2vocab, op_vocab2id, id2op_vocab


def get_data(img_path, sent, opt):
    """
    :param img_path: string
    :param sent: string
    :return:
    """
    vocab2id, _, _, _ = load_vocab(opt.vocab_dir, opt.session)
    max_len = 15

    def token2idx(token):
        idx = vocab2id[token] if token in vocab2id else 3
        return idx

    img = load_img(img_path)
    tokens = parse_sent(sent)
    valid_sent_idx = np.array([token2idx(token) for token in tokens])
    valid_len = len(valid_sent_idx)
    sent_idx = np.zeros(max_len, dtype=int)
    sent_idx[:min(valid_len, max_len)] = valid_sent_idx[:max_len]
    end = np.where(sent_idx == 0)[0]
    sent_idx = sent_idx.tolist()
    if len(end) > 0:
        sent_idx.insert(end[0], 2)
    else:
        sent_idx.append(2)
    sent_idx.insert(0, 1)
    sent_idx = torch.tensor(sent_idx).unsqueeze(0)
    return img, sent_idx


def test(model, img, txt):
    """train model"""
    model.eval()
    tik = time.time()
    # ship to device
    txt, img = list(map(lambda r: r.to(device), [txt, img]))
    with torch.no_grad():
        state, pred_imgs, pred_ops, pred_params = model.episode_forward(txt, img, mask_dict=None,
                                                                        reinforce_sample=False)
    tok = time.time()
    print('time {:.2f}'.format(tok - tik))
    return pred_imgs, pred_ops, pred_params


if __name__ == '__main__':
    # options
    opt = TrainOptions().parse()

    operation_names = ['brightness', 'contrast', 'saturation', 'color', 'inpaint', 'tone', 'sharpness', 'white']
    operation_param_num = {'brightness': 1, 'contrast': 1, 'saturation': 1, 'color': 24, 'inpaint': 0, 'tone': 8, 'sharpness': 1, 'white': 0}
    save_dir = 'output/FiveK_trial_1/demo_output'
    os.makedirs(save_dir, exist_ok=True)
    # load model
    model = Actor(opt)
    ckpt_dir = os.path.join(opt.run_dir, 'seq2seqL1_model')
    model_dir = os.path.join(ckpt_dir, 'checkpoint_best')
    model.load_state_dict(torch.load(os.path.join(model_dir, 'model.pth')), strict=False)
    print('loaded model from {}'.format(model_dir))
    model.cuda()

    # load and preprocessing img and text
    txts = [opt.request]
    img_path = [opt.img] if opt.img else ['data/FiveK/images/1480_O.jpg']
    img_paths = img_path * len(txts)

    infos = []

    save_multi_imgs = True

    for i in range(len(img_paths)):
        info_dict = {}
        img, sent = get_data(img_paths[i], txts[i], opt)
        # test
        pred_imgs, pred_ops, pred_params = test(model, img, sent)
        pred_ops = pred_ops.cpu().numpy()[0] - 3

        ops_len = len(pred_ops)
        for op_pos, op_id in enumerate(pred_ops):
            if op_id == -1:
                ops_len = op_pos

        pred_op_names = [operation_names[idx] for idx in pred_ops[:ops_len]]
        pred_params = [pred_param.cpu()[0].numpy().tolist()[:operation_param_num[pred_op_names[param_i]]]
                       for param_i, pred_param in enumerate(pred_params[:ops_len])]
        pred_op_info = [(op_name, param) for op_name, param in zip(pred_op_names, pred_params)]

        # save input image

        name = img_paths[i].split('/')[-1].split('.')[0]
        save_img_dir = os.path.join(save_dir, name)
        print('save img dir', save_img_dir)
        if os.path.exists(save_img_dir):
            shutil.rmtree(save_img_dir)
            if not os.path.exists(save_img_dir):
                print('delete previous dir')
        os.makedirs(save_img_dir, exist_ok=True)

        input_name = name+'_in.jpg'
        cv2.imwrite(os.path.join(save_img_dir, input_name), tensor2img(img))
        # if save multiple images
        if opt.multi_img:
            for img_i in range(ops_len):
                # write imgs
                output_name = '{}_inference_{}.jpg'.format(img_i+1, name)
                cv2.imwrite(os.path.join(save_img_dir, output_name), tensor2img(pred_imgs[:, img_i]))
                print('saved data in {}'.format(os.path.join(save_img_dir, output_name)))


        # if save final image
        else:
            pred_img = pred_imgs[:, -1]
            # write img
            name = img_paths[i].split('/')[-1]
            input_name = name+'_in.jpg'
            output_name = name+'.jpg'
            cv2.imwrite(os.path.join(save_dir, input_name), tensor2img(img))
            cv2.imwrite(os.path.join(save_dir, output_name), tensor2img(pred_img))
            print('saved data in {}'.format(os.path.join(save_dir, output_name)))


        info_dict['input'] = input_name
        info_dict['request'] = txts[i]
        info_dict['output'] = output_name
        info_dict['operations'] = pred_op_info
        infos.append(info_dict)

    with open(os.path.join(save_img_dir, name+'.json'), 'w') as f:
        json.dump(infos, f)
