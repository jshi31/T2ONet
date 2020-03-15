import os
import json
import pdb

import numpy as np
import cv2
import torch
from core.utils.visual_utils import load_train_img, load_infer_img, load_infer_img_short_size_bounded
from torch.utils.data import Dataset, DataLoader


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




class FiveK(Dataset):
    def __init__(self, img_dir, anno_dir, vocab_dir, phase, session, train_img_size=128):
        self.op_max_len = 6
        self.req_max_len = 15
        self.session = session
        self.phase = phase
        self.img_dir = img_dir
        self.data = self.load_data(anno_dir)
        self.train_img_size = train_img_size
        self.vocab2id, self.id2vocab, self.op_vocab2id, self.id2op_vocab = load_vocab(vocab_dir, self.session)

    def load_data(self, anno_dir):
        with open(os.path.join(anno_dir, '{}_sess_{}.json'.format(self.phase, self.session))) as f:
            data = json.load(f)
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        dic = self.data[item]
        req_idx = np.array(dic['request_idx'])
        req = dic['request']
        input_path = os.path.join(self.img_dir, dic['input'])
        output_path = os.path.join(self.img_dir, dic['output'])
        input_img = load_train_img(input_path, self.train_img_size) if self.phase == 'train' else load_infer_img(input_path)
        output_img = load_train_img(output_path, self.train_img_size) if self.phase == 'train' else load_infer_img(output_path)
        return input_img, output_img, req_idx, req


def analyze_traj(seq):
    seq = np.array(seq)
    diffs = seq[:-1] - seq[1:]
    over_shot = diffs / seq[0]
    try:
        trunc_len = np.where((over_shot > 0.01) == False)[0][0]
    except:
        trunc_len = len(over_shot)
    if trunc_len == 0:
        trunc_len = 1
    return trunc_len


class FiveKAct(Dataset):
    def __init__(self, img_dir, anno_dir, act_dir, vocab_dir, phase, session, train_img_size=128):
        self.op_max_len = 5
        self.req_max_len = 15
        self.session = session
        self.train_img_size = train_img_size
        self.phase = phase
        self.img_dir = img_dir
        self.act_dir = act_dir
        self.data = self.load_data(anno_dir)
        self.vocab2id, self.id2vocab, self.op_vocab2id, self.id2op_vocab = load_vocab(vocab_dir, self.session)
        self.actions = ['brightness', 'contrast', 'saturation', 'color', 'inpaint', 'tone', 'sharpness', 'white']
        self.act2pn = {'brightness': 1, 'contrast': 1, 'saturation': 1, 'color': 24, 'inpaint': 0, 'tone': 8, 'sharpness': 1, 'white': 0}

    def load_data(self, anno_dir):
        with open(os.path.join(anno_dir, '{}_sess_{}.json'.format(self.phase, self.session))) as f:
            data = json.load(f)
        return data

    def get_act(self, item):
        item_dir = os.path.join(self.act_dir, '{}{}'.format(self.phase, item))
        json_file = os.path.join(item_dir, '{:05d}.json'.format(item))
        with open(json_file, 'r') as f:
            dict = json.load(f)
        init_dist = dict['init distance']
        seq = dict['operation sequence'][0] # choose the top sequence
        seq_dist = [v[2] for v in seq]
        # doing normalize on values for 'color' and 'tone'
        seq_dist.insert(0, init_dist)
        trunc_len = min(analyze_traj(seq_dist), self.op_max_len)
        seq = seq[:trunc_len]
        params = np.zeros((self.op_max_len, 24), dtype=np.float32)
        op_seq = np.zeros(self.op_max_len + 2, dtype=int)
        for i, act in enumerate(seq):
            op_seq[i + 1] = self.actions.index(act[0]) + 3
            param_num = self.act2pn[act[0]]
            if act[0] == 'color' or act[0] == 'tone':
                max_abs = np.abs(np.array(act[1])).max()
                params[i, :param_num] = np.array(act[1]) / max_abs
            else:
                if np.abs(act[1][0]) > 5: # if the value is too big, than just predict 0
                    params[i, :param_num] = np.array([0])
                    # print('bad' * 100)
                else:
                    params[i, :param_num] = np.array(act[1])

        op_seq[0] = 1  # start
        op_seq[i + 2] = 2  # end
        # load intermediate images
        imgs = torch.zeros(self.op_max_len, 3, self.train_img_size, self.train_img_size, dtype=torch.float32) # no need for the final image
        for i in range(trunc_len):
            imgs[i] = load_train_img(os.path.join(item_dir, 'edit{}.jpg'.format(i)), self.train_img_size)

        return op_seq, params, imgs

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        dic = self.data[item]
        req_idx = np.array(dic['request_idx'])
        req = dic['request']
        input_path = os.path.join(self.img_dir, dic['input'])
        output_path = os.path.join(self.img_dir, dic['output'])
        input_img = load_train_img(input_path, self.train_img_size) if self.phase == 'train' else load_infer_img(input_path)
        output_img = load_train_img(output_path, self.train_img_size) if self.phase == 'train' else load_infer_img(output_path)
        ops, params, imgs = self.get_act(item)
        output_imgs = torch.cat([imgs, output_img.unsqueeze(0)])  # the last img is gt
        return input_img, output_imgs, req_idx, ops, params, req


class FiveKActVisualize(Dataset):
    def __init__(self, img_dir, anno_dir, act_dir, vocab_dir, phase, session):
        self.op_max_len = 5
        self.req_max_len = 15
        self.session = session
        self.phase = phase
        self.img_dir = img_dir
        self.act_dir = act_dir
        self.data = self.load_data(anno_dir)
        self.vocab2id, self.id2vocab, self.op_vocab2id, self.id2op_vocab = load_vocab(vocab_dir, self.session)
        self.actions = ['brightness', 'contrast', 'saturation', 'color', 'inpaint', 'tone', 'sharpness', 'white']
        self.act2pn = {'brightness': 1, 'contrast': 1, 'saturation': 1, 'color': 24, 'inpaint': 0, 'tone': 8, 'sharpness': 1, 'white': 0}

    def load_data(self, anno_dir):
        with open(os.path.join(anno_dir, '{}_sess_{}.json'.format(self.phase, self.session))) as f:
            data = json.load(f)
        return data

    def get_act(self, item):
        item_dir = os.path.join(self.act_dir, '{}{}'.format(self.phase, item))
        json_file = os.path.join(item_dir, '{:05d}.json'.format(item))
        with open(json_file, 'r') as f:
            dict = json.load(f)
        init_dist = dict['init distance']
        seq = dict['operation sequence'][0] # choose the top sequence
        seq_dist = [v[2] for v in seq]
        # doing normalize on values for 'color' and 'tone'
        seq_dist.insert(0, init_dist)
        # trunc_len = min(analyze_traj(seq_dist), self.op_max_len)
        seq = seq[:self.op_max_len]
        params = np.zeros((self.op_max_len, 24), dtype=np.float32)
        op_seq = np.zeros(self.op_max_len + 2, dtype=int)
        for i, act in enumerate(seq):
            op_seq[i + 1] = self.actions.index(act[0]) + 3
            param_num = self.act2pn[act[0]]
            if act[0] == 'color' or act[0] == 'tone':
                max_abs = np.abs(np.array(act[1])).max()
                params[i, :param_num] = np.array(act[1]) / max_abs
            else:
                if np.abs(act[1][0]) > 5: # if the value is too big, than just predict 0
                    params[i, :param_num] = np.array([0])
                    # print('bad' * 100)
                else:
                    params[i, :param_num] = np.array(act[1])

        op_seq[0] = 1  # start
        op_seq[i + 2] = 2  # end
        # load intermediate images
        return op_seq, params

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        dic = self.data[item]
        req_idx = np.array(dic['request_idx'])
        req = dic['request']
        input_path = os.path.join(self.img_dir, dic['input'])
        output_path = os.path.join(self.img_dir, dic['output'])
        input_img = load_infer_img_short_size_bounded(input_path, 600)
        output_img = load_infer_img_short_size_bounded(output_path, 600)
        ops, params = self.get_act(item)
        return input_img, output_img, req_idx, ops, params, req


class FiveKActDVisualize(Dataset):
    def __init__(self, img_dir, anno_dir, act_dir, vocab_dir, phase, session):
        self.op_max_len = 5
        self.req_max_len = 15
        self.session = session
        self.phase = phase
        self.img_dir = img_dir
        self.act_dir = act_dir
        self.data = self.load_data(anno_dir)
        self.vocab2id, self.id2vocab, self.op_vocab2id, self.id2op_vocab = load_vocab(vocab_dir, self.session)
        self.actions = ['brightness', 'contrast', 'saturation', 'color', 'inpaint', 'tone', 'sharpness', 'white']
        self.act2pn = {'brightness': 1, 'contrast': 1, 'saturation': 1, 'color': 24, 'inpaint': 0, 'tone': 8, 'sharpness': 1, 'white': 0}

    def load_data(self, anno_dir):
        with open(os.path.join(anno_dir, '{}_sess_{}.json'.format(self.phase, self.session))) as f:
            data = json.load(f)
        return data

    def get_act(self, item):
        item_dir = os.path.join(self.act_dir, '{}{}'.format(self.phase, item))
        json_file = os.path.join(item_dir, 'seq2seqGAN-disc_ops.json'.format(item))
        with open(json_file, 'r') as f:
            dict = json.load(f)
        init_dist = dict['init distance']
        seq = dict['operation sequence'][0] # choose the top sequence
        seq_dist = [v[2] for v in seq]
        # doing normalize on values for 'color' and 'tone'
        seq_dist.insert(0, init_dist)
        # trunc_len = min(analyze_traj(seq_dist), self.op_max_len)
        seq = seq[:self.op_max_len]
        params = np.zeros((self.op_max_len, 24), dtype=np.float32)
        op_seq = np.zeros(self.op_max_len + 2, dtype=int)
        for i, act in enumerate(seq):
            op_seq[i + 1] = self.actions.index(act[0]) + 3
            param_num = self.act2pn[act[0]]
            if act[0] == 'color' or act[0] == 'tone':
                max_abs = np.abs(np.array(act[1])).max()
                params[i, :param_num] = np.array(act[1]) / max_abs
            else:
                if np.abs(act[1][0]) > 5: # if the value is too big, than just predict 0
                    params[i, :param_num] = np.array([0])
                    # print('bad' * 100)
                else:
                    params[i, :param_num] = np.array(act[1])

        op_seq[0] = 1  # start
        op_seq[i + 2] = 2  # end
        # load intermediate images
        return op_seq, params

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        dic = self.data[item]
        req_idx = np.array(dic['request_idx'])
        req = dic['request']
        input_path = os.path.join(self.img_dir, dic['input'])
        output_path = os.path.join(self.img_dir, dic['output'])
        input_img = load_infer_img_short_size_bounded(input_path, 600)
        output_img = load_infer_img_short_size_bounded(output_path, 600)
        ops, params = self.get_act(item)
        return input_img, output_img, req_idx, ops, params, req


if __name__ == '__main__':
    img_dir = 'data/FiveK/images'
    anno_dir = 'data/FiveK/annotations'
    vocab_dir = 'data/language'
    act_dir = 'output/actions_set_1'
    phase = 'train'
    session = 1
    # test FiveK
    # dataset = FiveK(img_dir, anno_dir, vocab_dir, phase, session)
    #
    # for i, data in enumerate(dataset):
    #     print('{}/{}'.format(i, len(dataset)))
    #     input, output, req_idx, req = data
    #     pdb.set_trace()

    # test FiveKAct
    dataset = FiveKAct(img_dir, anno_dir, act_dir, vocab_dir, phase, session)
    loader = DataLoader(dataset, batch_size=1)

    for i, data in enumerate(loader):
        print('{}/{}'.format(i, len(loader)))
        try:
            input, outputs, req_idx, ops, params, req = data
        except:
            pdb.set_trace()

