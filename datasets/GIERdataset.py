import os
import sys

sys.path.append('')
from core.utils.visual_utils import load_train_img

import json
import pdb
import copy

import numpy as np
import cv2
import torch
from torch.utils.data import Dataset, DataLoader


from data.GIER.GIER import GIER



class GIERDataset(Dataset):
    def __init__(self, data_dir, vocab_dir, phase, data_mode, is_load_mask, session, train_img_size=128):
        self.op_max_len = 8
        self.req_max_len = 15
        self.session = session
        self.phase = phase
        self.train_img_size = train_img_size
        self.GIER = GIER(data_dir, vocab_dir, phase, data_mode, is_load_mask, session, train_img_size)
        self.vocab2id, self.id2vocab, self.op_vocab2id, self.id2op_vocab = \
            self.GIER.vocab2id, self.GIER.id2vocab, self.GIER.op_vocab2id, self.GIER.id2op_vocab

    def pad_req(self, req_idx):
        end = np.where(np.array(req_idx) == 0)[0]
        if len(end) > 0:
            req_idx.insert(end[0], 2)
        else:
            req_idx.append(2)
        req_idx.insert(0, 1)
        return req_idx

    def pad_op(self, op_idx):
        end = np.where(np.array(op_idx) == 0)[0]
        if len(end) > 0:
            op_idx.insert(end[0], 2)
        else:
            op_idx.append(2)
        op_idx.insert(0, 1)
        return op_idx

    def collate(self, batch):
        """
        :param batch:
        data, output_blob, req_idx, op_idx, req, union_mask_dict
        :return:
        - data: (B, 3, H, W)
        - im_info: (B, 3)
        - gt: (B, 3, H', W')
        - request_idx: (B, max_len)
        - op_idx: (B, max_len)
        - request: list of request string
        - mask_dict: list of mask_dict, each dict is {'op_id', mask (1, 1, H, W)}
        """
        blob = {}
        for key in batch[0]:
            if type(batch[0][key]) == dict or type(batch[0][key]) == list or type(batch[0][key]) == str:
                blob[key] = [b[key] for b in batch]
            elif type(batch[0][key]) == torch.Tensor:
                blob[key] = torch.stack([b[key] for b in batch])
        return blob

    def __len__(self):
        return len(self.GIER.ReqId2PairId)

    def __getitem__(self, item):
        dic = copy.deepcopy(self.GIER.get_req_item(item))
        dic['request_idx'] = self.pad_req(dic['request_idx'])
        dic['request_idx'] = torch.tensor(dic['request_idx'])
        return dic


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


class GIERDatasetAct(Dataset):
    def __init__(self, data_dir, vocab_dir, act_dir, phase, data_mode, is_load_mask, session, train_img_size=128):
        self.op_max_len = 8
        self.req_max_len = 15
        self.session = session
        self.phase = phase
        self.train_img_size = train_img_size
        self.GIER = GIER(data_dir, vocab_dir, phase, data_mode, is_load_mask, session, train_img_size)
        self.vocab2id, self.id2vocab, self.op_vocab2id, self.id2op_vocab = \
            self.GIER.vocab2id, self.GIER.id2vocab, self.GIER.op_vocab2id, self.GIER.id2op_vocab
        self.act_dir = act_dir
        self.actions = ['brightness', 'contrast', 'saturation', 'color', 'inpaint', 'tone', 'sharpness', 'white']
        self.act2pn = {'brightness': 1, 'contrast': 1, 'saturation': 1, 'color': 24, 'inpaint': 0, 'tone': 8, 'sharpness': 1, 'white': 0}

    def pad_req(self, req_idx):
        end = np.where(np.array(req_idx) == 0)[0]
        if len(end) > 0:
            req_idx.insert(end[0], 2)
        else:
            req_idx.append(2)
        req_idx.insert(0, 1)
        return req_idx

    def pad_op(self, op_idx):
        end = np.where(np.array(op_idx) == 0)[0]
        if len(end) > 0:
            op_idx.insert(end[0], 2)
        else:
            op_idx.append(2)
        op_idx.insert(0, 1)
        return op_idx

    def collate(self, batch):
        """
        :param batch:
        data, output_blob, req_idx, op_idx, req, union_mask_dict
        :return:
        - data: (B, 3, H, W)
        - im_info: (B, 3)
        - gt: (B, 3, H', W')
        - request_idx: (B, max_len)
        - op_idx: (B, max_len)
        - request: list of request string
        - mask_dict: list of mask_dict, each dict is {'op_id', mask (1, 1, H, W)}
        """
        blob = {}
        for key in batch[0]:
            if type(batch[0][key]) == dict or type(batch[0][key]) == list or type(batch[0][key]) == str:
                blob[key] = [b[key] for b in batch]
            elif type(batch[0][key]) == torch.Tensor:
                blob[key] = torch.stack([b[key] for b in batch])
            elif type(batch[0][key]) == np.ndarray:
                blob[key] = torch.stack([torch.from_numpy(b[key]) for b in batch])
            else:
                assert False, '{} cannot be collated'.format(type(batch[0][key]))
        return blob


    def get_act(self, item):
        pair_id = self.GIER.ReqId2PairId[item]
        data_id = self.GIER.op_data[pair_id]['input'].split('_')[0]
        item_dir = os.path.join(self.act_dir, '{}'.format(data_id))
        json_file = os.path.join(item_dir, 'acts.json')
        with open(json_file, 'r') as f:
            dict = json.load(f)
        init_dist = dict['init distance']
        seq = dict['operation sequence'][0] # choose the top sequence
        seq_dist = [v[-1] for v in seq]
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
        return len(self.GIER.ReqId2PairId)

    def __getitem__(self, item):
        dic = copy.deepcopy(self.GIER.get_req_item(item))
        dic['request_idx'] = self.pad_req(dic['request_idx'])
        dic['request_idx'] = torch.tensor(dic['request_idx'])

        ops, params, imgs = self.get_act(item)
        output_imgs = torch.cat([imgs, dic['output'].unsqueeze(0)]) # the last img is gt
        dic['output'] = output_imgs
        dic['operations'] = ops
        dic['parameters'] = params
        return dic



if __name__ == '__main__':
    session = 3
    act_id = 1
    phase = 'val'
    data_mode = 'shapeAlign'
    data_dir = 'data/GIER'
    vocab_dir = 'data/language'
    act_dir = 'output/GIER_actions_set_{}'.format(act_id)
    is_load_mask = False

    dataset = GIERDataset(data_dir, vocab_dir, phase, data_mode, is_load_mask, session)
    # dataset = GIERDatasetAct(data_dir, vocab_dir, act_dir, phase, data_mode, is_load_mask, session)

    dataloader = DataLoader(dataset, collate_fn=dataset.collate)
    print(len(dataloader))
    for data in dataloader:
        pdb.set_trace()


