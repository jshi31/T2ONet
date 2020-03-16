import os
import glob
import re
import json
import pdb
import base64
import h5py
import string
from functools import reduce

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cv2
import torch
from pycocotools.mask import decode as mask_decode
from utils.text_utils import parse_sent
from utils.visual_utils import load_train_img, load_infer_img, load_infer_img_short_size_bounded


"""
The top operators
[brightness, contrast, saturation, hue, inpaint_obj, tint, sharpness, color_bg]
"""

class GIER(object):
    """
    op_req_id: operator distinguished by request
    op_id: operator distinguished by image pair
    The following API functions are defined:
    IER          - IER api class
    getImgId     - get image id based on image name
    getReq       - get reqest based on request id
    getOp        - get operator based on op_id
    getOpReq     - get operator name based on op_req_id
    ImgId2PairId - get image pair id based on either input image id or output image id
    ReqId2PairId - get image id based on ReqId
    OpId2PairId  - get op id based on pair id
    OpReqId2ReqId- get ReqId based on OpReqId
    OpReqID2OpId - get OpId based on OpReqId

    getMask      - get mask based on pair id and operator: provide each mask index from ids, and also the total mask it will use.

    getAnnIds  - get ann ids that satisfy given filter conditions.
    getImgIds  - get image ids that satisfy given filter conditions.
    getCatIds  - get category ids that satisfy given filter conditions.
    loadRefs   - load refs with the specified ref ids.
    loadAnns   - load anns with the specified ann ids.
    loadImgs   - load images with the specified image ids.
    loadCats   - load category names with the specified category ids.
    getRefBox  - get ref's bounding box [x, y, w, h] given the ref_id
    showRef    - show image, segmentation or box of the referred object with the ref
    getMask    - get mask and area of the referred object given ref
    showMask   - show mask of the referred object given ref
    """

    def __init__(self, data_dir, vocab_dir, phase, data_mode, is_load_mask, session, train_img_size=128):
        self.op_max_len = 10
        self.req_max_len = 15
        self.session = session
        self.phase = phase
        self.data_mode = data_mode
        self.img_dir = os.path.join(data_dir, 'images')
        self.mask_dir = os.path.join(data_dir, 'masks')
        self.feature_dir = os.path.join(data_dir, 'features')
        self.split_dir = os.path.join(data_dir, 'splits')
        self.op_data = self.load_ops(phase, data_mode, session)
        self.vocab2id, self.id2vocab, self.op_vocab2id, self.id2op_vocab = self.load_vocab(vocab_dir)
        self.create_index(self.op_data)
        self.define_ids(len(self.op_data))
        self.train_img_size = train_img_size
        self.is_load_mask = is_load_mask

    def load_ops(self, phase, data_mode, session):
        with open(os.path.join(self.split_dir, '{}_sess_{}.json'.format(phase, session))) as f:
            op_data = json.load(f)

        data_modes = data_mode.split('+')
        idxs = []
        for data_mode in data_modes:
            if data_mode == 'valid':
                with open(os.path.join(self.split_dir, '{}_Ids_L1Thr_0.06_sess_{}.json'.format(phase, session))) as f:
                    idx = json.load(f)
            elif data_mode == 'shapeAlign_nonCrop':
                with open(os.path.join(self.split_dir, '{}_shapeAlignNonCrop_sess_{}.json'.format(phase, session))) as f:
                    idx = json.load(f)
            elif data_mode == 'shapeAlign':
                with open(os.path.join(self.split_dir, '{}_shapeAlign_sess_{}.json'.format(phase, session))) as f:
                    idx = json.load(f)
            elif data_mode == 'global':
                with open(os.path.join(self.split_dir, '{}_global_sess_{}.json'.format(phase, session))) as f:
                    idx = json.load(f)
            elif data_mode == 'full':
                idx = np.arange(len(op_data))
            else:
                assert False, '{} is not recognize'.format(self.data_mode)
            idxs.append(set(idx))

        idx = sorted(list(reduce(lambda x, y: x.intersection(y), idxs)))
        op_data = [op_data[i] for i in idx]
        return op_data

    def req2idx(self, sent):
        """change sentence to index"""
        def token2idx(token):
            idx = self.vocab2id[token] if token in self.vocab2id else 3
            return idx
        tokens = parse_sent(sent)
        valid_sent_idx = np.array([token2idx(token) for token in tokens])
        valid_len = len(valid_sent_idx)
        sent_idx = np.zeros(self.req_max_len, dtype=int)
        sent_idx[:min(valid_len, self.req_max_len)] = valid_sent_idx[:self.req_max_len]
        return sent_idx

    def define_ids(self, id_len):
        self.pair_ids = np.arange(id_len)
        self.req_ids = reduce(lambda x, y: x + y, [self.PairId2ReqId[i] for i in self.pair_ids])

    def filter_operator(self, op_list):
        filtered_op_list = list(filter(lambda x: x in self.op_vocab2id, list(op_list.keys())))
        return filtered_op_list

    def create_index(self, op_data):
        # go through all images
        imgs = []
        for data in op_data:
            imgs.append(data['input'])
            imgs.append(data['output'])
        imgs = np.unique(imgs)
        img_ids = [i for i in range(len(imgs))]
        getImgId = {name: id for name, id in zip(imgs, img_ids)}

        # go through all requests
        ReqId2PairId = {}
        ImgId2PairId = {}
        OpReqId2ReqId = {}
        OpReqId2OpId = {}
        OpId2PairId = {}
        getOpReq = {}
        getOp = {}
        getReq = {}
        getReqIdx = {} # ReqIdx: the idx in vocabulary
        req_id = 0
        op_req_id = 0
        op_id = 0
        for pair_i, data in enumerate(op_data):
            op_id_start = op_id
            # get data
            for op in self.filter_operator(data['operator']):
                OpId2PairId[op_id] = pair_i
                getOp[op_id] = op
                op_id += 1
            if data['expert_summary'] == [] and data['amateur_summary'] == []:
                pdb.set_trace()
            for req in data['expert_summary'] + data['amateur_summary']:
                getReq[req_id] = req
                getReqIdx[req_id] = self.req2idx(req)
                ReqId2PairId[req_id] = pair_i
                ImgId2PairId[getImgId[data['input']]] = pair_i
                ImgId2PairId[getImgId[data['output']]] = pair_i

                for op_i, op in enumerate(self.filter_operator(data['operator'])):
                    OpReqId2ReqId[op_req_id] = req_id
                    OpReqId2OpId[op_req_id] = op_id_start + op_i
                    getOpReq[op_req_id] = op
                    op_req_id += 1
                req_id += 1

        PairId2ReqId = {}
        for req_id in ReqId2PairId:
            pair_id = ReqId2PairId[req_id]
            if pair_id in PairId2ReqId:
                PairId2ReqId[pair_id].append(req_id)
            else:
                PairId2ReqId[pair_id] = [req_id]

        self.getImgId = getImgId
        self.getReq = getReq
        self.getReqIdx = getReqIdx
        self.getOpReq = getOpReq
        self.getOp = getOp
        self.ImgId2PairId = ImgId2PairId
        self.ReqId2PairId = ReqId2PairId
        self.PairId2ReqId = PairId2ReqId
        self.OpReqId2ReqId = OpReqId2ReqId
        self.OpReqId2OpId = OpReqId2OpId
        self.OpId2PairId = OpId2PairId

    def OpId2OpIdx(self, op_id):
        return self.op_vocab2id[self.getOp[op_id]]


    def get_mask(self, pair_id, operator):
        """
        get mask for certain operation
        :param pair_id: int
        :param operator: int
        :return: is_local bool
        :return: mask_id: list
        """
        mask_dict = self.op_data[pair_id]['operator'][operator]
        mask_id = mask_dict['ids']
        is_local = mask_dict['local']
        return is_local, mask_id

    def show_req_mask(self, img_name, out_name, req, masks, mask_mode):
        img = cv2.imread(os.path.join('images', img_name))[:, :, ::-1]
        out = cv2.imread(os.path.join('images', out_name))[:, :, ::-1]
        mask = np.zeros_like(img[:, :, 0])
        for mask_ in masks:
            h_mask, w_mask = mask_.shape
            h_img, w_img = mask.shape
            if h_mask != h_img or w_mask != w_img:
                print('unequal mask shape {} and image shape {}'.format((h_mask, w_mask), (h_img, w_img)))
            h = min(h_mask, h_img)
            w = min(w_mask, w_img)
            mask[:h, :w] += mask_[:h, :w]
        mask = np.clip(mask, 0, 1).astype(np.uint8)
        mask = 1 - mask if mask_mode == 'exclusive' else mask
        plt.axis("tight")
        fig = plt.figure(figsize=(25, 8), dpi=80)
        fig.suptitle(req, fontsize=12)
        ax = fig.add_subplot(131)
        ax.set_title('input image')
        ax.imshow(img)
        ax.axis("off")

        img[:, :, 0] = img[:, :, 0]*(1-mask) + 255*mask
        ax = fig.add_subplot(132)
        ax.set_title('masked part')
        ax.imshow(img)
        ax.axis("off")

        ax = fig.add_subplot(133)
        ax.set_title('edited image')
        ax.imshow(out)
        ax.axis("off")

        os.makedirs('vis', exist_ok=True)
        img_id, img_ext = img_name.split('.')
        img_suffix = base64.b64encode('{}'.format(req).encode()).decode()[:4]
        plt.savefig('./vis/{}'.format(img_id + img_suffix + '.' + img_ext))


    def load_mask_feature(self, pair_id):
        img_name = self.op_data[pair_id]['input']
        feature_name = img_name.split('.')[0] + '.h5'
        feature_file = os.path.join(self.feature_dir, feature_name)
        f = h5py.File(feature_file, 'r')
        pan_feats = f['pan_feat'][:]
        rcnn_feats = f['rcnn_feat'][:]
        pan_clss = f['cls_inds'][:]
        inst_inds = f['inst_inds'][:]
        inst_ids = f['inst_ids'][:]

        return pan_feats, rcnn_feats, pan_clss, inst_inds, inst_ids

    def load_mask(self, pair_id):
        """
        load all the candidate masks
        :param pair_id:
        :return:
        """
        img_name = self.op_data[pair_id]['input']
        # get the mask from the mask file
        mask_name = img_name.split('.')[0] + '_mask.json'
        mask_file = os.path.join(self.mask_dir, mask_name)
        with open(mask_file) as f:
            mask_data = json.load(f)
        masks = [mask_decode(mask_rle) for mask_rle in mask_data]
        return masks

    def load_vocab(self, vocab_dir):
        """load vocabulary from files under vocab_dir"""
        with open(os.path.join(vocab_dir, 'GIER_vocabs_sess_{}.json'.format(self.session))) as f:
            vocab = json.load(f)
        with open(os.path.join(vocab_dir, 'GIER_operator_vocabs_sess_{}.json'.format(self.session))) as f:
            op_vocab = json.load(f)
        vocab2id = {token: i for i, token in enumerate(vocab)}
        id2vocab = {i: token for i, token in enumerate(vocab)}
        op_vocab2id = {token: i for i, token in enumerate(op_vocab)}
        id2op_vocab = {i: token for i, token in enumerate(op_vocab)}
        return vocab2id, id2vocab, op_vocab2id, id2op_vocab


    def resize_and_union_mask(self, mask_ids, name, size):
        """
        resize and get the union of the mask
        :param mask_ids: mask_ids
        :param name: name id of each image
        :param size: (h, w)
        :return:
        """
        # load mask
        h, w = size[0], size[1]
        with open(os.path.join(self.mask_dir, '{}_{}_mask.json'.format(name, name))) as f:
            mask_rles = json.load(f)
        # resize masks
        masks = [cv2.resize(mask_decode(mask_rle), (w, h), interpolation=cv2.INTER_NEAREST)
                 for mask_rle in mask_rles]
        # select gt masks
        masks = np.array(masks, dtype=bool)[mask_ids]
        # get the union of the mask
        union_mask = masks.sum(0).astype(np.uint8)
        return union_mask

    def get_candidate_masks_with_clss(self, pair_id):
        """
        :return: masks: list of (h, w)
        :return: pan_clss: (n,)
        """
        input = self.op_data[pair_id]['input']
        # get the mask from the mask file
        mask_name = input.split('.')[0] + '_mask.json'
        mask_file = os.path.join(self.mask_dir, mask_name)
        with open(mask_file) as f:
            mask_data = json.load(f)
        masks = []
        sizes = []
        for mask_rle in mask_data:
            sizes.append(mask_rle['size'])
            mask = mask_decode(mask_rle)
            masks.append(mask)

        _, _, pan_clss, _, _ = self.load_mask_feature(pair_id)
        return masks, sizes, pan_clss


    def get_op_info(self, pair_id):
        """
        operator_idx: (max_op_len)
        is_local: (bs, masx_op_len) (1 or 0)
        mask_id: {'operator_id': [1, 2, 3]} (the direct id that can index the feature and mask)
        :param pair_id:
        :return:
        """
        op_dict = self.op_data[pair_id]['operator']
        is_local_list = []
        operator_idx = []
        mask_dict = {}
        for op in op_dict:
            if op in self.op_vocab2id:
                operator_idx.append(self.op_vocab2id[op])
                is_local, mask_ids = self.get_mask(pair_id, op)
                is_local_list.append(int(is_local))
                if is_local:
                    mask_dict[int(self.op_vocab2id[op])] = mask_ids
        operator_idx += [0] * (self.op_max_len - len(operator_idx))
        is_local_list += [0] * (self.op_max_len - len(is_local_list))
        return operator_idx, is_local_list, mask_dict


    def get_req_item(self, req_id):
        """ Get item for saving annotation
        :param req_id:
        :return: dict
        - request_idx: (max_req,)
        - operator_idx: (max_op,)
        - is_local: (max_op,) (1 or 0)
        - input: str
        - output: str
        - mask_id: {'operator_id': [1, 2, 3]} (the direct id that can index the feature and mask)
        """
        req_idx = self.getReqIdx[req_id].tolist() # (max_len,)
        req = self.getReq[req_id]
        pair_id = self.ReqId2PairId[req_id]
        input = self.op_data[pair_id]['input']
        output = self.op_data[pair_id]['output']
        input_path = os.path.join(self.img_dir, input)
        output_path = os.path.join(self.img_dir, output)
        input_img = load_train_img(input_path, self.train_img_size) if self.phase == 'train'\
            else load_infer_img_short_size_bounded(input_path)
        _, h, w = input_img.shape
        output_img = load_train_img(output_path, self.train_img_size) if self.phase == 'train' \
            else load_infer_img(output_path, (h, w))
        op_idx, is_local, mask_dict = self.get_op_info(pair_id)
        if self.is_load_mask:
            union_mask_dict = {}
            for op_key in mask_dict:
                mask_ids = mask_dict[op_key]
                mask = self.resize_and_union_mask(mask_ids, input.split('_')[0], (self.train_img_size, self.train_img_size)).astype(np.float32)
                union_mask_dict[op_key] = mask
        return_dict = {'input': input_img, 'output': output_img, 'is_local': is_local, 'op_idx': op_idx, 'request': req, 'request_idx': req_idx}
        if self.is_load_mask:
            return_dict['mask_dict'] = union_mask_dict
        return return_dict


    def get_pair_item(self, pair_id):
        """only support whole dataset looping"""
        input = self.op_data[pair_id]['input']
        output = self.op_data[pair_id]['output']
        input_path = os.path.join(self.img_dir, input)
        output_path = os.path.join(self.img_dir, output)
        input_img = load_train_img(input_path, self.train_img_size)
        output_img = load_train_img(output_path, self.train_img_size)
        op_idx, is_local, mask_dict = self.get_op_info(pair_id)
        req = self.op_data[pair_id]['expert_summary'] + self.op_data[pair_id]['amateur_summary']
        if self.is_load_mask:
            union_mask_dict = {}
            for op_key in mask_dict:
                mask_ids = mask_dict[op_key]
                mask = self.resize_and_union_mask(mask_ids, input.split('_')[0], (self.train_img_size, self.train_img_size)).astype(np.float32)
                union_mask_dict[op_key] = mask
        return_dict = {'input': input_img, 'output': output_img, 'is_local': is_local, 'op_idx': op_idx, 'request': req}
        if self.is_load_mask:
            return_dict['mask_dict'] = union_mask_dict
        return return_dict

    def __len__(self):
        return len(self.op_data)


if __name__ == '__main__':
    session = 3
    phase = 'train'
    data_mode = 'global'
    data_dir = 'data/GIER'
    vocab_dir = 'data/language'
    is_load_mask = False
    gier = GIER(data_dir, vocab_dir, phase, data_mode, is_load_mask, session)
    # ier.test()
    for i in range(len(gier)):
        gier.get_pair_item(i)
        if i > 10:
            break
    print('sucessfully load GIER')

