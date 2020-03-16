import os
import sys
sys.path.insert(0, '')
import json
import time
import pdb

import cv2
import torch
from utils.visual_utils import tensor2img
from options.fiveK_base_options import BaseOptions
from utils.beam_search import beam_search, get_dist
from executors.executor import Executor
from data.GIER.GIER import GIER


# beam search editing
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if __name__ == '__main__':

    opt = BaseOptions().parser.parse_args()

    fix_order = False
    eps_greedy = False

    # requires two files: request data, operator json
    data_dir = 'data/GIER'
    phase = 'train'
    data_mode = 'global'
    data_dir = 'data/GIER'
    vocab_dir = 'data/language'
    is_load_mask = True
    session = 3

    gier = GIER(data_dir, vocab_dir, phase, data_mode, is_load_mask, session, 256)
    set_id = 1 # set_id is the setting of the generation
    save_dir = 'output/GIER_actions_set_{}'.format(set_id)
    # configure for beam search
    beam_size = 1 if fix_order else 3
    operations = [0, 1, 2, 3, 4, 5, 6, 7]
    operation_names = ['brightness', 'contrast', 'saturation', 'color', 'inpaint', 'tone', 'sharpness', 'white']
    err = 1e-3
    dist_type = 'L1'
    optimizer = 'Nelder-Mead'
    # optimizer = 'adam'
    # optimizer = 'lbfgs'
    # executor
    executor = Executor(opt)
    avg_time = 0
    itr = 0
    for i in range(len(gier)):
        itr += 1
        data = gier.get_pair_item(i)
        tik = time.time()
        input = data['input'].unsqueeze(0)
        target = data['output'].unsqueeze(0)
        name = gier.op_data[i]['input'].split('_')[0]
        bs, _, h, w = input.shape
        mask_dict = data['mask_dict']
        req = data['request']
        input, target = list(map(lambda r: r.to(device), [input, target]))
        mask = [torch.ones(bs, 3, h, w).to(device)] + \
               [torch.from_numpy(v).to(device).expand(bs, 3, h, w) for v in mask_dict.values()] # global mask + local mask
        mask_op_idx = [-1] + [v for v in mask_dict.keys()]
        req_idx = None
        if not fix_order:
            if eps_greedy:
                act_seqs, img_seqs = beam_search_eps_greedy(input, target, req_idx, executor, None, beam_size, operations, operation_names, len(operations), err, dist_type, optimizer, replace=False)
            else:
                act_seqs, img_seqs = beam_search(input, target, req_idx, mask, mask_op_idx, executor, beam_size, operations, operation_names, len(operations), err, dist_type, optimizer, replace=False)
        else:
            act_seqs, img_seqs = beam_search_fixed_order(input, target, req_idx, executor, beam_size, operations, operation_names, len(operations), err, dist_type, optimizer, replace=False)
        tok = time.time()
        avg_time = avg_time * (1 - 1/(itr)) + (tok - tik)/(itr)

        print('{}/{}, time {:.2f}s, avg time {:.2f}s'.format(i, len(gier), tok - tik, avg_time))

        img_dir = os.path.join(save_dir, '{}'.format(name))
        os.makedirs(img_dir, exist_ok=True)
        # calc init dist
        if dist_type == 'self-disc':
            init_dist = get_disc_dist(img_x, img_y, x, discriminator).item()
        else:
            init_dist = get_dist(input, target, dist_type).item()
        # write operation episode
        info = {'request': req, 'init distance': init_dist, 'operation sequence': act_seqs}
        with open(os.path.join(img_dir, 'acts.json'), 'w') as f:
            json.dump(info, f)

        # write img
        cv2.imwrite(os.path.join(img_dir, 'input.jpg'), tensor2img(input))
        cv2.imwrite(os.path.join(img_dir, 'target.jpg'), tensor2img(target))
        if len(img_seqs) > 0:
            for idx, img in enumerate(img_seqs[0]):
                cv2.imwrite(os.path.join(img_dir, 'edit{}.jpg'.format(idx)), tensor2img(img))

