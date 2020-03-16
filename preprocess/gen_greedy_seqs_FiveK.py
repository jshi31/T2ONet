import os
import sys
sys.path.insert(0, '')
import json
import time
import pdb

import cv2
import torch
from options.fiveK_base_options import BaseOptions
from utils.visual_utils import tensor2img
from utils.beam_search import beam_search, get_dist
from utils.beam_search_fixed_order import beam_search as beam_search_fixed_order
from utils.beam_search_eps_greedy import beam_search as beam_search_eps_greedy
from datasets.FiveKdataset import FiveK
from executors.executor import Executor



if __name__ == '__main__':

    opt = BaseOptions().parser.parse_args()

    fix_order = False
    eps_greedy = False

    img_dir = 'data/FiveK/images'
    anno_dir = 'data/FiveK/annotations'
    vocab_dir = 'data/language'
    phases = ['train']
    for phase in phases:
        session = 1
        set_id = 11 # set_id is the setting of the generation  # TODO change set id
        save_dir = 'output/actions_set_{}'.format(set_id)
        dataset = FiveK(img_dir, anno_dir, vocab_dir, phase, session)
        loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)
        # configure for beam search
        beam_size = 1 if fix_order else 3
        operations = [6]  # TODO change operation lists
        operation_names = ['brightness', 'contrast', 'saturation', 'color', 'inpaint', 'tone', 'sharpness', 'white']
        err = 1e-2
        dist_type = 'L1'
        optimizer = 'Nelder-Mead'
        # optimizer = 'adam'
        # optimizer = 'lbfgs'
        # executor
        executor = Executor(opt)
        avg_time = 0
        itr = 0
        for i, data in enumerate(loader):
            itr += 1
            tik = time.time()
            input, target, req_idx, req = data
            if not fix_order:
                if eps_greedy:
                    act_seqs, img_seqs = beam_search_eps_greedy(input, target, req_idx, executor, None, beam_size, operations, operation_names, len(operations), err, dist_type, optimizer, replace=False)
                else:
                    act_seqs, img_seqs = beam_search(input, target, req_idx, executor, None, beam_size, operations, operation_names, len(operations), err, dist_type, optimizer, replace=False)
            else:
                act_seqs, img_seqs = beam_search_fixed_order(input, target, req_idx, executor, beam_size, operations, operation_names, len(operations), err, dist_type, optimizer, replace=False)
            tok = time.time()
            avg_time = avg_time * (1 - 1/(itr)) + (tok - tik)/(itr)

            print('{}/{}, time {:.2f}s, avg time {:.2f}s'.format(i, len(loader), tok - tik, avg_time))

            img_dir = os.path.join(save_dir, '{}{}'.format(phase, i))
            os.makedirs(img_dir, exist_ok=True)
            # calc init dist
            if dist_type == 'self-disc':
                init_dist = get_disc_dist(img_x, img_y, x, discriminator).item()
            else:
                init_dist = get_dist(input, target, dist_type).item()
            # write operation episode
            info = {'request': req, 'init distance': init_dist, 'operation sequence': act_seqs}
            with open(os.path.join(img_dir, '{:05d}.json'.format(i)), 'w') as f:
                json.dump(info, f)

            # write img
            cv2.imwrite(os.path.join(img_dir, 'input.jpg'), tensor2img(input))
            cv2.imwrite(os.path.join(img_dir, 'target.jpg'), tensor2img(target))
            if len(img_seqs) > 0:
                for idx, img in enumerate(img_seqs[0]):
                    cv2.imwrite(os.path.join(img_dir, 'edit{}.jpg'.format(idx)), tensor2img(img))

