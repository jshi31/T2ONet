import os
import argparse
import pdb
import numpy as np
import torch


class BaseOptions():
    """Base option class"""

    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument('--dataset', default='FiveK', type=str, help='select dataset')
        self.parser.add_argument('--run_dir', type=str, help='experiment directory')
        self.parser.add_argument('--data_mode', type=str, help='data mode for loading GIER, shapeAlign, valid, shapeAlign_nonCrop, global, full', default='shapeAlign')
        # Dataloader
        self.parser.add_argument('--shuffle', default=0, type=int, help='shuffle dataset')
        self.parser.add_argument('--num_workers', default=1, type=int, help='number of workers for loading data')

        # Run
        self.parser.add_argument('--manual_seed', default=10, type=int, help='manual seed')
        self.parser.add_argument('--gpu_ids', default='0', type=str, help='ids of gpu to be used')
        self.parser.add_argument('--visualize', default=0, type=int, help='visualize experiment')
        self.parser.add_argument('--trial', default=1, type=int, help='trial index')
        self.parser.add_argument('--session', default=1, type=int, help='to match different dataset setting')
        # Dataset catalog
        self.parser.add_argument('--vocab_dir', default='data/language',
                                 type=str, help='path to command vocab')

        # Model
        self.parser.add_argument('--encoder_max_len', default=17, type=int, help='max length of input sequence')
        self.parser.add_argument('--decoder_max_len', default=5, type=int, help='original 12 max length of output sequence')
        self.parser.add_argument('--hidden_size', default=256, type=int, help='hidden layer dimension')
        self.parser.add_argument('--word_vec_dim', default=300, type=int, help='dimension of word embedding vector')

        self.parser.add_argument('--use_attention', default=1, type=int, help='use attention in decoder')
        self.parser.add_argument('--use_vis_feat', default=1, type=int, help='use visual feature in decoder')
        self.parser.add_argument('--bidirectional', default=1, type=int, help='bidirectional encoder')
        self.parser.add_argument('--rnn_cell', default='lstm', type=str, help='encoder rnn cell type, options: lstm, gru')
        self.parser.add_argument('--n_layers', default=2, type=int, help='number of hidden layers')

        self.parser.add_argument('--use_vgg', default=1, type=int, help='fix the step length')

        self.parser.add_argument('--fusing_method', type=str, default='', help='fusing_method')
        # Executor
        self.parser.add_argument('--discrete_param', default=0, type=int, help='if the parameter for each operator should be discrete')
        self.parser.add_argument('--discrete_step', default=10, type=int, help='the number of parameter step that we want to discrete for each parameter')
        self.parser.add_argument('--vis_feat_dim', default=1024, type=int, help='visual feature dimension')
        self.parser.add_argument('--operator_fc_dim', default=512, type=int, help='fc1 layer dimension')
        self.parser.add_argument('--fix_step', default=1, type=int, help='fix the step length')


        # Operator
        self.parser.add_argument('--exposure_range', type=float, default=3.5, help='exposure_range')
        self.parser.add_argument('--sharpness_range', type=float, default=1.5, help='exposure_range')
        self.parser.add_argument('--brightness_range', type=float, default=2, help='brightness range')
        self.parser.add_argument('--curve_steps', type=int, default=8, help='the discrete steps of the curve')
        self.parser.add_argument('--tone_curve_range', type=tuple, default=(0.5, 2), help='tone curve range')
        self.parser.add_argument('--color_curve_range', type=tuple, default=(0.90, 1.10), help='color curve range')
        self.parser.add_argument('--saturation_range', type=tuple, default=(-0.2, 0.8), help='saturation range')


    def parse(self):
        # Instantiate option
        self.opt = self.parser.parse_args()

        # Parse gpu id list
        str_gpu_ids = self.opt.gpu_ids.split(',')
        self.opt.gpu_ids = []
        for str_id in str_gpu_ids:
            if str_id.isdigit() and int(str_id) >= 0:
                self.opt.gpu_ids.append(int(str_id))
        if len(self.opt.gpu_ids) > 0 and torch.cuda.is_available():
            torch.cuda.set_device(self.opt.gpu_ids[0])
        else:
            print('| using cpu')
            self.opt.gpu_ids = []

        # Set manual seed
        if self.opt.manual_seed is not None:
            torch.manual_seed(self.opt.manual_seed)
            if len(self.opt.gpu_ids) > 0 and torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.opt.manual_seed)
            np.random.seed(self.opt.manual_seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        # Print and save options
        args = vars(self.opt)
        print('| options')
        for k, v in args.items():
            print('%s: %s' % (str(k), str(v)))
        self.opt.run_dir = self.opt.run_dir if self.opt.run_dir is not None else \
            'output/{}_trial_{}'.format(self.opt.dataset, self.opt.trial)
        for phase in ['train', 'test']:
            os.makedirs(os.path.join(self.opt.run_dir, phase), exist_ok=True)
        if self.is_train:
            file_path = os.path.join(self.opt.run_dir, 'train', 'train_opt.txt')
        else:
            file_path = os.path.join(self.opt.run_dir, 'test', 'test_opt.txt')
        with open(file_path, 'wt') as fout:
            fout.write('| options\n')
            for k, v in args.items():
                fout.write('%s: %s\n' % (str(k), str(v)))

        return self.opt
