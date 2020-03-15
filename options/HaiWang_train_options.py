from .HaiWang_base_options import BaseOptions


class TrainOptions(BaseOptions):
    """Train option class"""

    def __init__(self):
        super(TrainOptions, self).__init__()

        ### pix2pix
        # for training
        self.parser.add_argument('--load_pretrain', type=str, default='', help='load the pretrained model from the specified location')
        self.parser.add_argument('--which_epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
        self.parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')
        self.parser.add_argument('--niter', type=int, default=100, help='# of iter at starting learning rate')
        self.parser.add_argument('--niter_decay', type=int, default=100, help='# of iter to linearly decay learning rate to zero')
        self.parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')
        self.parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate for gan adam')

        # for GAN
        self.parser.add_argument('--lambda_A', type=float, default=10.0, help='weight for L1')
        self.parser.add_argument('--lambda_feat', type=float, default=10.0, help='weight for feature matching loss')
        self.parser.add_argument('--no_ganFeat_loss', action='store_true', help='if specified, do *not* use discriminator feature matching loss')
        self.parser.add_argument('--no_vgg_loss', action='store_true', help='if specified, do *not* use VGG feature matching loss')
        self.parser.add_argument('--no_lsgan', action='store_true', help='do *not* use least square GAN, if false, use vanilla GAN')
        self.parser.add_argument('--pool_size', type=int, default=0, help='the size of image buffer that stores previously generated images')

        self.isTrain = True

        # Data
        self.parser.add_argument('--max_train_samples', default=None, type=int, help='max number of training samples')
        self.parser.add_argument('--max_val_samples', default=10000, type=int, help='max number of val samples')
        self.parser.add_argument('--input_dropout_p', default=0.2, type=float, help='dropout probability for input sequence')
        self.parser.add_argument('--dropout_p', default=0.2, type=float, help='dropout probability for output sequence')
        self.parser.add_argument('--variable_lengths', default=1, type=int, help='variable input length')
        self.parser.add_argument('--use_input_embedding', default=0, type=int, help='use pretrained word embedding for input sentences')
        self.parser.add_argument('--fix_input_embedding', default=1, type=int, help='fix word embedding for input sentences')
        self.parser.add_argument('--start_id', default=1, type=int, help='id for start token')
        self.parser.add_argument('--end_id', default=2, type=int, help='id for end token')
        self.parser.add_argument('--null_id', default=0, type=int, help='id for null token')
        self.parser.add_argument('--lam1', default=1, type=float, help='lambda for operator loss')
        self.parser.add_argument('--lam2', default=5, type=float, help='lambda for parameter loss')
        self.parser.add_argument('--op_reward_lam', default=0, type=float, help='operation reward coefficient')
        self.parser.add_argument('--img_reward_lam', default=1, type=float, help='img reward coefficient')

        # Model
        self.parser.add_argument('--load_checkpoint_path', default=None, type=str, help='checkpoint path')
        # Training
        self.parser.add_argument('--batch_size', default=1, type=int, help='batch size')
        self.parser.add_argument('--iter_size', default=16, type=int, help='update loss every iter_size')
        self.parser.add_argument('--num_iters', default=200000, type=int, help='total number of iterations')
        self.parser.add_argument('--print_every', default=100, type=int, help='display every')
        self.parser.add_argument('--visualize_every', default=100, type=int, help='visualize training with tensorboard')
        self.parser.add_argument('--checkpoint_every', default=1000, type=int, help='validate and save checkpoint every')
        self.parser.add_argument('--visualize_training', default=1, type=int, help='visualize training in web')

        self.parser.add_argument('--is_train', default=1, type=int, help='is train')

        self.parser.add_argument('--lambda_p', type=float, default=0.05, help='weight for perpectual loss')
