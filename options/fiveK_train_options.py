from .fiveK_base_options import BaseOptions


class TrainOptions(BaseOptions):
    """Train option class"""

    def __init__(self):
        super(TrainOptions, self).__init__()

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
        self.parser.add_argument('--reinforce', default=1, type=int, help='0: fs, 1 ws, 2 us. train reinforce')
        self.parser.add_argument('--operator_supervise', default=0, type=int,
                                 help='0: operator reward, 1: operator cross entropy')
        self.parser.add_argument('--batch_size', default=1, type=int, help='batch size')
        self.parser.add_argument('--iter_size', default=16, type=int, help='update loss every iter_size')
        self.parser.add_argument('--learning_rate', default=1e-3, type=float, help='learning rate')
        self.parser.add_argument('--entropy_factor', default=0.05, type=float, help='entropy weight in reinforce loss')
        self.parser.add_argument('--explore_prob', default=0.05, type=float, help='possibility to do pure exploration')
        self.parser.add_argument('--num_iters', default=200000, type=int, help='total number of iterations')
        self.parser.add_argument('--reward_decay', default=1, type=float, help='decay weight for reward moving average')
        self.parser.add_argument('--print_every', default=100, type=int, help='display every')
        self.parser.add_argument('--visualize_every', default=100, type=int, help='visualize training with tensorboard')
        self.parser.add_argument('--checkpoint_every', default=1000, type=int, help='validate and save checkpoint every')
        self.parser.add_argument('--visualize_training', default=1, type=int, help='visualize training in web')
        self.parser.add_argument('--sc_flag', default=0, type=int, help='self critic flag')

        self.parser.add_argument('--is_train', default=0, type=int, help='is train')
        self.is_train = False

        # RL penalty
        self.parser.add_argument('--exploration_penalty', default=0.05, type=float, help='exploration penalty')
        self.parser.add_argument('--operator_usage_penalty', default=1.0, type=float, help='operator usage penalty')

        # ddpg
        self.parser.add_argument('--target_tau', default=0.001, type=float, help='moving average for target network')
        self.parser.add_argument('--rmsize', default=800, type=int, help='replay memory size')
        self.parser.add_argument('--rm_batch_size', default=64, type=int, help='replay memory batch size')
        self.parser.add_argument('--param_noise_factor', default=0.6, type=int, help='3 tau spans factor of value range')
        self.parser.add_argument('--warmup', default=400, type=int, help='timestep without training but only filling the replay memory')
        self.parser.add_argument('--episode_train_times', default=10, type=int, help='train times for each episode')

        # modular
        self.parser.add_argument('--margin', default=0, type=float, help='the margin of triplet loss')
        self.parser.add_argument('--tri_lam', default=1, type=float, help='the margin of triplet loss')
        self.parser.add_argument('--perceptual_loss', default=1, type=int, help='use perceptual loss')

        # Debug
        self.parser.add_argument('--GT_OP_DEBUG', default=0, type=int, help='debug RL training force gt operator')
