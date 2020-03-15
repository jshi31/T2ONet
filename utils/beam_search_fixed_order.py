import time
import torch.nn as nn
from scipy.optimize import minimize

from core.utils_.utils import *
from operators import img2tensor, tensor2img
from core.models_.discriminator import Discriminator
from core.models_.disc_resnet import ResNet_wobn
from core.models_.lang_encoder1 import RNNEncoder
from core.models.actor import Actor
from core.models.seq2seqGAN.seq2seqGAN import Pix2PixHDModel
from core.options.seq2seqGAN_train_options import TrainOptions
from core.datasets_.FiveKdataset import FiveK
from core.executors.request_executor import Executor


# beam search editing
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_self_disc(opt):
    # vis model
    vis_encoder = ResNet_wobn(12, 18, 512)
    # text model
    vocab = load_vocab(opt.req_vocab_path, opt.op_vocab_path)
    glv_embedding = load_embedding(opt.word2vec_path)
    enc_param = {
        'vocab_size': len(vocab['command_token_to_idx']),
        'word_embedding_size': opt.word_vec_dim,
        'hidden_size': opt.hidden_size,
        'n_spec_token': 4,
        'bidirectional': True,
        'input_dropout_p': 0.2,
        'dropout_p': 0.2,
        'n_layers': 1,
        'rnn_type': 'lstm',
        'variable_lengths': True,
        'word2vec': glv_embedding,
        'fix_embedding': False,
        'pad_id': opt.null_id
    }
    txt_encoder = RNNEncoder(**enc_param)
    # model
    model = Discriminator(vis_encoder, txt_encoder)
    # load param
    ckpt_dir = os.path.join(opt.run_dir, 'self_disc_model')
    model_dir = os.path.join(ckpt_dir, 'checkpoint_best')
    # model_dir = os.path.join(ckpt_dir, 'checkpoint_iter00180000')
    model.load_state_dict(torch.load(os.path.join(model_dir, 'self-disc.pth')))
    print('loaded model from {}'.format(model_dir))
    # to device
    model.cuda()
    return model


def load_seq2seqgan_disc(opt):
    # init model
    actor = Actor(opt)
    model = Pix2PixHDModel(actor, opt)
    # load param
    ckpt_dir = os.path.join(opt.run_dir, 'seq2seqGAN_model')
    # model_dir = os.path.join(ckpt_dir, 'checkpoint_iter00009000')
    model_dir = os.path.join(ckpt_dir, 'checkpoint_best')
    model.load_state_dict(torch.load(os.path.join(model_dir, 'model.pth')), strict=False)
    print('loaded model from {}'.format(model_dir))
    model.cuda()
    model = nn.DataParallel(model, device_ids=opt.gpu_ids)
    return model

# def get_param_naive(img, out, txt, mask, param0, executor, op_ind, dist_type):
#     """ standard way to estimate the parameter from img to out
#     :param img: image before operation [tensor] (1, 3, h, w)
#     :param out: image before operation [tensor] (1, 3, h, w)
#     :param txt: text request
#     :param mask: mask [tensor] (1, 3, h, w]
#     :param param: initial parameter that out -> img
#     :param opname: operator name that out -> img
#     :param cfg:
#     :return: param: predicted best parameter
#     :return success_flag: boolean indicating whether it is successful
#     """
#     def func(param):
#         param = torch.tensor([param], dtype=torch.float).to(img.device)
#         pred_out, pred_param = executor.execute(img, op_ind, None, specified_param=param, has_noise=False)
#         # calc dist
#         if dist_type == 'self-disc':
#             err = get_disc_dist(img, pred_out, txt, discriminator).item()
#         else:
#             err = get_dist(pred_out, out, dist_type).item()
#         return err
#
#     res = minimize(func, param0, method='Nelder-Mead')
#     param = list(res.x)
#     success_flag = res.success
#     return torch.tensor([param]).to(img.device), success_flag

def get_param_naive(img, out, txt, mask, param0, executor, op_ind, dist_type, optimizer):
    """ standard way to estimate the parameter from img to out
    :param img: image before operation [tensor] (1, 3, h, w)
    :param out: image before operation [tensor] (1, 3, h, w)
    :param txt: text request
    :param mask: mask [tensor] (1, 3, h, w]
    :param param: initial parameter that out -> img
    :param dist_type: distance type
    :param optimizer: 'Nelder-Mead', 'adam', 'lbfgs'
    :return: param: predicted best parameter
    :return success_flag: boolean indicating whether it is successful
    """
    def func(param):
        param = torch.tensor([param], dtype=torch.float).to(img.device)
        pred_out, pred_param = executor.execute(img, op_ind, None, specified_param=param, has_noise=False)
        # calc dist
        if dist_type == 'self-disc':
            err = get_disc_dist(img, pred_out, txt, discriminator).item()
        elif dist_type == 'seq2seqGAN-disc':
            err = get_seq2seqGAN_disc_dist(img, pred_out, txt, discriminator).item()
        else:
            err = get_dist(pred_out, out, dist_type).item()
        return err
    res = minimize(func, param0, method='Nelder-Mead')
    param = list(res.x)
    success_flag = res.success
    return torch.tensor([param]).to(img.device), success_flag


def gd_minimize(func, param0, method='adam'):
    num_iters = 1000
    tol = 1e-5
    param0.requires_grad_()

    def closure():
        optimizer.zero_grad()
        loss = func(param0)
        # print('loss', loss.item())
        cur_loss = loss.item()
        # print('loss', cur_loss)
        loss.backward()
        return loss
    # init optimizer
    if method == 'lbfgs':
        success_flag = True
        optimizer = torch.optim.LBFGS([param0], lr=1)
        optimizer.step(closure)
    elif method == 'adam':
        optimizer = torch.optim.Adam([param0], lr=1e-2)
        loss_prev = 10000
        success_flag = False

        for i in range(num_iters):
            optimizer.zero_grad()
            loss = func(param0)
            cur_loss = loss.item()
            if (loss_prev - cur_loss) < tol:
                success_flag = True
                break
            loss_prev = cur_loss
            loss.backward()
            optimizer.step()
    param = param0.detach()
    return param, success_flag


def get_param_gd(img, out, txt, mask, param0, executor, op_ind, dist_type, optimizer):
    """gradient descent to optimize the problem"""
    def func(param):
        # param is tensor of shape (1, num_param)
        pred_out, pred_param = executor.execute(img, op_ind, None, specified_param=param, has_noise=False)
        # calc dist
        if dist_type == 'self-disc':
            err = get_disc_dist(img, pred_out, txt, discriminator)
        elif dist_type == 'seq2seqGAN-disc':
            err = get_seq2seqGAN_disc_dist(img, pred_out, txt, discriminator)
        else:
            err = get_dist(pred_out, out, dist_type)
        return err
    param, success_flag = gd_minimize(func, param0, method=optimizer)
    return param, success_flag


# def get_param(I0, I1, txt, operation, executor, dist_type):
#     param_num = executor.get_param_num(operation)
#     if operation in [0, 1, 2, 6]:
#         param0 = torch.zeros(param_num)
#     elif operation in [3, 5]:
#         param0 = torch.ones(param_num)
#     else:
#         assert False, 'the operation is not global operation'
#     param, sucess_flag = get_param_naive(I0, I1, txt, None, param0, executor, operation, dist_type)
#     return param, sucess_flag


def get_param(I0, I1, txt, operation, executor, dist_type, optimizer):
    param_num = executor.get_param_num(operation)
    if operation in [0, 1, 2, 6]:
        param0 = torch.zeros(param_num)
    elif operation in [3, 5]:
        param0 = torch.ones(param_num)
    else:
        assert False, 'the operation is not global operation'
    if optimizer == 'Nelder-Mead':
        param, sucess_flag = get_param_naive(I0, I1, txt, None, param0, executor, operation, dist_type, optimizer)
    else:
        param0 = param0.view(1, -1).to(I0.device)
        param, sucess_flag = get_param_gd(I0, I1, txt, None, param0, executor, operation, dist_type, optimizer)
    return param, sucess_flag


def execute(I, operation, param, executor):
    img, _ = executor.execute(I, operation, None, features=None, specified_param=param, has_noise=False)
    return img


def get_dist(x1, x2, dist_type):
    # L1 distance
    if dist_type == 'L1':
        dist = (x1 - x2).norm(1) / x1.numel()
    elif dist_type == 'L2':
        dist = (x1 - x2) ** 2 / x1.numel()
    elif dist_type == 'perceptual':
        pass
    else:
        assert False, '{} is invalid distance'.format(dist_type)
    return dist


def get_disc_dist(img1, img2, txt, discriminator):
    img = torch.cat([img1, img2], dim=1)
    # score = F.sigmoid(discriminator(img, txt))
    score = discriminator(img, txt)
    return 1 - score


def get_seq2seqGAN_disc_dist(img1, img2, txt, discriminator):
    losses, generated = discriminator(img1, img2, txt)
    # sum per device losses
    losses = [torch.mean(x) if not isinstance(x, int) else x for x in losses]
    loss_dict = dict(zip(discriminator.module.loss_names, losses))
    score = loss_dict['D_real']
    return 1 - score


def beam_search(I_0, I_gt, txt, executor, beam_size, operations, operation_names, max_step, err, dist_type, optimizer, replace=False):
    """
    :param I_0: input image
    :param I_gt: target image
    :param txt: request tokens. None if dist_type is not 'self-disc'
    :param executor: executor
    :param beam_size: beam size
    :param operations: list of operation index
    :param operations: list of operation names
    :param max_step: max step
    :param err: error
    :param dist_type: 'L1', 'L2', 'perceptual', 'self-disc'
    :param replace: whether select operations repeatitively
    :return: sequence of actions
    """
    min_dist = float('inf')
    sequences = [[[], float('inf')]] # (operation, param)
    I_buff = [I_0]
    for i in range(max_step):
        all_candidates = []
        I_tmp_list = []
        no_update_flag = True
        finish_flag = False
        tmp_min_dists = []
        for j, I in enumerate(I_buff):
            operation = operations[i]
            if not replace and operation in [operation_names.index(v[0]) for v in sequences[j][0]]:
                continue
            if dist_type == 'self-disc':
                param, success_flag = get_param(I_0, I, txt, operation, executor, dist_type, optimizer)
            elif dist_type == 'seq2seqGAN-disc':
                param, success_flag = get_param(I_0, I, txt, operation, executor, dist_type, optimizer)
            else:
                param, success_flag = get_param(I, I_gt, txt, operation, executor, dist_type, optimizer)
            I_out = execute(I, operation, param, executor)
            # calc dist
            if dist_type == 'self-disc':
                dist = get_disc_dist(I_0, I_out, txt, discriminator).item()
            elif dist_type == 'seq2seqGAN-disc':
                dist = get_seq2seqGAN_disc_dist(I_0, I_out, txt, discriminator).item()
            else:
                dist = get_dist(I_out, I_gt, dist_type).item()
            # judge if I_out should be added
            if dist < min_dist:
                tmp_min_dists.append(dist)
                candidate = [sequences[j][0] + [(operation_names[operation], param[0].tolist(), dist, I_out.cpu())], dist]
                all_candidates.append(candidate)
                I_tmp_list.append(I_out)
                no_update_flag = False
                if dist < err:
                    finish_flag = True
        min_dist = min(tmp_min_dists) if len(tmp_min_dists) > 0 else min_dist

        # if all_candidates length is less than beam_size
        if len(all_candidates) < beam_size:
            all_candidates += sequences
            I_tmp_list += I_buff
        # order all candidates by score
        dists = np.array([v[1] for v in all_candidates])
        order = np.argsort(dists)
        # select beam size best
        sequences = [all_candidates[idx] for idx in order][:beam_size]
        I_buff = [I_tmp_list[idx] for idx in order][:beam_size]
        if no_update_flag or finish_flag:
            break
    # get all actions and images
    actions = [[act[:-1] for act in seq[0]] for seq in sequences]
    Is = [[act[-1] for act in seq[0]] for seq in sequences]
    return actions, Is


if __name__ == '__main__':
    # options
    opt = TrainOptions().parse()
    opt.is_train = False

    phase = 'train'

    # data loader
    img_dir = 'data/FiveK/images'
    anno_dir = 'data/FiveK/annotations'
    dataset = FiveK(img_dir, anno_dir, opt.vocab_dir, phase, opt.session)
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=opt.num_workers)

    # executor
    executor = Executor(opt)

    save_dir = 'output/greedy_analysis'
    operation_names = ['brightness', 'contrast', 'saturation', 'color', 'inpaint', 'tone', 'sharpness', 'white']
    beam_size = 3
    operations = [0, 1, 2, 3, 5, 6]
    # err = 1e-2
    err = -10
    # dist_type = 'self-disc'
    # dist_type = 'L1'
    dist_type = 'seq2seqGAN-disc'
    optimizer = 'Nelder-Mead'
    # optimizer = 'adam'
    # optimizer = 'lbfgs'

    replace = False

    # discriminator
    if dist_type == 'self-disc':
        discriminator = load_self_disc(opt)
        discriminator.eval()
    elif dist_type == 'seq2seqGAN-disc':
        discriminator = load_seq2seqgan_disc(opt)
        discriminator.eval()

    itr = 0
    avg_time = 0
    for i, data in enumerate(loader):
        itr += 1
        if itr <= 2:
            continue
        if itr > opt.num_iters:
            break
        tik = time.time()
        img_x, img_y, x, req = data
        x, img_x, img_y = list(map(lambda r: r.to(device), [x, img_x, img_y]))

        seqs, imgs = beam_search(img_x, img_y, x, executor, beam_size, operations, operation_names, len(operations), err, dist_type, optimizer, replace)
        tok = time.time()
        avg_time = avg_time * (1 - 1/itr) + (tok - tik)/itr
        print('iter {}/{} time {:.2f}s'.format(itr, opt.num_iters, avg_time))

        img_dir = os.path.join(save_dir, '{}{}'.format(phase, i))
        os.makedirs(img_dir, exist_ok=True)
        # calc init dist
        if dist_type == 'self-disc':
            init_dist = get_disc_dist(img_x, img_y, x, discriminator).item()
        elif dist_type == 'seq2seqGAN-disc':
            init_dist = get_seq2seqGAN_disc_dist(img_x, img_y, x, discriminator).item()
        else:
            init_dist = get_dist(img_x, img_y, dist_type).item()
        # write operation episode
        info = {'request': req, 'init distance': init_dist, 'operation sequence': seqs}
        with open(os.path.join(img_dir, '{}_ops.json'.format(dist_type)), 'w') as f:
            json.dump(info, f)

        # write img
        cv2.imwrite(os.path.join(img_dir, 'input.jpg'), tensor2img(img_x))
        cv2.imwrite(os.path.join(img_dir, 'gt.jpg'), tensor2img(img_y))
        for idx in range(len(seqs)):
            cv2.imwrite(os.path.join(img_dir, '{}_edit{}.jpg'.format(dist_type, idx)), tensor2img(imgs[idx][-1]))


