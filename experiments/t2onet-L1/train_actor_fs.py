import os
import time
import pdb
import json

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import myhtml
from core.options.fiveK_train_options import TrainOptions
from core.datasets_.FiveKdataset import FiveK, FiveKAct
from core.models.actor import Actor
from core.test_actor_fs import test

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(model, train_loader, val_loader, optimizer, crtNLL, crtMSE, opt):
    """train model"""
    model.train()
    itr = 0
    epoch = 0
    num_iters = opt.num_iters
    avg_time = 0
    stats = {
        'val_dist': [],
        'train_iter': [],
        'best_iter': 0,
        'best_val_dist': float('inf'),
    }
    ckpt_dir = os.path.join(opt.run_dir, 'fs_actor_model')
    writer = SummaryWriter(log_dir=os.path.join(opt.run_dir, '..', 'runs',
                                                'fs_actor_{}_trial_{}'.format(opt.dataset, opt.trial)))
    avg_op_loss, avg_param_loss, avg_loss = 0, 0, 0
    while itr < num_iters:
        epoch += 1
        for i, data in enumerate(train_loader):
            itr += 1
            tik = time.time()
            img_x, img_y, x, y, gt_params, req = data
            # ship to device
            x, y, img_x, img_y, gt_params = list(map(lambda r: r.to(device), [x, y, img_x, img_y, gt_params]))
            step = (y != opt.null_id).sum(1).max().item()  # Variable (batch, )
            pred_imgs, pred_params, pred_logprobs = model.supervised_forward(x, y, img_x, img_y, gt_params, mask=None)
            pred_logprobs.view(-1, 11)
            # debug
            # bs, seq_len, _ = pred_logprobs.shape
            # tops = pred_logprobs.topk(1)[1].view(bs, seq_len)
            # print('tops', tops[:4].cpu().numpy())
            # print('gt', y[:4].cpu().numpy())

            target = y[:, 1:step].contiguous().view(-1)
            _, _, n_cls = pred_logprobs.shape
            pred = pred_logprobs.view(-1, n_cls)
            op_loss = crtNLL(pred, target)
            param_loss = crtMSE(pred_params, gt_params[:, :step-2]) / ((gt_params[:, :step-2] != 0).sum())
            loss = op_loss + param_loss

            # debug
            # print([(pred_params[i, 0, 0].item(), gt_params[:, :step-2][i, 0, 0].item()) for i in range(32)])
            # item_dist = np.array([crtMSE(pred_params[i], gt_params[:, :step-2][i]).item() for i in range(32)])
            # ids = np.where(item_dist > 5)[0]
            # print([(id, item_dist[id]) for id in ids])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            avg_op_loss = avg_op_loss * (1 - 1/itr) + op_loss.item()/itr
            avg_param_loss = avg_param_loss * (1 - 1/itr) + param_loss.item()/itr
            avg_loss = avg_op_loss + avg_param_loss
            tok = time.time()
            avg_time = avg_time * (1 - 1/itr) + (tok - tik) / itr

            if itr % opt.print_every == 0:
                print('iter {:6d} / {}, epoch {:2d}, avg op loss {:.2f}, avg param loss {:.2f} avg loss {:.2f} time {:.2f}'.format(itr, num_iters, epoch, avg_op_loss, avg_param_loss, avg_loss, avg_time))
                writer.add_scalar('train/op_loss', op_loss.item(), itr)
                writer.add_scalar('train/param_loss', param_loss.item(), itr)
                writer.add_scalar('train/loss', loss.item(), itr)

            if itr % opt.checkpoint_every == 0 or itr >= num_iters:

                print('start evaluation...')
                init_val_dist, val_dist = test(model, val_loader, opt)
                model.train()
                print('validation L1 dist {:.2f}'.format(val_dist))
                print('validation init L1 dist {:.2f}'.format(init_val_dist))
                stats['val_dist'].append(val_dist)
                stats['train_iter'].append(itr)

                print('save check point at iter {}'.format(itr))
                save_model_dir = os.path.join(ckpt_dir, 'checkpoint_iter{:08d}'.format(itr))
                os.makedirs(save_model_dir, exist_ok=True)
                print('model dir', save_model_dir)

                torch.save(model.state_dict(), os.path.join(save_model_dir, 'actor.pth'))

                with open(os.path.join(save_model_dir, 'checkpoint_iter{:08d}.json'.format(itr)), 'w') as f:
                    json.dump(stats, f)

                if val_dist < stats['best_val_dist']:
                    print('best model')
                    stats['best_val_dist'] = val_dist
                    stats['best_iter'] = itr
                    best_model_dir = os.path.join(ckpt_dir, 'checkpoint_best')
                    os.makedirs(best_model_dir, exist_ok=True)
                    torch.save(model.state_dict(), os.path.join(best_model_dir, 'actor.pth'))
                    with open(os.path.join(best_model_dir, 'checkpoint_best.json'), 'w') as f:
                        json.dump(stats, f)
            if itr >= num_iters:
                break


def set_web():
    web_dir = os.path.join(opt.run_dir, 'val', 'web')
    img_dir = os.path.join(web_dir, 'images')
    webpage = myhtml.HTML(web_dir, 'val result', reflesh=1)
    webpage.add_header('Visualization of train result for trial {}'.format(opt.trial))
    return webpage, img_dir


if __name__ == '__main__':
    # options
    opt = TrainOptions().parse()
    resume = False

    # data loader
    img_dir = 'data/FiveK/images'
    anno_dir = 'data/FiveK/annotations'
    act_dir = 'output/actions_set_{}'.format(opt.action_id)
    train_dataset = FiveKAct(img_dir, anno_dir, act_dir, opt.vocab_dir, 'train', opt.session)
    train_loader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=opt.num_workers)
    val_dataset = FiveK(img_dir, anno_dir, opt.vocab_dir, 'val', opt.session)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=1)

    # init model
    model = Actor(opt)
    if resume:
        ckpt_dir = os.path.join(opt.run_dir, 'fs_actor_model')
        model_dir = os.path.join(ckpt_dir, 'checkpoint_best')
        model.load_state_dict(torch.load(os.path.join(model_dir, 'actor.pth')))
        print('loaded model from {}'.format(model_dir))
    model.cuda()

    # optimizer
    optimizer = torch.optim.Adam(filter(lambda x: x.requires_grad, model.parameters()), lr=opt.learning_rate)

    # criterion
    crtNLL = torch.nn.NLLLoss()
    crtMSE = torch.nn.MSELoss(reduction='sum')

    # train
    train(model, train_loader, val_loader, optimizer, crtNLL, crtMSE, opt)

