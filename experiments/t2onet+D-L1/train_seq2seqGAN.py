import os
import time
import pdb
import json

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import myhtml
from core.options.seq2seqGAN_train_options import TrainOptions
from core.datasets_.FiveKdataset import FiveK, FiveKAct
from core.models.actor import Actor
from core.models.seq2seqGAN.seq2seqGAN import Pix2PixHDModel
from core.test_seq2seqGAN import test

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(model, train_loader, val_loader, optimizer_fs, optimizer_G, optimizer_D, crtNLL, crtMSE, opt):
    """train model"""
    model.train()
    itr = 0
    epoch = 0
    num_iters = opt.num_iters
    avg_fs_time = 0
    avg_GAN_time = 0
    stats = {
        'val_dist': [],
        'train_iter': [],
        'best_iter': 0,
        'best_val_dist': float('inf'),
    }
    ckpt_dir = os.path.join(opt.run_dir, 'seq2seqGAN_model')
    writer = SummaryWriter(log_dir=os.path.join(opt.run_dir, '..', 'runs',
                                                'seq2seqGAN{}_trial_{}'.format(opt.dataset, opt.trial)))
    avg_op_loss, avg_param_loss, avg_loss = 0, 0, 0
    avg_D_fake_loss, avg_D_real_loss = 0, 0
    avg_G_GAN_loss, avg_G_GAN_Feat_loss, avg_G_VGG_loss = 0, 0, 0
    while itr < num_iters:
        epoch += 1
        for i, data in enumerate(train_loader):
            itr += 1
            tik = time.time()
            img_x, img_y, x, y, gt_params, req = data
            # ship to device
            x, y, img_x, img_y, gt_params = list(map(lambda r: r.to(device), [x, y, img_x, img_y, gt_params]))

            # fully supervised forward GAN forward backward
            if itr % 2 == 1:
                step = (y != opt.null_id).sum(1).max().item()  # Variable (batch, )
                pred_imgs, pred_params, pred_logprobs = model.actor.supervised_forward(x, y, img_x, img_y, gt_params, mask=None)
                pred_logprobs.view(-1, 11)

                target = y[:, 1:step].contiguous().view(-1)
                _, _, n_cls = pred_logprobs.shape
                pred = pred_logprobs.view(-1, n_cls)
                op_loss = crtNLL(pred, target)
                param_loss = crtMSE(pred_params, gt_params[:, :step-2]) / ((gt_params[:, :step-2] != 0).sum())
                loss = op_loss + param_loss

                optimizer_fs.zero_grad()
                loss.backward()
                optimizer_fs.step()

                avg_op_loss = avg_op_loss * (1 - 1/(itr//2 + 1)) + op_loss.item()/(itr//2 + 1)
                avg_param_loss = avg_param_loss * (1 - 1/(itr//2 + 1)) + param_loss.item()/(itr//2 + 1)
                avg_loss = avg_op_loss + avg_param_loss
                tok = time.time()
                avg_fs_time = avg_fs_time * (1 - 1/(itr//2 + 1)) + (tok - tik) / (itr//2 + 1)

            # GAN forward backward
            else:
                losses, generated = model(img_x, img_y[:, -1], x)
                # sum per device losses
                losses = [torch.mean(x) if not isinstance(x, int) else x for x in losses]
                loss_dict = dict(zip(model.loss_names, losses))

                # calculate final loss scalar
                loss_D = (loss_dict['D_fake'] + loss_dict['D_real']) * 0.5
                loss_G = loss_dict['G_GAN'] + loss_dict.get('G_GAN_Feat', 0) + loss_dict.get('G_VGG', 0)

                # update generator weights
                optimizer_G.zero_grad()
                loss_G.backward()
                optimizer_G.step()

                # update discriminator weights
                optimizer_D.zero_grad()
                loss_D.backward()
                optimizer_D.step()

                avg_D_real_loss = avg_D_real_loss * (1 - 1/(itr//2)) + loss_dict['D_real'].item()/(itr//2)
                avg_D_fake_loss = avg_D_fake_loss * (1 - 1/(itr//2)) + loss_dict['D_fake'].item()/(itr//2)
                avg_G_GAN_loss = avg_G_GAN_loss * (1 - 1/(itr//2)) + loss_dict['G_GAN'].item()/(itr//2)
                if not opt.no_ganFeat_loss:
                    avg_G_GAN_Feat_loss = avg_G_GAN_Feat_loss * (1 - 1/(itr//2)) + loss_dict['G_GAN_Feat'].item()/(itr//2)
                if not opt.no_vgg_loss:
                    avg_G_VGG_loss = avg_G_VGG_loss * (1 - 1/(itr//2)) + loss_dict['G_GAN'].item()/(itr//2)
                tok = time.time()
                avg_GAN_time = avg_GAN_time * (1 - 1/(itr//2)) + (tok - tik)/(itr//2)

            if itr % opt.print_every == 0:
                print('iter {:6d} / {}, epoch {:2d}, op loss {:.2f}, param loss {:.2f}, fs loss {:.2f}, D real {:.2f}, D fake {:.2f}, G_GAN {:.2f}, GAN feat {:.2f}, VGG {:.2f}, fs time {:.2f}, GAN time {:.2f}'.format(itr, num_iters, epoch, avg_op_loss, avg_param_loss, avg_loss, avg_D_real_loss, avg_D_fake_loss, avg_G_GAN_loss, avg_G_GAN_Feat_loss, avg_G_VGG_loss, avg_fs_time, avg_GAN_time))
                writer.add_scalar('train/op_loss', op_loss.item(), itr)
                writer.add_scalar('train/param_loss', param_loss.item(), itr)
                writer.add_scalar('train/fs_loss', loss.item(), itr)
                writer.add_scalar('train/G_loss', loss_G.item(), itr)
                writer.add_scalar('train/D_loss', loss_D.item(), itr)

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

                torch.save(model.state_dict(), os.path.join(save_model_dir, 'model.pth'))

                with open(os.path.join(save_model_dir, 'checkpoint_iter{:08d}.json'.format(itr)), 'w') as f:
                    json.dump(stats, f)

                if val_dist < stats['best_val_dist']:
                    print('best model')
                    stats['best_val_dist'] = val_dist
                    stats['best_iter'] = itr
                    best_model_dir = os.path.join(ckpt_dir, 'checkpoint_best')
                    os.makedirs(best_model_dir, exist_ok=True)
                    torch.save(model.state_dict(), os.path.join(best_model_dir, 'model.pth'))
                    with open(os.path.join(best_model_dir, 'checkpoint_best.json'), 'w') as f:
                        json.dump(stats, f)

            ### instead of only training the local enhancer, train the entire network after certain iterations
            if (opt.niter_fix_global != 0) and (epoch == opt.niter_fix_global):
                model.update_fixed_params()

            ### linearly decay learning rate after certain iterations
            if epoch > opt.niter:
                model.update_learning_rate()

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
    act_dir = 'output/actions_set_1'
    train_dataset = FiveKAct(img_dir, anno_dir, act_dir, opt.vocab_dir, 'train', opt.session)
    train_loader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=opt.num_workers)
    val_dataset = FiveK(img_dir, anno_dir, opt.vocab_dir, 'val', opt.session)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=1)

    # init model
    actor = Actor(opt)
    model = Pix2PixHDModel(actor, opt)
    model.cuda()
    if resume:
        ckpt_dir = os.path.join(opt.run_dir, 'fs_actor_model')
        model_dir = os.path.join(ckpt_dir, 'checkpoint_best')
        model.load_state_dict(torch.load(os.path.join(model_dir, 'actor.pth')))
        print('loaded model from {}'.format(model_dir))

    # optimizer fs
    optimizer_fs = torch.optim.Adam(filter(lambda x: x.requires_grad, model.actor.parameters()), lr=opt.learning_rate)
    # optimizer G
    optimizer_G = torch.optim.Adam(filter(lambda x: x.requires_grad, model.actor.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))

    # optimizer D
    params = list(model.netD.parameters())
    optimizer_D = torch.optim.Adam([{'params': filter(lambda x: x.requires_grad, list(model.netD.parameters()))}, {'params': model.cond_encoder.parameters()}], lr=opt.lr, betas=(opt.beta1, 0.999))

    # criterion
    crtNLL = torch.nn.NLLLoss()
    crtMSE = torch.nn.MSELoss(reduction='sum')

    # train
    train(model, train_loader, val_loader, optimizer_fs, optimizer_G, optimizer_D, crtNLL, crtMSE, opt)

