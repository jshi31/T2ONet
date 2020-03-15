import pdb
import numpy as np
import torch
import os
from torch.autograd import Variable
from core.models.actor import Actor
from . import networks

class Pix2PixHDModel(torch.nn.Module):
    def name(self):
        return 'Pix2PixHDModel'

    def init_loss_filter(self, use_gan_feat_loss, use_vgg_loss):
        flags = (True, use_gan_feat_loss, use_vgg_loss, True, True)
        def loss_filter(g_gan, g_gan_feat, g_vgg, d_real, d_fake):
            return [l for (l,f) in zip((g_gan,g_gan_feat,g_vgg,d_real,d_fake),flags) if f]
        return loss_filter

    def __init__(self, actor, opt):
        super(Pix2PixHDModel, self).__init__()
        self.opt = opt
        self.gpu_ids = opt.gpu_ids
        self.isTrain = opt.is_train
        self.Tensor = torch.cuda.FloatTensor if self.gpu_ids else torch.Tensor

        if not opt.is_train: # when training at full res this causes OOM
            torch.backends.cudnn.benchmark = True
        self.use_features = opt.instance_feat or opt.label_feat
        self.gen_features = self.use_features and not self.opt.load_features
        input_nc = opt.input_nc

        ##### define networks
        # Generator network
        self.actor = actor
        self.lang_encoder = actor.lang_encoder

        # Discriminator network
        use_sigmoid = opt.no_lsgan
        netD_input_nc = input_nc + opt.output_nc
        self.netD = networks.define_D(netD_input_nc, opt.cond_nc, opt.ndf, opt.n_layers_D, opt.norm, use_sigmoid,
                                      opt.num_D, not opt.no_ganFeat_loss, gpu_ids=self.gpu_ids)
        # condition encoding
        self.cond_encoder = networks.ConditionEncoding(opt.cond_nc)

                # set loss functions and optimizers
        self.old_lr = opt.lr

        # define loss functions
        self.loss_filter = self.init_loss_filter(not opt.no_ganFeat_loss, not opt.no_vgg_loss)

        self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan, tensor=self.Tensor)
        self.criterionFeat = torch.nn.L1Loss()
        if not opt.no_vgg_loss:
            self.criterionVGG = networks.VGGLoss(self.gpu_ids)


            # Names so we can breakout loss
        self.loss_names = self.loss_filter('G_GAN','G_GAN_Feat','G_VGG','D_real', 'D_fake')


    def discriminate(self, input_label, test_image, txt_feat, use_pool=False):
        input_concat = torch.cat((input_label, test_image.detach()), dim=1)
        assert not use_pool, 'use_pool must be False'
        if use_pool:
            # no use pool
            fake_query = self.fake_pool.query(input_concat)
            return self.netD.forward(fake_query)
        else:
            return self.netD.forward(input_concat, txt_feat)

    def forward(self, src_img, trg_img, txt, infer=False):
        _, fake_imgs, pred_ops, _ = self.actor.episode_forward(txt, src_img, None)
        bs, max_len, cn, h, w = fake_imgs.shape

        end_fake_imgs = []
        # for loop to get column index with end token
        for bs_i in range(bs):
            idxs = (pred_ops[bs_i] == self.opt.end_id).nonzero()
            col_idx = idxs[0][0] if len(idxs) > 0 else max_len - 1
            end_fake_imgs.append(fake_imgs[bs_i, col_idx])
        fake_img = torch.stack(end_fake_imgs)
        # fake_img = fake_imgs[:, -1]

        # encode text feature (even if actor already have such encode feature)
        with torch.no_grad():
            _, txt_feat, _ = self.lang_encoder(txt)
        txt_feat = self.cond_encoder(txt_feat[0])

        # Fake Detection and Loss
        pred_fake_pool = self.discriminate(src_img, fake_img, txt_feat, use_pool=False)
        loss_D_fake = self.criterionGAN(pred_fake_pool, False)

        # Real Detection and Loss
        pred_real = self.discriminate(src_img, trg_img, txt_feat)
        loss_D_real = self.criterionGAN(pred_real, True)

        # GAN loss (Fake Passability Loss)
        pred_fake = self.netD.forward(torch.cat((src_img, fake_img), dim=1), txt_feat.detach())
        loss_G_GAN = self.criterionGAN(pred_fake, True)

        # GAN feature matching loss
        loss_G_GAN_Feat = 0
        if not self.opt.no_ganFeat_loss:
            feat_weights = 4.0 / (self.opt.n_layers_D + 1)
            D_weights = 1.0 / self.opt.num_D
            for i in range(self.opt.num_D):
                for j in range(len(pred_fake[i])-1):
                    loss_G_GAN_Feat += D_weights * feat_weights * \
                                       self.criterionFeat(pred_fake[i][j], pred_real[i][j].detach()) * self.opt.lambda_feat

        # VGG feature matching loss
        loss_G_VGG = 0
        if not self.opt.no_vgg_loss:
            loss_G_VGG = self.criterionVGG(fake_img, trg_img) * self.opt.lambda_feat

        # Only return the fake_B image if necessary to save BW
        return [ self.loss_filter(loss_G_GAN, loss_G_GAN_Feat, loss_G_VGG, loss_D_real, loss_D_fake ), None if not infer else fake_img ]

    def inference(self, label, inst, image=None):
        # Encode Inputs
        image = Variable(image) if image is not None else None
        input_label, inst_map, real_image, _ = self.encode_input(Variable(label), Variable(inst), image, infer=True)

        # Fake Generation
        if self.use_features:
            if self.opt.use_encoded_image:
                # encode the real image to get feature map
                feat_map = self.netE.forward(real_image, inst_map)
            else:
                # sample clusters from precomputed features
                feat_map = self.sample_features(inst_map)
            input_concat = torch.cat((input_label, feat_map), dim=1)
        else:
            input_concat = input_label

        if torch.__version__.startswith('0.4'):
            with torch.no_grad():
                fake_image = self.netG.forward(input_concat)
        else:
            fake_image = self.netG.forward(input_concat)
        return fake_image

    def sample_features(self, inst):
        # read precomputed feature clusters
        cluster_path = os.path.join(self.opt.checkpoints_dir, self.opt.name, self.opt.cluster_path)
        features_clustered = np.load(cluster_path, encoding='latin1').item()

        # randomly sample from the feature clusters
        inst_np = inst.cpu().numpy().astype(int)
        feat_map = self.Tensor(inst.size()[0], self.opt.feat_num, inst.size()[2], inst.size()[3])
        for i in np.unique(inst_np):
            label = i if i < 1000 else i//1000
            if label in features_clustered:
                feat = features_clustered[label]
                cluster_idx = np.random.randint(0, feat.shape[0])

                idx = (inst == int(i)).nonzero()
                for k in range(self.opt.feat_num):
                    feat_map[idx[:,0], idx[:,1] + k, idx[:,2], idx[:,3]] = feat[cluster_idx, k]
        if self.opt.data_type==16:
            feat_map = feat_map.half()
        return feat_map

    def encode_features(self, image, inst):
        image = Variable(image.cuda(), volatile=True)
        feat_num = self.opt.feat_num
        h, w = inst.size()[2], inst.size()[3]
        block_num = 32
        feat_map = self.netE.forward(image, inst.cuda())
        inst_np = inst.cpu().numpy().astype(int)
        feature = {}
        for i in range(self.opt.label_nc):
            feature[i] = np.zeros((0, feat_num+1))
        for i in np.unique(inst_np):
            label = i if i < 1000 else i//1000
            idx = (inst == int(i)).nonzero()
            num = idx.size()[0]
            idx = idx[num//2,:]
            val = np.zeros((1, feat_num+1))
            for k in range(feat_num):
                val[0, k] = feat_map[idx[0], idx[1] + k, idx[2], idx[3]].data[0]
            val[0, feat_num] = float(num) / (h * w // block_num)
            feature[label] = np.append(feature[label], val, axis=0)
        return feature

    def get_edges(self, t):
        edge = torch.cuda.ByteTensor(t.size()).zero_()
        edge[:,:,:,1:] = edge[:,:,:,1:] | (t[:,:,:,1:] != t[:,:,:,:-1])
        edge[:,:,:,:-1] = edge[:,:,:,:-1] | (t[:,:,:,1:] != t[:,:,:,:-1])
        edge[:,:,1:,:] = edge[:,:,1:,:] | (t[:,:,1:,:] != t[:,:,:-1,:])
        edge[:,:,:-1,:] = edge[:,:,:-1,:] | (t[:,:,1:,:] != t[:,:,:-1,:])
        if self.opt.data_type==16:
            return edge.half()
        else:
            return edge.float()

    def save(self, which_epoch):
        self.save_network(self.netG, 'G', which_epoch, self.gpu_ids)
        self.save_network(self.netD, 'D', which_epoch, self.gpu_ids)
        if self.gen_features:
            self.save_network(self.netE, 'E', which_epoch, self.gpu_ids)

    def update_fixed_params(self):
        # after fixing the global generator for a number of iterations, also start finetuning it
        params = list(self.netG.parameters())
        if self.gen_features:
            params += list(self.netE.parameters())
        self.optimizer_G = torch.optim.Adam(params, lr=self.opt.lr, betas=(self.opt.beta1, 0.999))
        if self.opt.verbose:
            print('------------ Now also finetuning global generator -----------')

    def update_learning_rate(self):
        lrd = self.opt.lr / self.opt.niter_decay
        lr = self.old_lr - lrd
        for param_group in self.optimizer_D.param_groups:
            param_group['lr'] = lr
        for param_group in self.optimizer_G.param_groups:
            param_group['lr'] = lr
        if self.opt.verbose:
            print('update learning rate: %f -> %f' % (self.old_lr, lr))
        self.old_lr = lr

class InferenceModel(Pix2PixHDModel):
    def forward(self, inp):
        label, inst = inp
        return self.inference(label, inst)


