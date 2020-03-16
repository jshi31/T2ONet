import math
import os
import pdb

import torch
import torch.nn as nn
import torch.nn.functional as F

from .actor_resnet import ResNet as Vis_encoder
from .lang_encoder import RNNEncoder as Lang_encoder
from .action_decoder import Decoder
from executors.executor import Executor
from utils.text_utils import load_embedding, load_vocab


def create_seq2seq_net(input_vocab_size, output_vocab_size, hidden_size,
                       word_vec_dim, n_layers, bidirectional, variable_lengths,
                       use_attention, decoder_max_len, input_dropout_p,
                       dropout_p, word2vec_path=None, fix_embedding=False, pad_id=0):
    word2vec = None
    if word2vec_path is not None:
        word2vec = load_embedding(word2vec_path)

    n_spec_token = 4
    encoder = Lang_encoder(input_vocab_size, word_vec_dim, hidden_size, n_spec_token,
                           bidirectional=bidirectional, input_dropout_p=input_dropout_p,
                           dropout_p=dropout_p, n_layers=n_layers, variable_lengths=variable_lengths,
                           word2vec=word2vec, fix_embedding=fix_embedding, pad_id=pad_id)

    decoder = Decoder(output_vocab_size, decoder_max_len,
                      word_vec_dim, hidden_size, n_layers,
                      bidirectional=bidirectional, use_attention=use_attention)

    return encoder, decoder

class Actor(nn.Module):
    def __init__(self, opt):
        """init  visual encoder, language encoder, decoder
        compatible for both single step forward and episode forward
        """
        super(Actor, self).__init__()
        self.opt = opt
        self.vocab2id, self.id2vocab, self.op_vocab2id, self.id2op_vocab = load_vocab(opt.vocab_dir, opt.dataset, opt.session)
        self.vis_encoder, self.lang_encoder, self.decoder = self.init_model()
        self.variable_lengths = opt.variable_lengths
        self.null_id = opt.null_id
        self.start_id = opt.start_id
        self.end_id = opt.end_id
        self.executor = Executor(opt)
        self.bn1 = nn.BatchNorm1d(512)


    def _get_net_params(self, opt, vocab2id, op_vocab2id):
        net_params = {
            'input_vocab_size': len(vocab2id),
            'output_vocab_size': len(op_vocab2id),
            'hidden_size': opt.hidden_size,
            'word_vec_dim': opt.word_vec_dim,
            'n_layers': opt.n_layers,
            'bidirectional': bool(opt.bidirectional),
            'variable_lengths': opt.variable_lengths,
            'use_attention': opt.use_attention,
            'decoder_max_len': opt.decoder_max_len,
            'input_dropout_p': opt.input_dropout_p,
            'dropout_p': opt.dropout_p,
            'word2vec_path': os.path.join(opt.vocab_dir, '{}_vocabs_glove_feat_{}.h5'.format(opt.dataset, opt.session)),
            'fix_embedding': opt.fix_input_embedding,
            'pad_id': opt.null_id
        }
        return net_params

    def init_model(self):
        net_params = self._get_net_params(self.opt, self.vocab2id, self.op_vocab2id)
        lang_encoder, decoder = create_seq2seq_net(**net_params)
        vis_encoder = Vis_encoder(3, 18, 512)  # img
        return vis_encoder, lang_encoder, decoder

    def get_gt_mask(self, img, mask_dict, op):
        """get gt mask
        Assumption: if no gt mask in the index, then the next are all global editing.
        :param img (bs, 3, h, w)
        :param op (bs, 1)
        :param mask_dict list of dict
        :return masks (bs, 3, h, w)
        """
        bs = len(mask_dict)
        masks = []
        for i in range(bs):
            if str(op[i][0]) in mask_dict[i]:
                try:
                    mask = torch.from_numpy(mask_dict[i][str(op[i][0])][0]).cuda().expand_as(img[i])
                except:
                    mask = torch.ones_like(img[i]).cuda()
            else:
                mask = torch.ones_like(img[i]).cuda()
            masks.append(mask)
        masks = torch.stack(masks, dim=0)
        return masks

    def divide_op_group(self, ops):
        """
        :param ops: (n,)
        :return unqs: (m,)
        :return group_inds: list of (k,)
        :return inv_inds: (n,) # map back the order of sample in a batch
        """
        group_inds = []
        unqs = torch.unique(ops)
        for unq in unqs:
            inds = torch.nonzero(ops == unq).squeeze(1)
            group_inds.append(inds)
        inds = torch.cat(group_inds)
        rev_inds = torch.argsort(inds)
        return unqs, group_inds, rev_inds

    def supervised_forward(self, x, y, img_x, img_y, gt_params, mask):
        """
        supervesed forward that only learn the language
        :param x: (bs, len_x)
        :param y: (bs, max_y)
        :param img_x: (bs, 3, h, w)
        :param img_y: (bs, max_y - 1, 3, h, w)
        :param gt_params: (bs, max_y - 2, 24)
        :param mask: (bs, len_y - 2, 1, h, w)
        :return: pred_imgs (bs, len_y - 3, 3, h, w)
        :return: pred_params (bs, len_y - 2, 8)
        :return: op_logprob (bs, len_y - 1, n_cls)
        """
        encoder_outputs, encoder_hidden, _ = self.lang_encoder(x) # (n_rnnlayer, 1, hidden_size), (bs, encoder_valid_len, hidden)
        decoder_hidden = self.decoder._init_state(encoder_hidden) # (2, bs, 512)
        bs, _ = y.shape
        step = (y != self.null_id).sum(1).max().item()  # Variable (batch, )
        pred_params = []
        pred_imgs = []
        op_logprobs = []
        if mask is not None:
            bs, _, _, h, w = mask.shape

        ops = y[:, 0].unsqueeze(-1)  # (bs, 1)
        # batch training
        for i in range(1, step): # skip the start
            img_feat = self.vis_encoder(img_x)  # (bs, d)
            img_feat = F.relu(self.bn1(img_feat))
            op_logprob, decoder_hidden, step_attn, context = self.decoder.forward_step(ops, decoder_hidden, encoder_outputs, img_feat)  # (bs, 1, n_cls)
            op_logprobs.append(op_logprob)
            ops = y[:, i].unsqueeze(-1)  # (bs, 1)
            if i == step - 1:
                # obtain the probability for end token and break
                break
            # divide group
            out_gs = []
            out_param_gs = []
            if mask is not None:
                mask = mask.to(img_x.device)

            # divide group
            unqs, group_inds, rev_inds = self.divide_op_group(ops.view(-1))
            for j in range(len(group_inds)):
                op_ind = unqs[j].item()
                inds = group_inds[j]
                group_size = len(inds)
                img_g = torch.index_select(img_x, 0, inds) # (m, 3, h, w) g: group
                context_g = torch.index_select(context, 0, inds)
                mask_g = torch.index_select(mask, 0, inds) if mask is not None else mask # (m, 1, h, w)
                out_g, pred_param_g = self.executor.execute(img_g, op_ind - 3, mask_g, context_g, has_noise=False)
                pred_param_g = torch.cat([pred_param_g, torch.zeros(group_size, 24 - pred_param_g.shape[-1], dtype=torch.float).to(pred_param_g.device)], 1)
                out_gs.append(out_g)
                out_param_gs.append(pred_param_g)
            out_gs = torch.cat(out_gs)
            out_param_gs = torch.cat(out_param_gs)  # (bs, n)
            outs = torch.index_select(out_gs, 0, rev_inds) # recover order
            out_params = torch.index_select(out_param_gs, 0, rev_inds)
            pred_imgs.append(outs)
            pred_params.append(out_params)
            # update image
            img_x = img_y[:, i - 1]
        pred_imgs = torch.stack(pred_imgs, 1)
        pred_params = torch.stack(pred_params, 1)
        pred_logprobs = torch.cat(op_logprobs, 1)

        return pred_imgs, pred_params, pred_logprobs


    def episode_forward(self, x, img_x, mask_dict, reinforce_sample=1):
        """ execute one episode
        :param x: (bs, len_x)
        :param img_x: (bs, 3, h, w)
        :param mask_dict: mask_dict, provided by gt
        :param reinforce_sample:
        :param operator_supervise:

        :return reqs: (bs, len_x)
        :return gt_ops: (bs, len_y)
        :return hist_imgs: (bs, n, 3, h, w)
        :return gt_imgs: (bs, 3, h, w)
        :return hist_ops: (bs, n + 1, 1) # begin with <START>
        :return hist_params: (bs, n + 1, param_len) #  begin with all zeros
        :return masks: (bs, n, 3, h, w)
        :return input_lengths: (bs,)
        """
        batch_size = x.shape[0]
        encoder_outputs, encoder_hidden, _ = self.lang_encoder(x) # (n_rnnlayer, 1, hidden_size), (bs, encoder_valid_len, hidden)
        decoder_hidden = self.decoder._init_state(encoder_hidden)
        decoder_hiddens = [tuple(map(lambda x: x.detach(), decoder_hidden))]

        pred_ops = []
        pred_params = []
        pred_imgs = []
        pred_masks = []
        # block local editing operation
        op_mask = torch.tensor([False, False, True, True, True, True, True, False, True, True, False], dtype=torch.float, device=img_x.device).repeat(batch_size, 1) # (bs, op_len)
        pred_op = torch.tensor([[self.start_id]] * batch_size).to(img_x.device)

        for i in range(0, self.opt.decoder_max_len):
            img_feat = self.vis_encoder(img_x)
            img_feat = F.relu(self.bn1(img_feat))
            # decoder_output (bs, len, n_vocab)
            op_logprob, decoder_hidden, step_attn, context = \
                self.decoder.forward_step(pred_op, decoder_hidden, encoder_outputs, img_feat)
            decoder_hiddens.append(tuple(map(lambda x: x.detach(), decoder_hidden)))
            # update operator probs and add explore prob
            _, _, op_len = op_logprob.shape
            op_probs = torch.exp(op_logprob).squeeze(1)  # (bs, 1, n_vocab)
            op_probs = op_probs * (1 - self.opt.explore_prob) + self.opt.explore_prob
            # add op_probs mask
            # all ops: brightness, contrast, color, saturation, tone inpaint white_op, sharpness, whitebalance, bnw
            op_probs = op_probs * op_mask
            op_probs = op_probs / (op_probs.sum(1, keepdim=True) + 1e-30)
            if reinforce_sample:
                dist = torch.distributions.Categorical(probs=op_probs) # better initialize with logits
                pred_op = dist.sample().view(batch_size, -1)  # (bs, 1)
            else:
                pred_op = op_probs.topk(1)[1].view(batch_size, -1) # (bs, 1)
            # assign op_mask to 0 with predicted op
            for batch_i in range(batch_size):
                op_mask[batch_i, pred_op[batch_i, 0]] = 0

            pred_mask = self.get_gt_mask(img_x, mask_dict, pred_op.detach().cpu().numpy()) \
                if mask_dict is not None else mask_dict

            # divide group
            out_gs = []
            out_param_gs = []
            unqs, group_inds, rev_inds = self.divide_op_group(pred_op.view(-1))
            for j in range(len(group_inds)):
                op_ind = unqs[j].item()
                inds = group_inds[j]
                group_size = len(inds)
                img_g = torch.index_select(img_x, 0, inds) # (m, 3, h, w) g: group
                context_g = torch.index_select(context, 0, inds)
                mask_g = torch.index_select(pred_mask, 0, inds) if pred_mask is not None else pred_mask # (m, 1, h, w)
                out_g, pred_param_g = self.executor.execute(img_g, op_ind - 3, mask_g, context_g, has_noise=False)
                pred_param_g = torch.cat([pred_param_g, torch.zeros(group_size, 24 - pred_param_g.shape[-1], dtype=torch.float).to(pred_param_g.device)], 1)
                out_gs.append(out_g)
                out_param_gs.append(pred_param_g)
            out_gs = torch.cat(out_gs)
            out_param_gs = torch.cat(out_param_gs)  # (bs, n)
            pred_img = torch.index_select(out_gs, 0, rev_inds) # recover order
            pred_param = torch.index_select(out_param_gs, 0, rev_inds)

            pred_imgs.append(pred_img)
            pred_params.append(pred_param)
            pred_masks.append(pred_mask)
            img_x = pred_img
            pred_ops.append(pred_op.squeeze(-1))  # require batch size = 1
        if len(pred_ops) == 0:
            pred_ops = []
            pred_imgs = img_x.unsqueeze(1)
            pred_masks = None
            state = {}
        else:
            pred_ops = torch.stack(pred_ops, 1)  # convert operator to vocab
            # pred_params = torch.stack(pred_params, 1) TODO: make params stackable
            pred_imgs = torch.stack(pred_imgs, 1)
            pred_masks = torch.stack(pred_masks, 1) if pred_mask is not None else None
            state = {
                'reqs': x,
                'imgs': pred_imgs.detach(),
                'ops': pred_ops,
                'param': pred_params,
                'hidden': decoder_hiddens,
                'masks': pred_masks
            }
        return state, pred_imgs, pred_ops, pred_params  # list

    def forward(self, x, img_x, hidden, op, mask_dict=None):
        """
        TODO: need a state to store history operations
        :param x:
        - req: (bs, len_x)
        - img: (bs, 3, h, w)
        - op: (bs,)
        - hidden: (n_rnnlayer, bs, hidden_size)
        - gt_img: (bs, 3, h, w)
        :return:
        """
        op = op.view(-1, 1)
        bs, step, _, _ = img_x.shape # step >=1
        with torch.no_grad():
            encoder_outputs, encoder_hidden, _ = self.lang_encoder(x) # (n_rnnlayer, 1, hidden_size), (bs, encoder_valid_len, hidden)
        # block local editing operation
        op_mask = torch.tensor([False, False, True, True, True, True, True, False, True, True, False], dtype=torch.float, device=img_x.device).repeat(bs, 1) # (bs, op_len)

        img_feat = self.vis_encoder(img_x)
        img_feat = F.relu(self.bn1(img_feat))
        # decoder_output (bs, len, n_vocab)
        op_logprob, decoder_hidden, step_attn, context = \
            self.decoder.forward_step(op, hidden, encoder_outputs, img_feat)
        # get entropy penalty
        entropy_penalty = self.get_entropy_penalty(op_logprob) # (bs, 1)
        # update operator probs and add explore prob
        _, _, op_len = op_logprob.shape
        op_probs = torch.exp(op_logprob).squeeze(1)  # (bs, 1, n_vocab)
        op_probs = op_probs * (1 - self.opt.explore_prob) + self.opt.explore_prob
        # add op_probs mask
        # all ops: brightness, contrast, color, saturation, tone inpaint white_op, sharpness
        # get operator repetitive penalty
        # repetitive_penalty = (hist_ops[:, :-1] == (ops.unsqueeze(1) + 3)).sum(1, keepdim=True).float()  # (bs, 1)
        op_probs = op_probs * op_mask
        op_probs = op_probs / (op_probs.sum(1, keepdim=True) + 1e-30)
        dist = torch.distributions.Categorical(probs=op_probs) # better initialize with logits
        pred_op = dist.sample().view(bs, -1)  # (bs, 1)
        # assign op_mask to 0 with predicted op
        for batch_i in range(bs):
            op_mask[batch_i, pred_op[batch_i, 0]] = 0
        pred_mask = self.get_gt_mask(img_x, mask_dict, pred_op.detach().cpu().numpy()) \
            if mask_dict is not None else mask_dict

        # divide group
        out_gs = []
        out_param_gs = []
        unqs, group_inds, rev_inds = self.divide_op_group(pred_op.view(-1))
        for j in range(len(group_inds)):
            op_ind = unqs[j].item()
            inds = group_inds[j]
            group_size = len(inds)
            img_g = torch.index_select(img_x, 0, inds)  # (m, 3, h, w) g: group
            context_g = torch.index_select(context, 0, inds)
            mask_g = torch.index_select(pred_mask, 0, inds) if pred_mask is not None else pred_mask # (m, 1, h, w)
            out_g, pred_param_g = self.executor.execute(img_g, op_ind - 3, mask_g, context_g, has_noise=False)
            pred_param_g = torch.cat([pred_param_g, torch.zeros(group_size, 24 - pred_param_g.shape[-1], dtype=torch.float).to(pred_param_g.device)], 1)
            out_gs.append(out_g)
            out_param_gs.append(pred_param_g)
        out_gs = torch.cat(out_gs)
        out_param_gs = torch.cat(out_param_gs)  # (bs, n)
        pred_img = torch.index_select(out_gs, 0, rev_inds) # recover order
        pred_param = torch.index_select(out_param_gs, 0, rev_inds)

        # pass rnn cell again to get the RNN feature
        img_feat = self.vis_encoder(pred_img)
        img_feat = F.relu(self.bn1(img_feat))
        _, _, _, next_context = self.decoder.forward_step(pred_op, decoder_hidden, encoder_outputs, img_feat)  # (bs, 1, n_cls)

        return pred_img, op_logprob, entropy_penalty, context, next_context  # outs include the gradient to the parameter

    def get_entropy_penalty(self, logprobs):
        """
        :param logprobs: (bs, d)
        :return: entropy_penalty: (bs, 1)
        """
        probs = torch.exp(logprobs)
        entropy = - (probs * logprobs).sum(1, keepdim=True)
        entropy_penalty = math.log(logprobs.shape[-1]) - entropy
        return entropy_penalty



