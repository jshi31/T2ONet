import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import Attention


class Decoder(nn.Module):
    """Decoder RNN module to get the output
    """
    def __init__(self, vocab_size, max_len, word_vec_dim, hidden_size,
                 n_layers, rnn_type='lstm',
                 bidirectional=False, input_dropout_p=0,
                 dropout_p=0, use_attention=False):
        super(Decoder, self).__init__()
        self.max_length = max_len
        self.output_size = vocab_size
        self.hidden_size = hidden_size
        self.word_vec_dim = word_vec_dim
        self.bidirectional_encoder = bidirectional
        if bidirectional:
            self.hidden_size *= 2
        self.use_attention = use_attention

        self.embedding = nn.Embedding(self.output_size, self.word_vec_dim)
        self.rnn = getattr(nn, rnn_type.upper())(self.word_vec_dim + self.hidden_size, self.hidden_size, n_layers,
                                 batch_first=True, dropout=dropout_p)

        self.out_linear = nn.Linear(self.hidden_size, self.output_size) # TODO: might be self.output_size - 3
        self.vis_linear = nn.Linear(self.hidden_size, self.hidden_size)
        if use_attention:
            self.attention = Attention(self.hidden_size)
        # get image embedding
        self.input_dropout = nn.Dropout(p=input_dropout_p)


    def forward_step(self, input_var, hidden, encoder_outputs, img_feat):
        """
        single forward pass to predict next operator
        :param input_var: (bs, 1)
        :param hidden: tuple for LSTM, tensor for GRU. (n_rnnlayer, 1, hidden_size)
        :param encoder_outputs: (bs, encoder_valid_len, hidden) encoder_valid_len: end up at 2
        :param img_feat: (bs, d)
        :return predicted_softamx: (bs, 1, n_cls)
        :return hidden: tuple for LSTM, tensor for GRU. (bs*n_rnnlayer, 1, hidden_size)
        :return attn: (bs, 1, encoder_valid_len)
        :return context: (bs, hidden_size)
        """
        batch_size = input_var.size(0)
        output_size = input_var.size(1)
        vis_feat = F.relu(self.vis_linear(img_feat))

        embedded = self.embedding(input_var)
        embedded = torch.cat((embedded, vis_feat.view(batch_size, 1, -1)), 2)
        embedded = self.input_dropout(embedded)
        context, hidden = self.rnn(embedded, hidden)

        attn = None
        if self.use_attention:
            context, attn = self.attention(context, encoder_outputs)
        output = self.out_linear(context.contiguous().view(-1, self.hidden_size))
        predicted_softmax = F.log_softmax(output.view(batch_size, output_size, -1), -1) # (bs, seq_len=1, n_cls)
        return predicted_softmax, hidden, attn, context.squeeze(1)

    def _init_state(self, encoder_hidden):
        if encoder_hidden is None:
            return None
        if isinstance(encoder_hidden, tuple):
            encoder_hidden = tuple([self._cat_directions(h) for h in encoder_hidden])
        else:
            encoder_hidden = self._cat_directions(encoder_hidden)
        return encoder_hidden

    def _cat_directions(self, h):
        if self.bidirectional_encoder:
            h = torch.cat([h[0:h.size(0):2], h[1:h.size(0):2]], 2)
        return h
