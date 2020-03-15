import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb


class Embedding(nn.Embedding):
    """
    if fix embedding, it fix the word embedding, but just train the special token embedding.
    """
    def __init__(self, num_embeddings, embedding_dim, num_spec, fix_embedding=False, padding_idx=None,
                 max_norm=None, norm_type=2., scale_grad_by_freq=False,
                 sparse=False, _weight=None):
        super(Embedding, self).__init__(num_embeddings, embedding_dim, padding_idx,
                                        max_norm, norm_type, scale_grad_by_freq,
                                        sparse, _weight)
        self.fix_embedding = fix_embedding
        self.register_buffer('mask_spec', torch.cat([torch.ones(num_spec, embedding_dim, dtype=torch.float),
                                                     torch.zeros(num_embeddings - num_spec, embedding_dim, dtype=torch.float)]))
        self.register_buffer('mask_word', -self.mask_spec + 1)

    def forward(self, input):
        if self.fix_embedding:
            spec_embd = F.embedding(input, self.weight * self.mask_spec, self.padding_idx, self.max_norm,
                                    self.norm_type, self.scale_grad_by_freq, self.sparse)
            word_embd = F.embedding(input, self.weight.detach() * self.mask_word, self.padding_idx, self.max_norm,
                                    self.norm_type, self.scale_grad_by_freq, self.sparse)
            return spec_embd + word_embd
        else:
            return F.embedding(input, self.weight, self.padding_idx, self.max_norm,
                               self.norm_type, self.scale_grad_by_freq, self.sparse)


class RNNEncoder(nn.Module):
    def __init__(self, vocab_size, word_embedding_size, hidden_size, n_spec_token, bidirectional=False,
                 input_dropout_p=0, dropout_p=0, n_layers=1, pad_id=0, rnn_type='lstm', variable_lengths=True, word2vec=None,
                 fix_embedding=False):
        """
        Requires: the input language is padded with <START> token
        :param vocab_size: vocabulary size
        :param word_embedding_size: word embedding dimension
        :param hidden_size: hidden vector dimension
        :param bidirectional: if bi direction
        :param input_dropout_p: dropout after embedding layer
        :param dropout_p: dropout in rnn
        :param n_layers: number of layer in rnn
        :param rnn_type: the type of rnn
        :param variable_lengths: if True,
        :param word2vec:
        """
        super(RNNEncoder, self).__init__()
        self.variable_lengths = variable_lengths
        self.embedding = Embedding(vocab_size, word_embedding_size, n_spec_token, fix_embedding)
        if word2vec is not None:
            assert word2vec.shape[0] == vocab_size - n_spec_token, \
                'the size of vocabulary is {}, word2vec feature is {}, ' \
                'which does not satisfy #vocab - #word2vec == {}'. \
                    format(vocab_size, word2vec.shape[0], n_spec_token)
            spec_embedding = self.embedding.weight[:n_spec_token].clone()
            self.embedding.weight = nn.Parameter(torch.cat((spec_embedding, word2vec))).to(self.embedding.weight.device)
        self.input_dropout = nn.Dropout(input_dropout_p)
        self.pad_id = pad_id
        self.rnn_type = rnn_type
        self.rnn = getattr(nn, rnn_type.upper())(word_embedding_size, hidden_size, n_layers,
                                                 batch_first=True,
                                                 bidirectional=bidirectional,
                                                 dropout=dropout_p)
        self.num_dirs = 2 if bidirectional else 1

    def forward(self, input_labels):
        """
        Inputs:
        - input_labels: Variable long (batch, seq_len), must pad zero
        Outputs:
        - output  : Variable float (batch, max_len, hidden_size * num_dirs)
        - hidden  : Variable float (batch, num_layers * num_dirs * hidden_size)
        - embedded: Variable float (batch, max_len, word_vec_size)
        """
        if self.variable_lengths:
            input_lengths = (input_labels != self.pad_id).sum(1)  # Variable (batch, )

            sorted_input_lengths, sort_ixs = input_lengths.sort(descending=True)
            recover_ixs = sort_ixs.argsort()
            input_labels = input_labels[:, :input_lengths.max().item()]
            assert input_lengths.max().item() == input_labels.shape[1]

            # sort input_labels by descending order
            input_labels = input_labels[sort_ixs]

        # embed
        embedded = self.embedding(input_labels)  # (n, seq_len, word_embedding_size)
        embedded = self.input_dropout(embedded) # (n, seq_len, word_embedding_size)
        if self.variable_lengths:
            embedded = nn.utils.rnn.pack_padded_sequence(embedded, sorted_input_lengths, batch_first=True)
        # forward rnn
        output, hidden = self.rnn(embedded)

        # recover
        if self.variable_lengths:

            # recover rnn
            output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True) # (batch, max_len, hidden)
            output = output[recover_ixs]

            # recover hidden
            if self.rnn_type == 'lstm':
                hidden = (hidden[0][:, recover_ixs, :], hidden[1][:, recover_ixs, :])

            #     hidden = hidden[0]  # we only use hidden states for the final hidden representation
            # hidden = hidden[:, recover_ixs, :]  # (num_layers * num_dirs, batch, hidden_size)
            # hidden = hidden.transpose(0, 1).contiguous()  # (batch, num_layers * num_dirs, hidden_size)
            # hidden = hidden.view(hidden.shape[0], -1)  # (batch, num_layers * num_dirs * hidden_size)
        return output, hidden, embedded
