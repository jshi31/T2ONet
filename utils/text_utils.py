import os
import string
import json

import torch
import numpy as np
import h5py

def parse_sent(desc):
    """
    parse sentence into tokens and do cleaning
    :param desc: sentence
    :return: tokens
    """
    table = str.maketrans('', '', string.punctuation)
    # tokenize
    desc = desc.split()
    # convert to lower case
    desc = [word.lower() for word in desc]
    # remove punctuation from each token
    desc = [w.translate(table) for w in desc]
    # remove hanging 's' and 'a'
    desc = [word for word in desc if len(word) > 1]
    # remove tokens with numbers in them
    tokens = [word for word in desc if word.isalpha()]
    return tokens


def load_vocab(vocab_dir, dataset, session):
    """load vocabulary from files under vocab_dir"""
    with open(os.path.join(vocab_dir, '{}_vocabs_sess_{}.json'.format(dataset, session))) as f:
        vocab = json.load(f)
    with open(os.path.join(vocab_dir, '{}_operator_vocabs_sess_{}.json'.format(dataset, session))) as f:
        op_vocab = json.load(f)
    vocab2id = {token: i for i, token in enumerate(vocab)}
    id2vocab = {i: token for i, token in enumerate(vocab)}
    op_vocab2id = {token: i for i, token in enumerate(op_vocab)}
    id2op_vocab = {i: token for i, token in enumerate(op_vocab)}
    return vocab2id, id2vocab, op_vocab2id, id2op_vocab


def txt2idx(sent, vocab2id, max_len):
    """
    :param sent: string
    :param vocab2id: diction
    :param max_len: encoder max length, including start, end position.
    :return: sent_idx: idx of the sentence in max_len
    """
    def token2idx(token):
        idx = vocab2id[token] if token in vocab2id else 3
        return idx

    max_len = max_len - 2
    tokens = parse_sent(sent)
    valid_sent_idx = np.array([token2idx(token) for token in tokens])
    valid_len = len(valid_sent_idx)
    sent_idx = np.zeros(max_len, dtype=int)
    sent_idx[:min(valid_len, max_len)] = valid_sent_idx[:max_len]
    end = np.where(sent_idx == 0)[0]
    sent_idx = sent_idx.tolist()
    if len(end) > 0:
        sent_idx.insert(end[0], 2)
    else:
        sent_idx.append(2)
    sent_idx.insert(0, 1)
    sent_idx = torch.tensor(sent_idx).unsqueeze(0)
    return sent_idx


def load_embedding(path):
    data = h5py.File(path, 'r')
    glv = data['glove'][()]
    return torch.tensor(glv)


