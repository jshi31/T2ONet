# wrap the language into index format.
import os
import json
import numpy as np
from core.utils.text_utils import parse_sent


def define_split(data_file, out_dir, session):
    save_path = os.path.join(out_dir, '{}_sess_{}.json'.format('train', session))
    if os.path.exists(save_path):
        print('Do nothing! split session {} already exists in {}'.format(session, save_path))
        return

    with open(data_file, 'r') as f:
        data = json.load(f)
    inputs = np.unique([v['input'] for v in data])
    ratio = {'train': 0.7, 'val': 0.1, 'test': 0.2}
    total_len = len(inputs)
    train_len = int(total_len * ratio['train'])
    val_len = int(total_len * ratio['val'])
    np.random.seed(0)
    np.random.shuffle(inputs)
    train_inputs = inputs[:train_len]
    val_inputs = inputs[train_len: train_len + val_len]
    test_inputs = inputs[train_len + val_len:]
    splits = [train_inputs, val_inputs, test_inputs]

    def get_split_data(split_inputs):
        split_data = []
        for dic in data:
            if dic['input'] in split_inputs:
                split_data.append(dic)
        return split_data

    for phase, split_input in zip(['train', 'val', 'test'], splits):
        split_data = get_split_data(split_input)
        save_path = os.path.join(out_dir, '{}_sess_{}.json'.format(phase, session))
        with open(save_path, 'w') as f:
            json.dump(split_data, f)
        print('saved {} split in {}'.format(phase, save_path))


def load_vocab(vocab_dir, session):
    """load vocabulary from files under vocab_dir"""
    with open(os.path.join(vocab_dir, 'FiveK_vocabs_sess_{}.json'.format(session))) as f:
        vocab = json.load(f)
    with open(os.path.join(vocab_dir, 'FiveK_operator_vocabs_sess_{}.json'.format(session))) as f:
        op_vocab = json.load(f)
    vocab2id = {token: i for i, token in enumerate(vocab)}
    id2vocab = {i: token for i, token in enumerate(vocab)}
    op_vocab2id = {token: i for i, token in enumerate(op_vocab)}
    id2op_vocab = {i: token for i, token in enumerate(op_vocab)}
    return vocab2id, id2vocab, op_vocab2id, id2op_vocab


def save_txt2idx(data_dir, vocab_dir, session, max_len):

    def token2idx(token):
        idx = vocab2id[token] if token in vocab2id else 3
        return idx

    vocab2id, _, _, _ = load_vocab(vocab_dir, session)
    phases = ['train', 'val', 'test']
    save_dir = os.path.join(data_dir, 'annotations')
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, '{}_sess_{}.json'.format('train', session))
    if os.path.exists(save_path):
        print('Do nothing! text to index transform already exists in {}'.format(save_path))
        return

    for phase in phases:
        path = os.path.join(data_dir, 'splits', '{}_sess_{}.json'.format(phase, session))
        save_path = os.path.join(save_dir, '{}_sess_{}.json'.format(phase, session))
        with open(path, 'r') as f:
            data = json.load(f)
        for dic in data:
            sent = dic['request']
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
            dic['request_idx'] = sent_idx
        with open(save_path, 'w') as f:
            json.dump(data, f)
            print('saved text to index transform in {}'.format(save_path))


if __name__ == '__main__':
    data_dir = 'data/FiveK'
    data_path = 'data/FiveK/FiveK.json'
    split_dir = 'data/FiveK/splits'
    vocab_dir = 'data/language'
    session = 1
    max_len = 15

    os.makedirs(split_dir, exist_ok=True)
    define_split(data_path, split_dir, session)

    save_txt2idx(data_dir, vocab_dir, session, max_len)

