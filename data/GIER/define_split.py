import os
import json
import pdb

import cv2
import numpy as np


def split_data(session, data, out_dir):
    id_len = len(data)
    pair_ids = np.arange(id_len)
    np.random.seed(0) # very important
    np.random.shuffle(pair_ids)

    # directly store the sperate datas
    train_pair_ids = pair_ids[:int(id_len*0.8)]
    val_pair_ids = pair_ids[int(id_len*0.8):int(id_len*0.9)]
    test_pair_ids = pair_ids[int(id_len*0.9):]

    train_data = [data[i] for i in train_pair_ids]
    val_data = [data[i] for i in val_pair_ids]
    test_data = [data[i] for i in test_pair_ids]
    for phase, split_data in zip(['train', 'val', 'test'], [train_data, val_data, test_data]):
        save_path = os.path.join(out_dir, '{}_sess_{}.json'.format(phase, session))
        with open(save_path, 'w') as f:
            json.dump(split_data, f)
        print('saved {} split in {}'.format(phase, save_path))
    return train_data, val_data, test_data


def check_image_size_single(datas, phase, img_dir, save_dir, session):
    idx = []
    for i, data in enumerate(datas):
        input = data['input']
        output = data['output']
        input_path = os.path.join(img_dir, input)
        output_path = os.path.join(img_dir, output)
        input_img = cv2.imread(input_path)
        output_img = cv2.imread(output_path)
        if input_img.shape[0] == output_img.shape[0] and input_img.shape[1] == output_img.shape[1]:
            idx.append(i)
        if i % 200 == 0:
            print('{}/{}'.format(i, len(datas)))
    with open(os.path.join(save_dir, '{}_shapeAlign_sess_{}.json'.format(phase, session)), 'w') as f:
        json.dump(idx, f)
    print('dumped idx to {}'.format(os.path.join(save_dir, '{}_shapeAlign_sess_{}.json'.format(phase, session))))
    return idx


def check_image_size(train, val, test, img_dir, save_dir, session):
    for data, phase in zip([train, val, test], ['train', 'val', 'test']):
        check_image_size_single(data, phase, img_dir, save_dir, session)

def check_img_non_crop_single(phase, save_dir, session):
    non_crop_list = []
    with open(os.path.join(save_dir, '{}_sess_{}.json'.format(phase, session))) as f:
        datas = json.load(f)
    for i, data in enumerate(datas):
        if 'crop' not in data['operator']:
            non_crop_list.append(i)
    print('len of all data {}, len of non crop data {}'.format(len(datas), len(non_crop_list)))
    return non_crop_list


def check_img_non_crop(save_dir, session):
    non_crop_list_train = check_img_non_crop_single('train', save_dir, session)
    non_crop_list_val = check_img_non_crop_single('val', save_dir, session)
    non_crop_list_test = check_img_non_crop_single('test', save_dir, session)
    return non_crop_list_train, non_crop_list_val, non_crop_list_test


def check_img_with_L1_single(phase, save_dir, session):

    with open(os.path.join(save_dir, '{}_shapeAlign_sess_{}.json'.format(phase, session))) as f:
        align_idx = json.load(f)
    with open(os.path.join(save_dir, '{}_sess_{}.json'.format(phase, session))) as f:
        data = json.load(f)

    full_names = [data[i]['input'].split('_')[0] for i in range(len(data))]
    shape_align_names = [full_names[i] for i in align_idx]

    with open(os.path.join(save_dir, 'Ids_L1Thr_0.06.json')) as f:
        valid_names = json.load(f)

    valid_names = [name for name in valid_names if name in full_names]

    non_crop_list = check_img_non_crop_single(phase, save_dir, session)
    non_crop_names = [full_names[i] for i in non_crop_list]

    # same shape and same
    strange_list = []
    for name in valid_names:
        if name not in shape_align_names:
            strange_list.append(name)
            print('exist image with different shape but very similar, check {}'.format(name))

    print('totally {} images exists in valid but do not have equal image shape'.format(len(strange_list)))
    print('{} valid/all: {:.2f}'.format(phase, len(valid_names)/len(full_names)))
    print('{} align/all: {:.2f}'.format(phase, len(shape_align_names)/len(full_names)))


def check_img_with_L1(save_dir, session):
    for phase in ['train', 'val', 'test']:
        check_img_with_L1_single(phase, save_dir, session)


def check_img_non_crop_align_single(phase, save_dir, session):
    with open(os.path.join(save_dir, '{}_shapeAlign_sess_{}.json'.format(phase, session))) as f:
        align_idx = json.load(f)
    with open(os.path.join(save_dir, '{}_sess_{}.json'.format(phase, session))) as f:
        data = json.load(f)
    full_names = [data[i]['input'].split('_')[0] for i in range(len(data))]
    align_names = [full_names[i] for i in align_idx]

    non_crop_list = check_img_non_crop_single(phase, save_dir, session)
    non_crop_names = [full_names[i] for i in non_crop_list]

    names = list(set(align_names).union((set(non_crop_names) - set(align_names))))
    names_idx = sorted([full_names.index(name) for name in names])
    with open(os.path.join(save_dir, '{}_shapeAlignNonCrop_sess_{}.json'.format(phase, session)), 'w') as f:
        json.dump(names_idx, f)
    print('the shape align non crop saved at {}'.format(os.path.join(save_dir, '{}_shapeAlignNonCrop_sess_{}.json'.format(phase, session))))


def check_img_non_crop_align(save_dir, session):
    for phase in ['train', 'val', 'test']:
        check_img_non_crop_align_single(phase, save_dir, session)


def split_L1_valid_single(phase, save_dir, session):
    with open(os.path.join(save_dir, 'Ids_L1Thr_0.06.json')) as f:
        valid_names = json.load(f)

    with open(os.path.join(save_dir, '{}_sess_{}.json'.format(phase, session))) as f:
        data = json.load(f)

    full_names = [data[i]['input'].split('_')[0] for i in range(len(data))]
    valid_names = [name for name in valid_names if name in full_names]

    valid_idx = sorted([full_names.index(valid_name) for valid_name in valid_names])
    with open(os.path.join(save_dir, '{}_Ids_L1Thr_0.06_sess_{}.json'.format(phase, session)), 'w') as f:
        json.dump(valid_idx, f)
    print('split valid data idx save to {}'.format(os.path.join(save_dir, '{}_Ids_L1Thr_0.06_sess_{}.json'.format(phase, session))))


def split_L1_valid(save_dir, session):
    for phase in ['train', 'val', 'test']:
        split_L1_valid_single(phase, save_dir, session)


def check_global_single(phase, save_dir, session):
    with open(os.path.join(save_dir, '{}_sess_{}.json'.format(phase, session))) as f:
        datas = json.load(f)
    global_idx = []
    for i, data in enumerate(datas):
        if 'inpaint_obj' in data['operator'] or 'color_bg' in data['operator']:
            continue
        global_idx.append(i)
    with open(os.path.join(save_dir, '{}_global_sess_{}.json'.format(phase, session)), 'w') as f:
        json.dump(global_idx, f)

def check_global(save_dir, session):
    for phase in ['train', 'val', 'test']:
        check_global_single(phase, save_dir, session)


if __name__ == '__main__':
    session = 3
    json_path = 'data/GIER/GIER.json'
    out_dir = 'data/GIER/splits'
    img_dir = 'data/GIER/images'
    os.makedirs(out_dir, exist_ok=True)
    with open(json_path, 'r') as f:
        data = json.load(f)

    train_data, val_data, test_data = split_data(session, data, out_dir)
    # check_image_size(train_data, val_data, test_data, img_dir, out_dir, session)

    # check_img_non_crop(out_dir, session)

    # check_img_with_L1(out_dir, session)

    # check_img_non_crop_align(out_dir, session)

    # split_L1_valid(out_dir, session)

    check_global(out_dir, session)
