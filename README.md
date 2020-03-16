# Learning by Planning: Language-Guided Global Image Editing

## Instroduction
This is the Pytorch implementation for paper "Learning by Planning: Language-Guided Global Image Editing".

## Dependency

- Pytorch >= 1.0.0
- opencv-python 
- panopticapi
- easydict
- tensorboardX
- tensorflow<2.0.0
- tabulate
- dominate
- kornia



## Installation

- Clone this repo

  ```shell
  git clone https://github.com/jshi31/T2ONet.git --recursive
  ```

- Install the submodule `pyutils/edgeconnect` according to its [README](https://github.com/jshi31/edge-connect/tree/1f2658e3b190de47b86b9e25ff39227ed90d5f26).

  The critical thing is to download pre-trained model.

## Dataset

### MIT-Adobe Five

Test the dataloader by running

```shell
PYTHONPATH='.' python datasets/FiveKdataset.py
```



## Plan Action Sequences

```shell
PYTHONPATH='.' CUDA_VISIBLE_DEVICES=0 python preprocess/gen_greedy_seqs_FiveK.py
```



## T2ONet

### FiveK Train

```shell
PYTHONPATH='.' CUDA_VISIBLE_DEVICES=0 python experiments/t2onet/train_seq2seqL1.py --batch_size 64 --print_every 50 --checkpoint_every 1000 --num_iter 10000 --trial 1
```

### FiveK Test

```shell
CUDA_VISIBLE_DEVICES=0 python core/test_seq2seqL1.py --print_every 100 --visualize_every 100  --visualize 0 --is_train 0 --trial 1 
```

### GIER Train

```shell
CUDA_VISIBLE_DEVICES=0 python core/train_GIER_seq2seqL1.py  --dataset GIER --session 3 --batch_size 64 --data_mode global+shapeAlign --print_every 100 --checkpoint_every 1000 --num_iter 20000 --trial 1 
```

### GIER Test

```shell
CUDA_VISIBLE_DEVICES=0 python core/test_GIER_seq2seqL1.py  --dataset GIER --session 3 --data_mode global+shapeAlign --print_every 20 --visualize_every 5 --visualize 1 --trial 6
```
