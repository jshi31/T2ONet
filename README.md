# Learning by Planning: Language-Guided Global Image Editing

## Instroduction
This is the Pytorch implementation for paper "Learning by Planning: Language-Guided Global Image Editing".

## Dependency

- Pytorch >= 1.0.0
- opencv-python 
- panopticapi
- pycocotools
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

All the working directory for the following commands are project root.

### MIT-Adobe FiveKReq

- Download the fiveK image

  ```shell
  aws s3 cp s3://cil-ldie/LDIE/data/FiveK/images data/FiveK/images --recursive
  ```

- Test the dataloader by running

```shell
PYTHONPATH='.' python datasets/FiveKdataset.py
```

### GIER

- Download the GIER images into `data/GIER/images`

  ```shell
  aws s3 cp s3://cil-ldie/LDIE/data/GIER/images data/GIER/images --recursive
  ```

- Download the GIER mask into `data/GIER/masks`

  ```shell
  aws s3 cp s3://cil-ldie/LDIE/data/GIER/masks data/GIER/masks --recursive
  ```

- Download the GIER feature to `data/GIER/features`

  ```shell
  aws s3 cp s3://cil-ldie/LDIE/data/GIER/features data/GIER/features --recursive
  ```

  

Test the data loader by running 

```shell
PYTHONPATH='.' python datasets/GIERdataset.py
```



## Plan Action Sequences

#### FiveK

Generate action sequence using operation planning

```shell
PYTHONPATH='.' CUDA_VISIBLE_DEVICES=0 python preprocess/gen_greedy_seqs_FiveK.py
```

Or download the sequence 

```shell
aws s3 cp s3://cil-ldie/LDIE/output/actions_set_1 output/actions_set_1 --recursive
```

#### GIER

Generate action sequence using operation planning (**currenty has error**)

```shell
PYTHONPATH='.' CUDA_VISIBLE_DEVICES=0 python preprocess/gen_greedy_seqs_GIER.py
```

Or download the sequence

```shell
aws s3 cp s3://cil-ldie/LDIE/output/GIER_actions_set_1 output/GIER_actions_set_1 --recursive
```

## T2ONet

### FiveK Train

```shell
PYTHONPATH='.' CUDA_VISIBLE_DEVICES=0 python experiments/t2onet/train_seq2seqL1.py --batch_size 64 --print_every 50 --checkpoint_every 1000 --num_iter 10000 --trial 2
```

### FiveK Test

```shell
PYTHONPATH='.' CUDA_VISIBLE_DEVICES=0 python experiments/t2onet/test_seq2seqL1.py --print_every 100 --visualize_every 100  --visualize 0 --is_train 0 --trial 1 
```

select the trial number indicates which model you will use. To test our provided model, first download it by

```shell
aws s3 cp s3://cil-ldie/LDIE/output/FiveK_trial_1/seq2seqL1_model output/FiveK_trial_1/seq2seqL1_model --recursive
```

and set the trial argument as 1 in the testing model.

To visualize the result, set visualize argument as 1, and the result will be in `FiveK_trial_1/test/web`

### GIER Train

```shell
PYTHONPATH='.' CUDA_VISIBLE_DEVICES=0 python experiments/t2onet/train_GIER_seq2seqL1.py  --dataset GIER --session 3 --batch_size 64 --data_mode global+shapeAlign --print_every 100 --checkpoint_every 1000 --num_iter 20000 --trial 1 
```

### GIER Test

```shell
PYTHONPATH='.' CUDA_VISIBLE_DEVICES=0 python experiments/t2onet/test_GIER_seq2seqL1.py  --dataset GIER --session 3 --data_mode global+shapeAlign --print_every 20 --visualize_every 5 --visualize 0 --trial 7
```

To test our provided model, first download it by

```shell
aws s3 cp s3://cil-ldie/LDIE/output/GIER_trial_7/seq2seqL1_model output/GIER_trial_7/seq2seqL1_model --recursive
```

and set the trial argument as 7 in the testing model.

### 