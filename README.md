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


## FiveKReq

### Build Vocabulary

```shell
python prepro/build_vocab.py --session 1 --dataset FiveK
```

### Data Preprocessing

Data size=4950 x 5

get the split and store the annotation (already in index format)

```shell
python data/FiveK/FiveK.py
```

Train:val:test = 7:1:2 = 17325:2475:4950

The annoation is stored at `data/FiveK/annotations`

Short edge 600.

### Generate greedy actions

```shell
CUDA_VISIBLE_DEVICES=0 python prepro/gen_greedy_seqs.py
```

- set 1: Nelder-Mead with full operation

  train init dist 0.1201 dist 0.0136

- set 2: lbfgs lr=1

  train init dist 0.1201 dist 0.0202

- set 3: Nelder-Mead only use brightness operation

  train init dist 0.1201 dist 0.0521

- set 4: Nelder-Mead only use brightness, contrast, saturation, sharpness. (GPU1)

  train init dist 0.1201 dist 0.0358

- set 5: Nelder-Mead only use fixed sequence order: brightness, contrast, saturation, sharpness, color, tone

  train init dist 0.1202 dist 0.0198 

  *conclusion* set1 and set5 show no much difference about the order
  
- Set 6: eps greedy

  train init dist 0.1201 dist 0.0197
  
- set 7: test with color op

  train init dist 0.1201 dist 0.0173

- set 8: test with tone op 

  train init dist 0.1201 dist 0.0277

- set 9: test with contrast op

  train init dist 0.1201 dist 0.0859

- set 10: test with saturation op 

  train init dist 0.1201 dist 0.1037

- set 11: test with sharpness op 

  train init dist 0.1201 dist 0.1163

```shell
CUDA_VISIBLE_DEVICES=0 python prepro/gen_greedy_seqs_GIER.py
```

- set 1: full list (inpainting, rm_bg)

  whole set: init dist 0.1073 dist 0.0411

### Train

```shell
CUDA_VISIBLE_DEVICES=0 python core/train_actor_fs.py --batch_size 64 --print_every 100 --checkpoint_every 2000 --num_iter 20000 --trial 1
```

- trial 1: purely use all the operation does not quite work.

  The L1 dist increase more than the input ...

   Possible reason:

  - the beam searched result is not alwasy good.
  - the parameter for color and tone can be rescaled, which is not good. Now I just rescaled them in [-1, 1]
  - the other operation might fall into local optimal, or the beam search algorithm contain problems. (why at the begining the parameter is high?)

- trial 2: just use brightness operation as input.

  Test init L1 dist 0.1190; L1 dist 0.0893

- trial 3: the same with trial 1, but fix the training bug.

  Test init L1 dist 0.1190; L1 dist 0.0898. http://10.9.226.94:8113/

- trial 4: use brightness, contrast, saturation, sharpness as input.

  Test init L1 dist 0.1190; L1 dist 0.0860

  input SSIM 0.7992, output SSIM 0.8355
  input FID 12.3714, output FID 10.4180

- trial 5: all operations, set1

  input L1 dist 0.1190, output L1 dist 0.0949
  input SSIM 0.7992, output SSIM 0.8300
  input FID 12.3714, output FID 8.2482

  avg var: 0.000532

- trial 6: fixed operation, set5. 

  Test init L1 dist 0.1190; L1 dist 0.0853

### Test

```shell
CUDA_VISIBLE_DEVICES=0 python core/test_actor_fs.py --print_every 100 --visualize_every 100 --trial 3
```

## Seq2seq+Pix2pixHD

### Train

```shell
CUDA_VISIBLE_DEVICES=0 python core/train_seq2seqGAN.py --batch_size 64 --print_every 50 --checkpoint_every 1000 --num_iter 10000 --trial 1 
```

- Trial 0: add the pix2pixHD discriminator 10k epoch

  http://10.9.226.94:8115/

- Trial 1: add the pix2pixHD discriminator

  input L1 dist 0.1190, output L1 dist 0.0801
  input SSIM 0.7992, output SSIM 0.8464
  input FID 12.3714, output FID 6.9436
  
  avg var: 0.005671
  
- Trial 2: add the pix2pixHD discriminator 20k iteration

  Test init L1 dist 0.1190; L1 dist 0.0830

- Trial 3: add the pix2pixHD discriminator without vgg loss

  Test init L1 dist 0.1190; L1 dist 0.0836

- Trial 4: add the pix2pixHD discriminator no vgg loss, no GANfeat loss 

  Test init L1 dist 0.1190; L1 dist 0.0922

  *conclusion*: purely GAN loss does not have effect. The effect comes from the vgg loss and Dfeat.

- Trial 5: add the pix2pixHD discriminator with vanila GAN loss

  **Problem**: VGG loss is increasing. Doult that it is because the training collaps.

  Test init L1 dist 0.1184; L1 dist 0.0956

- Trial 6: same with trial 1

  Test init L1 dist 0.1190; L1 dist 0.0860 (**large variance**)

- Trial 7: add the pix2pixHD discriminator with only vanila GAN loss 

  Test init L1 dist 0.1190; L1 dist 0.0847

  *conclusion*: purely vanila GAN loss can help.

- 

### Test

```shell
CUDA_VISIBLE_DEVICES=0 python core/test_seq2seqGAN.py --print_every 100 --visualize_every 100  --visualize 0 --is_train 0 --trial 1 
```

## Seq2seq Adapt GAN

Enable the Discriminator to apply on both pseudo target and gt target.

### Train

```shell
CUDA_VISIBLE_DEVICES=0 python core/train_seq2seqAdaptGAN.py --batch_size 64 --print_every 50 --checkpoint_every 1000 --num_iter 10000 --trial 1 
```

- Trial 1: default Discriminator

  Test init L1 dist 0.1190; L1 dist 0.0842

- Trial 2: pure LS GAN 

  Test init L1 dist 0.1190; L1 dist 0.0877

- Trial 3: pure vanila GAN

  Test init L1 dist 0.1190; L1 dist 0.0885

### Test

```shell
CUDA_VISIBLE_DEVICES=0 python core/test_seq2seqAdaptGAN.py --print_every 100 --visualize_every 100  --visualize 0 --is_train 0 --trial 1 
```



## Seq2Seq+L1

### Train

```shell
CUDA_VISIBLE_DEVICES=0 python core/train_seq2seqL1.py --batch_size 64 --print_every 50 --checkpoint_every 1000 --num_iter 10000 --trial 1
```

- Trial 1:

  Train init L1 dist 0.1202; L1 dist 0.0708

  Test init L1 dist 0.1190; L1 dist 0.0770

  input L1 dist 0.1190, output L1 dist 0.0770
  input SSIM 0.7992, output SSIM 0.8530
  input FID 12.3714, output FID 6.2376

  avg var: 0.006112

  Proving the result is not overfitting.

- Trial 2: train with longer time

  init L1 dist 0.1190; L1 dist 0.0814

- Trial 3: train with fixed action order over full operations

  add `--action_id 5` in the train command

  Test init L1 dist 0.1184; L1 dist 0.0832

- Trial 4: train with only brightness

  add `--action_id 3` in the train command

  Test init L1 dist 0.1184; L1 dist 0.1345

- Trial 5: train with four operations

  add `--action_id 4` in the train command

  Test init L1 dist 0.1184; L1 dist 0.0857
  
- Trial 10: train with epislon greedy 5%. (GPU1)

  add `--action_id 6` in the train command
  
  input L1 dist 0.1190, output L1 dist 0.0853
  input SSIM 0.7992, output SSIM 0.8452
  input FID 12.3714, output FID 6.7967
  
- Trial 18: train with only color op. (GPU0)

  add  `--action_id 7`

  input L1 dist 0.1190, output L1 dist 0.1129
  input SSIM 0.7992, output SSIM 0.8208
  input FID 12.3714, output FID 8.8494

  avg var: 0.011123

- Trial 19: train with only tone op (GPU1)

  add `--action_id 8`

  input L1 dist 0.1190, output L1 dist 0.1006
  input SSIM 0.7992, output SSIM 0.7775
  input FID 12.3714, output FID 10.5067

- Trial 20: train with only contrast op (GPU2)

  add `--action_id 9`

  input L1 dist 0.1190, output L1 dist 0.1178
  input SSIM 0.7992, output SSIM 0.7715
  input FID 12.3714, output FID 16.6828

- Trial 21: train with only saturation op (GPU4)

  add `--action_id 10`

  input L1 dist 0.1190, output L1 dist 0.1163
  input SSIM 0.7992, output SSIM 0.7999
  input FID 12.3714, output FID 13.3602

- Trial 22: train with only sharpness op (GPU5)

  add `--action_id 11`

  input L1 dist 0.1190, output L1 dist 0.1256
  input SSIM 0.7992, output SSIM 0.6765
  input FID 12.3714, output FID 27.3662

- Trail 23: train without attention on full list (GPU6)

  add `--use_attention`

  input L1 dist 0.1190, output L1 dist 0.0801
  input SSIM 0.7992, output SSIM 0.8450
  input FID 12.3714, output FID 6.3990

  0.006870

### Test

```shell
CUDA_VISIBLE_DEVICES=0 python core/test_seq2seqL1.py --print_every 100 --visualize_every 100  --visualize 0 --is_train 0 --trial 1 
```

**GIER**

### Train

```shell
CUDA_VISIBLE_DEVICES=0 python core/train_GIER_seq2seqL1.py  --dataset GIER --session 3 --batch_size 64 --data_mode global+shapeAlign --print_every 100 --checkpoint_every 1000 --num_iter 20000 --trial 1 
```

- Trial 5: `--data_mode shapeAlign`

- Trial 6:  `--data_mode global+shapeAlign`.

  nput L1 dist 0.1079, output L1 dist 0.1028
  input SSIM 0.8048, output SSIM 0.8118
  input FID 49.6229, output FID 48.7880

  avg var: 0.000721

- Trial 7: (GPU5) `--data_mode global+shapeAlign`.

  input L1 dist 0.1079, output L1 dist 0.1000
  input SSIM 0.8048, output SSIM 0.8179
  input FID 49.6229, output FID 48.7796

  avg var: 0.000228

  iter 9k

  input L1 dist 0.1079, output L1 dist 0.1091
  input SSIM 0.8048, output SSIM 0.7980
  input FID 49.6229, output FID 49.4269

   avg var: 0.002443

### Test

```shell
CUDA_VISIBLE_DEVICES=0 python core/test_GIER_seq2seqL1.py  --dataset GIER --session 3 --data_mode global+shapeAlign --print_every 20 --visualize_every 5 --visualize 1 --trial 6
```



## Pix2pixHD

It requires the image size to be **dividable by 16 !**

### Train

```shell
CUDA_VISIBLE_DEVICES=0 python core/train_pix2pixHD.py --batch_size 64 --print_every 50 --checkpoint_every 1000 --num_iter 20000 --trial 1
```

- Trial 1: default setting

  Test init L1 dist 0.1190; L1 dist 0.0921 (cannot distinguish different language input)

  http://10.9.226.94:8116/
  
- Trial 2: triple: real image + neg txt. add flag, batch size is limited to 48.  `--triple`  

  Test init L1 dist 0.1190; L1 dist 0.0944

  Even if I use triplet, **it still cannot distinguish different language input.**

- Trial 3: Only train Langauge encoder in G, not in D.  bs 48

  Test init L1 dist 0.1190; L1 dist 0.0915 (wait to test langauge differentiability)

  **it still cannot distinguish different language input, so only train LSTM in G is not enough!** 

  1. modify it with two encoder and decoder.

- Trial 4: Only train Langauge encoder in G, not in D + triple loss bs 48

  Test init L1 dist 0.1190; L1 dist 0.0937

  **it still cannot distinguish different language input**

- Trial 5: Only train Language encoder in G, not in D, no lsgan `--no_lsgan` 

  Test init L1 dist 0.1190; L1 dist 0.0985

  **it still cannot distinguish different language input**

- Trial 6: train with language encoder totally separate in G and D (cond_c 512)

  not dependent on language

- Trial 7: train with language encoder totally separate in G and D (cond_c 128)

  `--cond_nc 128`

  not dependent on language

- Trial 8: train with language encoder totally separate in G and D (cond_c 32) 

  `--cond_nc 32`

  not dependent on langauge

### Test

```shell
CUDA_VISIBLE_DEVICES=0 python core/test_pix2pixHD.py --print_every 100 --visualize_every 100  --visualize 0 --is_train 0 --trial 1 
```

### Demo

```shell
CUDA_VISIBLE_DEVICES=0 python core/demo_pix2pixHD.py --no_lsgan --no_vgg_loss --no_ganFeat_loss --trial 2
```



## Bilinear GAN

### Train

```shell
python core/train_bilinearGAN.py --fusing_method lowrank_BP --batch_size 64 --print_every 10 --checkpoint_every 2000 --num_iter 20000 --gpu_ids 0,1 --trial 15 
```

- Trial 1- >15: val init L1 dist 0.1233; L1 dist 0.2184

  inference init L1 dist 0.1190; L1 dist 0.2211 **Wrong code **

- Trial 16: (**but cannot be trained or  reproduceable**)

- Trial 17: GPU1

  input L1 dist 0.1191, output L1 dist 0.1559
  input SSIM 0.7958, output SSIM 0.4988
  input FID 12.5824, output FID 102.1330

  var 0.008031

### Test

```shell
CUDA_VISIBLE_DEVICES=0 python core/test_bilinearGAN.py --fusing_method lowrank_BP --print_every 100 --visualize_every 100 --visualize 1 --trial 15
```

GIER

### Train

```shell
python core/train_GIER_bilinearGAN.py --fusing_method lowrank_BP --dataset GIER --session 3 --data_mode global+shapeAlign --batch_size 64 --print_every 100 --checkpoint_every 2000 --num_iter 20000 --gpu_ids 0,1 --trial 8
```

- Trial 8

### Test

```shell
CUDA_VISIBLE_DEVICES=0 python core/test_GIER_bilinearGAN.py --fusing_method lowrank_BP --dataset GIER --session 3 --data_mode global+shapeAlign --print_every 100 --visualize_every 5 --visualize 1 --trial 8
```

input L1 dist 0.1081, output L1 dist 0.1918
input SSIM 0.8002, output SSIM 0.4395
input FID 50.2319, output FID 214.7331

avg var: 0.012164

## DDPG

### Train

### Test

## HaiWang

### Train

Modify the` pix2pix_bucket_5`, which is the filterBank model

```shell
CUDA_VISIBLE_DEVICES=0 python core/train_HaiWang.py --which_model_netG unet_128  --num_G 5 --no_lsgan --triple 0 --model FilterBank --batch_size 64  --num_workers 8 --print_every 100 --checkpoint_every 2000 --num_iter 20000 --trial 1
```

- Trial 1: bankfilter default with triplet

  Test: init L1 dist 0.1190; L1 dist 0.1480

- Trial 2: bankfilter default without triplet 

  Test: init L1 dist 0.1190; L1 dist 0.1869

```shell
CUDA_VISIBLE_DEVICES=0 python core/train_HaiWang.py --which_model_netG unet_128  --num_G 5 --no_lsgan --triple 0 --model Pix2Pix --batch_size 64 --num_workers 8 --print_every 100 --checkpoint_every 2000 --num_iter 20000 --trial 1
```

- Trial 3: pix2pix with triplet

  Test: init L1 dist 0.1190; L1 dist 0.1223

  Has effect over different language input

- Trial 4: pix2pix without triplet 

  input L1 dist 0.1199, output L1 dist 0.1114
  input SSIM 0.7695, output SSIM 0.7299
  input FID 14.5154, output FID 49.2135

  avg var: a

  Also has effect over different language input

Conclusion: this method can only improve the result a little bit

```shell
CUDA_VISIBLE_DEVICES=0 python core/train_HaiWang.py --which_model_netG resnet_7blocks  --num_G 5 --no_lsgan --triple 0 --model Pix2Pix --batch_size 64 --num_workers 8 --print_every 100 --checkpoint_every 2000 --num_iter 20000 --trial 1
```

- Trial 5: pix2pix without triplet, use resnet_7blocks.

  input L1 dist 0.1199, output L1 dist 0.0928
  input SSIM 0.7695, output SSIM 0.7938
  input FID 14.5154, output FID 14.5538

  avg var: 0.005401

And the key point is that both generator and discriminator should have the language it self. They should not use common language encoder.

### Test

```shell
CUDA_VISIBLE_DEVICES=0 python core/test_HaiWang.py --which_model_netG unet_128  --model FilterBank --num_G 5 --print_every 100 --visualize_every 100  --visualize 0 --trial 1
```



```shell
CUDA_VISIBLE_DEVICES=0 python core/test_HaiWang.py --which_model_netG unet_128  --model Pix2Pix --num_G 5 --print_every 100 --visualize_every 100  --visualize 0 --trial 1
```

```shell
CUDA_VISIBLE_DEVICES=0 python core/test_HaiWang.py --which_model_netG resnet_7blocks  --model Pix2Pix --num_G 5 --print_every 100 --visualize_every 100  --visualize 0 --trial 1
```



### Demo

```shell
CUDA_VISIBLE_DEVICES=0 python core/demo_HaiWang.py --which_model_netG unet_128  --model Pix2Pix --num_G 5 --trial 3
```

**For GIER dataset**

### Train

```shell
CUDA_VISIBLE_DEVICES=0 python core/train_GIER_HaiWang.py --which_model_netG resnet_7blocks --dataset GIER --session 3  --num_G 5 --no_lsgan --triple 0 --model Pix2Pix --batch_size 64 --num_workers 8 --print_every 100 --checkpoint_every 2000 --num_iter 20000 --trial 1
```

- Trial 1:  GPU1

  data mode shapeAlign

  input L1 dist 0.0939, output L1 dist 0.1033
  input SSIM 0.8077, output SSIM 0.7650
  input FID 43.1350, output FID 55.1558

  avg var: 0.006509

- Trial 2: use valid sample

  add `--data_mode valid`

  input L1 dist 0.0939, output L1 dist 0.1053
  input SSIM 0.8077, output SSIM 0.7513
  input FID 43.1350, output FID 57.7457

  avg var: 0.002683

- Trial 3: use global+shape align, add `--data_mode global+shapeAlign` (GPU1)

  input L1 dist 0.1079, output L1 dist 0.1255
  input SSIM 0.7760, output SSIM 0.7293
  input FID 52.9235, output FID 74.7761
  inference init L1 dist 0.1079; L1 dist 0.1255 **??why different init distance?? because of the reshape**

  avg var: 0.026838

- Trial 4: `--which_model_netG unet_128 ` add `--data_mode global+shapeAlign` (GPU2)

  input L1 dist 0.1079, output L1 dist 0.1219
  input SSIM 0.7760, output SSIM 0.7123
  input FID 52.9235, output FID 132.0352
  inference init L1 dist 0.1079; L1 dist 0.1219

  avg var: 0.000168

### Test

```shell
CUDA_VISIBLE_DEVICES=0 python core/test_GIER_HaiWang.py --which_model_netG resnet_7blocks  --dataset GIER --session 3 --model Pix2Pix --num_G 5 --print_every 100 --visualize_every 100  --visualize 0 --trial 1
```



## Self-discriminator

### Train

```shell
CUDA_VISIBLE_DEVICES=0 python core/train_self_discriminator.py --checkpoint_every 5000  --shuffle 1 --print_every 50 --use_attention 1 --num_workers 16 --learning_rate 0.001 --num_iters 20000 --session 3 --batch_size 16 --trial 1
```

- Trial 1: lr 1e-3
- Trial 2: lr 1e-2

```shell
CUDA_VISIBLE_DEVICES=0 python core/train_self_discriminator1.py --checkpoint_every 2000 --print_every 50  --num_iters 10000 --batch_size 64 --trial 1
```

Train on FiveK, and train using mixed trajectory.

- Trial 3: accuracy 0.5.

  reason: it will predict all samples as false sample. Lets see the most favoured sample:

  It will favor black image... So we need additional training sample with random editing.

- Trial 4: add the random sample negative 

  accuracy 0.5.

- Trial 5: x2 the coefficient on positive data.

Train with lower dimension, lower resolution 

```shell
CUDA_VISIBLE_DEVICES=0 python core/train_self_discriminator1.py  --cond_nc 64 --hidden_size 64 --checkpoint_every 2000 --print_every 50  --num_iters 10000 --batch_size 64 --trial 1
```

- Trial 6: default (GPU1)

  not work 50% accuracy, seems the modeling ability is too small, it will always output 0.5 accuracy.

- Trial 7: add the triplet loss to classification loss (GPU2)

  Saturateds

- Trial 8: purely rankingf loss (GPU4)

- Trial 9: mimic L1 distance

  reweighted by L1 distance

- Trial 10: D and G combination.

### Test

```shell
CUDA_VISIBLE_DEVICES=0 python core/test_self_discriminator1.py --print_every 50   --trial 1
```



## Planning on Discriminator

```shell
CUDA_VISIBLE_DEVICES=0 python core/utils/beam_search.py --trial 7
```

- trial 7: working on seq2seqGAN with only vanila GAN

  ```shell
  CUDA_VISIBLE_DEVICES=0 python core/utils/beam_search.py --no_lsgan --no_vgg_loss --no_ganFeat_loss --trial 7
  ```

- 

## Demo on Discriminator

```shell
CUDA_VISIBLE_DEVICES=0 python core/demo_seq2seqGAN_plan.py --no_lsgan --no_vgg_loss --no_ganFeat_loss --trial 7
```

## Noun Pharse Selection

- [ ] Get the allen ALP installed to exam the semantic role labeling and semantic parser.

## Edge-connect Adaptation

1. For image generation, I just use the most simple case, with single pari. However, finally I need to change the data feeding, from input raw image and mask, to input a batch of tensor. Because the final model just let tensor pass through operators. 

2. Need to know how to control the output file.

3. Edge-connect cautions:

   - mask must be stored in .png format. 

   - the input image and mask size should be the divide by 4. 

     This requires additional data processing.

   - might need to dilate the pixels.
   - to get good inpainting result, ensure the input image size < 500x500.

#### Usage

Go to `$ROOT/synthdata`, run `gen_local.py`, it will generate mask file at `$ROOT/synthdata/refer/data/images/mscoco/masks/train2014` and the inpainted images at `$ROOT/pyutils/edge-connect/checkpoints/places2/results`.

## AMT for Real Image

### Segmentation

```shell
cd $ROOT/misc
```

Visualize both segment every thing and UPS Net

```shell
python vis_seg_every_thing
```

Visualize UPSNet on Zhopped

```shell
python vis_zhopped.py
```

Visualize UPSNet on reddit

```shell
python vis_reddit.py
```

### Round 1 Annotation

```shell
cd $ROOR/misc
```

Visualize Zhopped annotation round 1

```shell
python vis_zhopped_round_1.py
```

Visualize Reddit annotation round 1

```shell
python vis_reddit_round_1.py
```