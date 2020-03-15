

## Instroduction

Given basic editing operators, to train a higher level operator selector using RL. 

Slides: [Introduction talk](https://git.corp.adobe.com/jingshi/LDIE/misc/Intro_talk.pptx)



## Dependency

- Pytorch 1.0.0
- opencv-python 
- panopticapi
- easydict
- tensorboardX
- tensorflow<2.0.0
- tabulate
- dominate
- kornia

## Preprocessing

####  Build up data

1. Put FiveK image to `$ROOT/data/FiveK`, it contains 5k images. [TO BE DELETE]

2. go to `pyutils/refer` and buidl up cocoref dataset as the instruction in its README.md.

#### Synthesize data

```bash
cd $ROOT/synthdata
python gen_data.py --release_bs xx --session xx --start xx --end xx
```


Arguments:  

- release_bs: batch size for AMT release
- session: distinguish different session of experiment
- start: starting image index in the dataset
- end: ending image index in the dataset

It will generate the following folders and files in `$ROOT/output`:

- `$ROOT/output/images` : folder containing all of the synthesized data.
- `$ROOT/output/op_record.json`: file that contain all recorded operations.
- `$ROOT/output/web`: folder containing the html that visualizes the synthesizing process.
- `$ROOT/output/config.txt`: file that record the running environment of the current session.
- `$ROOT/AMT/csv`: folder for AMT webpage batch release.

## AMT releasing pipeline

1. Upload the image files to [AWS 3S bucket](https://aws.amazon.com/s3/) and make them public.

   ```bash
   aws s3 cp localfolder s3://cil-ldie --recursive --grants read=uri=http://acs.amazonaws.com/groups/global/AllUsers
   ```

2. Go to [AMT Requester Sandbox](https://requestersandbox.mturk.com/) to build up the annotation layout html, then go to [AMT Worker Sandbox](https://workersandbox.mturk.com/) to check the real work interface.

3. Copy the layout html from sandbox to the real [AMT Requester](https://requester.mturk.com/).

4. For batch release, upload the csv files in `$ROOT/AMT/csv` and start collecting.

### AMT session illustration

- Session 0: one invertible, 1 local color, 1 local inpainting, 1 compound. 0-10000
- Session 1: one invertible.

### AMT results post-process

```bash
cd $ROOT/AMT
python gen_results.py --dataset $DATASET --session $SESSION
```

- combine multiple results into one result.
- Filter out undesired data by manually scan or get the result
- Write the data checking webpage to decide or manually modify the part of sentence needing change or discard.
- **CAUTIONS**: the image name encoding is based on the first 3 element in the record string.

### Real data collecting

#### Crwaling Data

* zhopped

  ```shell
  cd ZhoppedCrawler
  python crawler.py
  python crawl_detail_thread.py
  ```

* reddit 

  ```shell
  cd reddit_crawler
  ```

  and follow the step of the README.md

#### Round1

* generate UPS ponaptic file

  ```shell
  cd UPSNet
  python infer_simple.py --cfg upsnet/experiments/upsnet_resnet101_dcn_coco_3x_16gpu.yaml --weight_path model/upsnet_resnet_101_dcn_coco_270000.pth
  ```

  the detailed configuration should go inside this program.

* generate csv file

  ```shell
  cd AMT
  ```

  * zhopped

    ```shell
    python gen_csv_zhopped.py
    ```

  * reddit

    ```shell
    python gen_reddit.py
    ```

* collecting the result by download all of collected csv file into `results` folder.

* Calculate payment

  ```shell
  python calc_wage.py
  ```

* get result statistic

  ```shell
  python get_statistic.py
  ```

* clean the result

  ```
  python clean_results.py
  ```

#### Round 2

* generate csv files

  ```shell
  python gen_csv_round2.py
  ```

* collecting the result by download all of the collected csv file into `results` folder

* Calculate payment

  ```shell
  python calc_wage_round2.py
  ```

#### Round 3

* generate csv files for AMT

  ```shell
  python gen_csv_round3_AMT.py
  ```

* collecting the result by download all of the collected csv file into `results` folder

* clean the results (majority voting to eliminate the example that cannot see the difference)

  ```shell
  python clean_results_round3.py
  ```

* Visualize together round2 and round3

  ```shell
  cd ../misc
  python vis_IER_round2.py
  ```

* Generate csv file for upwork

  ```shell
  python gen_csv_round3_upwork.py
  ```

  In this step the upwork will need to check the previous result and then type in the language.

#### Round 4

* generate the language for it

  ```shell
  python gen_csv_round4.py
  ```

* collecting the result by downloading all of the collected csv file into `results` folder

* clean the results

  ```shell
  python clean_results_round4.py
  ```

  It will generate the rejected samples back to the AMT so that the rejected sample can be released to the other workers.
  
* Feedback the check result from AMT in `results/IER_round_4/check_IER_round_4_feedback.csv`.

  Process the result by run `clean_results_round4_check1.py`, the valid part is stored in `results/IER_round_3/IER_round_3_checked1_validpart.csv`. And the invalid part is `csv/round_3/AMT/round_3_relabel1_1_5604.csv`, which is reposted to AMT.

  The feedback csv `results/IER_round_4/AMT_IER_relabeled1`, with 71% of them done. I should repost the rest of data.

  Run `python clean_results_round4_1.py`, the newly generated csv about the rest of data is stored in `csv/round_3/AMT/round_3_relabel2_1_1589.csv`. 

* Feedback from **Upwork** is in `results/IER_round_3_check`. Need to fuse all the file.

  Run `python clean_results_round3_upwork.py`, merge the checking data and previous data together at `results/IER_round_4/IER_round_4_upwork.json`

#### Final 

- Fuse the two data by run ` python merge_upwork_and_AMT_json.py` and will output result in `result/IER.json`.
- Visualize the result by run `data_analysis.py`.

**Checked Upwork data 6179 **

Checked AMT data 

image pair number: 6731.

image number in initial json file 6822

image number in feedback from upwork 6671

- [ ] the data has problem. there are 95 operation with nan

### From Name to URL

#### Zhopped

`'http://zhopped.com/fiv/{}'.format('bwCWr.jpg')`

#### Reddit

Reddit does not have the fixed format.

## Augment FiveK Dataset

all the images are moved to a single file `adobe/dataset/FiveK/images`, with the file name `4639_E` as the output name, and `E` can be replaced by `A-E`, and input file name `4639_O`. And it is uploaded to bucket name `s3://cil-ldie/fiveK/images/2226_E.jpg`. 

## MAttNet Debug

- [x] It cannot generate the result from run_detect_to_mask at the final dump json. Because in the `rles` at line 83 in `run_detect_to_mask` file has byte object. Because the COCOmask.encode(np.asfortranarray(m)) will decode to a dict with 'counts' as byte object. However, it might be the problem inside pycocotool.

  Hence my solution is not to dump json file, but dump the pickle file.

  Hence in the evaluation code, it needs modification.

- [x] Check the correctness of maskRCNN

- [x] Check the correctness of the mrcn_head_feats

  'head' (1, 1024, 38,57), 'im_info' (resized_h, resized_w, real_h/resized_h)

**Bug Found:** The prepro part is different. The image list has different order with python2. So the problem might be caused by different python version. 

```bash
python tools/prepro.py --dataset refcoco --splitBy unc
```

This will create `cache/prepro/refcoco_unc/data.json`, this is different with the python2 counterpart.

### Grounding Model Interface Description

For training, every time we can only execute one trajectory. Hence for grounding we only input single image. So we cannot train it in batch.

Input

- Sentence token
- image

output

- Mask

The code is in `example_demo.ipynb`.

### MAttNet training comprehension

##### training command

```shell
./experiments/scripts/train_mattnet.sh 0 refcoco unc
```



The forward pass is in `lossFun` 

#### Data Loader provide

`GtMRCNLoader`  (lib/loaders/gt_mrcn_loader)

Needed output.

- `Feats`
  - `fc7` : (n, 2048, 7, 7): 
  - `pool5`: (n, 1024, 7, 7)
  - `lfeats`: (n, 5) location feature
  - `dif_lfeats`: (n, 25) location feature of different box surounding to the current one.
  - `cxt_fc7`: (n, 5, 2048) context feature of the 5 surrounding boxes.
  - `cxt_lfeats`: (n, 5, 5) context location of the 5 surrounding boxes.

- `labels` (n, 10)
- `neg_Feats`: the same as `Feats`
- `neg_labels`: (n, 10)

##### The pipeline of the data loader

```python
# set up loader
loader = GtMRCNLoader(data_h5=data_h5, data_json=data_json)
loader.prepare_mrcn(head_feats_dir, args)
loader.loadFeats({'ann': ann_feats}) # load fc7 and pool5 feature (196771, 2048)
# fetch data
data = loader.getBatch('train', opt)
Feats = loader.combine_feats(Feats, data['neg_Feats']) # neg feature
Feats = loader.combine_feats(Feats, data['Feats']) # pos feature
```

Some parts needed but unknown: `data['att_labels']`, `data['select_ixs']` 

`att_labels` is attribute label.

- [ ] what is `label` ?

- [ ] how to make the feature to 7x7?

  it directly loads feature map from `cache/feats/refcoco_unc/mrcn/res101_coco_minus_refer_notime/xxxxx.h5`. And then use bbox to crop such feature out. So it has to save the feature map. But now I have no time to do so. I can do it tomorrow after the meeting. 

###### The sample strategy

Sample negative reference=sentence; sample negative object=ann

1. get the neighboring objects

2. classify them into same type and different type 

3. sample the same type object with 0.3 probability and different type object with 0.7 probability. If not enough, just sample from the whole datasets.

   For each positive sample, it just sample one negative sample

4. **TODO** Also have to sample the same sentence but different operation. Currently I do not support such case.

#### Test data loader

Only load single image at once. It will predict the index of the input visual feature and save it. And the input should input the index that can transfer the relative ann_id to absolute ann_id for saving.

However, the evaluation should take more consideration. Since we are mutli-object accuracy, we cannot only choose the top one. We should take the top N and decide how many of its top N precision.

**Metric**: To eliminate the negative effect of the threshold, we build ROC curve over all of the classes. With sklearn, we just need to the gt label and score for each entry, then we get. 

#### Model

`JointMatching` (lib/layers/joint_match)

- combine pos & neg features n->3n

- `Matching`: get the cosine similarity between visual feature and language feature

- `RelationMatching`: pick the most similar visual feature from m features given the language feature.

- `PhraseAttention`: Calculate the module language attention that takes in LSTM embedded input sentence and maintain one module feature. This is the part I need to augment.

  - in each phraseAttention module, make the self.fc to a list of self.fc.

- `SubjectEncoder`: 

  - - [ ] Pay attention to the normalize scale layer, which might be problematic
  - Fuse the fc7 and pool5 feature and then use phrase attention to get the visual feature.
  - For my adoption, since I already extract the feature based on the semantic feature map and the feature is the averaged feature over the feature map, so I don't need to calculate the visual attention.

- `LocationEncoder`:

  - Input each bbox location and the 5 bbox location surronding it.

    **Have to see how to get the surrounded box location**

  - then concatenate them and then go through fc.

- `RelationEncoder`:

  - take the relative bbox location and relative box feature as the input of the algorithm.
  - **Need to prepare the surronding feature**

#### Loss

## Language Model Preprocess

- need to filter out .gif file, which cannot be opened using opencv

  ```bash
  cd $ROOT/prepro
  python filter_gif_data.py
  ```

  Then the original `train.json` and `val.json` will be updated.

- Build up vocabulary for command and operators

  - The first part of the vocabulary is extracted from COCO, since COCO refer dataset may need it, which we will later use to do data augmentation.

    Totally 1999 words, only contain `<UNK>` as special token. In `$MRCN_ROOT/output/refcoco_unc/mrcn_cmr_with_st.json`

  - The second part of the vocabulary comes from the collected command and operators. Also store the Glove feature for dictionary.

    ```bash
    cd $ROOT/prepro
    python get_vocab.py --session $SESSION
    ```

    It will output generated command and operator vocab json file and Glove feature  in `data/refcoco/language`.

  - Build up the vocabs which Glove feature. If it is special token, random set up them.

- Build up index list to represent sentence and operators and split data into train/val/test.

  ```bash
  cd $ROOT/core/utils_
  python preprocess_commands.py --session $SESSION
  ```

  It will output generated 

## Data Loader

### Test the data loader 

```bash
cd $ROOT/core/datasets_
python eddatasets.py
```

## Operator Driven Grounding

### Data Processing

* fuse the operator and the mask index annotation by 

  ```shell
  python fuse_op_request.py
  ```

* a simple data loader that can load all of the mask in `IER` class

  ```shell
  python IER.py
  ```

  - [ ] Difficulty: how to ground to the background is a problem. Because not all of the background mask has the difficulty over there.
  
* test the grounding dataloader

  ```shell
  cd core/datasets_
  python IERdataset.py
  ```

  


#### problems 

- [ ] The mask is not strictly the same size as the original data. Need manually filter out those data.

  **Solution**: when resizing the image, do not use resize ratio, but use fixed size to resize the image.

  But changing the code might affect the labeling mask. Hence just leave it and circumvent the problem in dataloader. 

- [x] feature number does not equal to the nubmer of the ids. 

  - [x] find the image name of the bad sample 

    `reddit` input `4u881h_4u881h.png` output `4u881h_jbl83Im.png` , `pan_feat_num=4`, `object_id_num = 5`.

    In feature domain, there is only 4 labels. [11 house, 51 wall, 53 person, 120 cell phone]

    But in the data domain, there are indeed 5 classes, there is also class belongs to sky!!!

    ```
    pan_2ch = np.uint32(pred_pans_2ch[0])
    pan = 1000 * pan_2ch[:, :, 0] + pan_2ch[:, :, 1] 
    label = np.unique(pan)
    ```

  (2181) reddit_crawler/image/3lzwlp/3lzwlp.jpg

  (2649) reddit_crawler/image/3zhrc3/3zhrc3.png (the left tree will disappear when downsample. However, the good news is that it is unknow class, it does not affet.)

  Stopped at (3566), later keep going on.

  **Solved.** The bug is that get the pan-2ch on small size is not the same as get pan-2ch on large size

#### TODO 

- - [x] Build up the vocabulary for the input language. Also build up the language for the token.

### Data Loading

#### Requirements

`tabulate`, `nltk`

#### Illustration

- The dataloader index is for each combo of operator+request.

- Since the grounding has multiple objects, there are two settings for the final ranking loss.
  1. For positive sample, it has multiple objects, so for each batch we sample one of the positive objects and one of the negative objects.
  2. For each batch, given a sentence, pair all of the positive samples and equal number of negative samples (we might add some constraint to rank all of the similarity of the positive samples that can narrow down the negative samples)

- How to obtain the ids. Since it labels with include and exclude, and the unknown mask is not recorded. Hence there is a dilemma: if I only choose the grounding using inclusive way, it might miss the unknown part for the sample labeled in exclusive way; if we only choose the foreground part, and get the complementary area, we cannot foresee we should predict inclusive or exclusive

#### Bugs

* [x] Fix the box deviation problem

  * check which part cause the box deviation: the part are often lies at the boundary. However, the sample 

    area: [**3.5**, 278, 20037.5, 322, **241**, **13**]

    Len of contour: [15, 100, 777, 86, 460, 28]

    criterion: area < perimeter; is_thing; the area of the second max area is less then 1/10 of the biggist one.

* [x] Visualization of the gt for No: [28, 131, 228] is incorrect, which might be a problem about inclusive/exclusive.

  Fixed. The reason is that I haven't change the inclusive and exclusive.

#### TODO

- [x] Visualization
  
- word attention by sub loc rel
  
- [x] Evaluation based on mask IoU

- [x] Evaluation top-K accuracy based on different confidence threshold

- [x] compare with random operator as prior when testing

- [x] Compare with the single operator as prior for training and testing

- [x] Deploy the AMT for the sentence correction.

- [ ] Check the result from the Upwork

  .....................

- [ ] Propose the thresh based loss that can do better for top2-top-K

- [ ] Train a binary classifier for global or local editing  (need global image feature, which might cause different model with the previous one. Hence need to consider more) 

### Train

```shell
cd $ROOT/core
CUDA_VISIBLE_DEVICES=0 python train_ground.py --id 1 --session 3
```

```shell
cd $ROOT/core
CUDA_VISIBLE_DEVICES=0 python train_op_ground.py --id 3 --session 3
```

For train_op_ground:

We train with two settings:

For evaluation of the validation set, no computing of iou and visualization.

- Id1: normal 
- Id2: classifier with no visual feature.
- 

### Test

```shell
cd $ROOT/core
CUDA_VISIBLE_DEVICES=0 python test_ground.py --id 0
```

```shell
cd $ROOT/core
CUDA_VISIBLE_DEVICES=0 python test_op_ground.py --id 1 --session 3 --phase test
```

- Vis in port 8107

#### Result

- sess 1: standard

  roc: 0.7967

- Sess 2: use the model of sess1, but test with random operator.

  Roc: 0.7327

- Sess 3: train without operator prior 

  ```shell
  python train_ground.py --id 3 --use_op_prior 0
  python test_ground.py --id 3 --use_op_prior 0
  ```

  Roc: 0.80.

  **Explaination**: Even if considering the performance fluctuation, the performance of with or without operator prior does not differ much, which means the prior does not have a good effect. This problem is rooted in our data. There are few examples with different local editing on different objects.

#### OpGround Result

- trial 1: standard 

  Roc .8969; acc@0.5=.9397

- Trial 2: ablation study 2

  `CUDA_VISIBLE_DEVICES=0 python test_op_ground.py --id 2 --session 3 --phase test`

  Roc ..9804; acc@0.5=.9508

#### Op Ground Result

- trial 1: standard.

## Whole pipeline

- [x] Exam the operator parameter number

- [x] Put the input of GT parameter into decoder

- [x] It will skip the first element of y. Hence I need to feed in everything in the same max length.

- [x] Improve the nms layer and roi_pooling layer from torchvision.op

- [ ] Deal with the psudo loss function in inpaint operation

- [x] Assume all paremter loss is MSE, and change the color label from discrete number to real number.

- [ ] try use_attention = False, this code has problem.

- [x] In color space system, all the image are in rgb. However, in detection, it uses BGR. Hence need to manually make the image to RGB.

- [x] Problem with the predicted action, it is always the same with the ground truth one, from the visualization.

  Solved: It is because the operation are always None.

- [x] Test inpainting indivisually by run `python operator`.

- [x] The intermedia input to each layer are cpu tensor. 

- [ ] TODO: change all the operator into GPU mode!! Now all of them are implemented in cpu except for inpainting operator.

- [x] Single example does not converge. The data is not updating.

  All of the parameters are putted into optimizer. parameter predictor has 0 gradient, which has problem. 

  Single converge. The reason for not converge is that learning rate is low.

- [ ] might need to manually increase the batch size by averaging different samples.

- [ ] TODO: only train special token but fix Glove feature. Now all the token embedding are trained with Glove initialization.

- [x] TODO:Write the code for pure evaluation

  - [ ] deal with noun operation predicted that makes empty list.

- [x] Write the evaluation for the action accruacy and parameter accuracy.

  - operator accuracy: BLEU score: nltk.corpus_blue
  - parameter accuracy: mse score | conditioned on correctly matched operation.

- [ ] TODO: check the purpose of sort input.

- [x] collect more data for testing on toy data.

- [x] get the dataloader compatible for both IER and refcoco

- [x] modify the model save path. 

- [x] add the validation test.

- [x] remove the CUDA memory bug of special image.

  This is caused by the buggy code written by the author of MAttNet. He will reduce the def conf at 0.1 each time if the det list is empty.

- [x] Add worker filter code. Build up a black name list.

- [x] Run test over whole dataset and get an accurate result. Compare with no visual feature.

- [x] Store the mask data in the record.

- [ ] Write dataloader with mask as supervision.

- [x] Write the attention visualization code.

- [x] Write the poster.

- [x] Re-filter the examples which does not exist. (Clean the result csv files!)

- [x] Get a csv file both contain zhopped and reddit images, then read images according to this.

  Especially for preparation of the second round.

- [x] Adjust the UPSNet visualization box

- [x] Tell worker the box deviation 

- [ ] Check the correspondence of the mask. (tonight must done)

- [x] Check the crop problem (Not label the operation with crop firstly)

- [x] Check if we need to add more operators. (add black and white, exposure, remove no editing is done)

- [x] Process the result data, pay, work etc.

### Train

```bash
cd $ROOT/core/tools
python run_train.py --checkpoint_every 1500 --num_iters 7500 --trial $TRIAL
```

##### Arguments

- `use_vis_feat`: use the visual feature in decoder. Default 1.

* `display_every`: display every certain number of iters.

#### Memo

- Only support 1 batch size, which might be unstable for training.
- Speed: 0.15s per sample. 
- Model size: 40M

### Test

```bash
cd $ROOT/core/tools
python run_test.py --load_checkpoint_path ../../output/refcoco_train_trial_2/model/checkpoint_best.pt --trial $TRIAL --dataset $DATASET
```

#### Result

| Trial | Dataset | Comment                      | Statistic | BLEU1 | Bright | Contast | Blur  | Sharp | **Color** |
| :---: | :------ | :--------------------------- | :-------- | :---- | :----- | :------ | :---- | :---- | :-------- |
|   2   | Refcoco | standard                     | 1022/129  | 0.90  | 0.065  | 0.098   | 0.001 | 0.022 | 0.034     |
|   3   | Refcoco | no visual feature in decoder | 1022/129  | 0.91  | 0.023  | 0.098   | 0.001 | 0.016 | 0.019     |
|       |         |                              |           |       |        |         |       |       |           |

#### Analysis

- The result shows that even without visual feature, just according to pure langauge, it still can get comparable result.

  

## Multi label classification

### Train

```shell
cd $ROOT/core
CUDA_VISIBLE_DEVICES=0 python train_multilabel.py --session 3 --id 1
```

- Id 1: baseline

- Id2: without visual feature (ablation study, chnage the network structure) (called train ground trial 2)

  `CUDA_VISIBLE_DEVICES=0 python train_multilabel_abl1.py --session 3 --id 2`

### Test

```shell
cd $ROOT/core
CUDA_VISIBLE_DEVICES=0 python test_multilabel.py --session 3 --id 1 --phase test
```

- Id1: baseline.

  test roc 0.9153

- Id2 : no visual feature

  `CUDA_VISIBLE_DEVICES=0 python test_multilabel_abl1.py --session 3 --id 2`

  test roc 0.9182, val roc 

- 

## Integral Model

### Build Vocabulary

Since in our case we would add more or reduce some operators, so we build the vocabulary by running

```shell
cd $ROOT/prepro
python build_vocab.py  --dataset IER --session $SESSION
```

It will save the vocabulary as name 

- `$DATASET_voacbs_sess_$SESS.json`: word vocabulary
- `$DATASET_operator_vocabs_sess_$SESS.json`: operator vocabulary
- `$DATASET_vocabs_glove_feat_$SESS.json`: operator vocabulary

For the DDPG, the vocabulary is session 2

For CVPR20, the vocabulary is session 3

Vocab: merge the panapotic segment vocab with the vocab in the dataset with occurance greater than 2. Finally 2275 vocals and 8 operations. (excluding special token)

### Data Preprocessing

Save the proessed label from AMT to  `$ROOT/data/IER2/IER2.json`.

The structure:

```json
{'input': 'lHT34/lHT34.jpg',
 'output': 'lHT34/jwtyK.jpg', 
 'segment': 'lHT34/lHT34.jpg', 
 'palette': 'lHT34/lHT34plt.jpg',
 'request': 'Two focus points...',
 'detailed_request': 'One focus point on the guy on the far
 'dataset': 'zhopped',
 'workerId': 'ANOTWUSBR3VB1', 
 'operator': {'brightness': {'mask_mode': 'inclusive', 'local': False, 'ids': []},  'saturation': {'mask_mode':   'inclusive', 'local': False, 'ids': []}, 'tint':     {'mask_mode': 'inclusive', 'local': True, 'ids': [1,3,4]}}, 
 'expert_summary': ['turn the image black and white', 'the on the left is very dark, the one one right has more light', 'grayscale and vinaginette alot'], 
'amateur_summary': ['turn the image black and white', 'the on the left is very dark, the one one right has more light']}
```

From the structure to dataloading structure

Implemented in `$ROOT/data/IER2/IER.py `  function  `save_annos` will store the data in `$ROOT/data/IER2/annotations/IER_rl_$SESSION.json`. The structure is as 

```json
'input': '35nxqz_35nxqz.jpg'
'output': '35nxqz_0kp4JpA.jpg'
'is_local': [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
'request_idx': [166, 83, 275, 481, 98, 275, 230, 303, 3, 361, 0, 0, 0, 0, 0]
'request': 'Please remove the man in the background from this photo.'
'operator_idx': [10, 0, 0, 0, 0, 0, 0, 0, 0, 0]
'mask_id': {'10': [3]}} (key is operator idx, value is the mask id refered by the mask proposals in the dataset)
```

The above format is indexed by request_idx

Compare IER.getReqIdx's reqid is continuous?

len(IER.getReqIdx) = 20179, but len(IER.getReqId) = 16453 ?? Because the self.getReqId has multiple keys the same.



hyXJ4_tvwnM.jpg



The data loader:

- [x] The request id is greater than the number of annos

The grounding model: 

- [x] Write the generation process of vocab, commands (all index instead of raw words).
  - hidden the show id to mask
  - resize the original mask

- [x] change the data loader (batch size = 1); change the input image size
- [x] Get the collate funcs
- [ ] make input feature from UPSNet from `fun_feat` 

The executor

- [x] Deal with the operator name

- [ ] How to constraint the action space? As the current action space is redundant

  - when loading data

- [x] Implement the color_bg operator.

- [ ] Currently I cannot train the end class, which needs to be addressed, need to borrow some experience. The Exposure paper use fixed 5 step optimization.

  Add the negative panelty

- [x] Why there is only the first one has large reward?

  It is because the generated image are ranging from [0, 1]. But now the image feature is extracted based on image ranging from [0, 1], but the loaded image is [0, 255] minus [102.9801 115.9465 122.7717] BGR. Hence we should renormalize it back to [0, 1] improve it with some other ideas.

  Moreover, for the parameter it will alwasy predict -1, so the output image will all be zero, so for the later step there is no reward.

- [x] TODO: add random thing 

- [ ] TODO: make the exception back in `decoder` `get_gt_mask`, need to reexame the mask_dict, because the mask is not tensor

- [ ] [Bug] : when resort, the sample 1500 will not pass.

- [ ] Training problem: it will stuck to the situation where only predict one operation and then reduce. So at the begining, supervise is important. (need kind of supervised training)

  Q1: should the parameter prediction being deterministic?

  Q2: does the constant baseline matter? according to my understanding, it can matter. 

  Way to solve the dilemma: 

  - [x] Make a good visualization and loss reward while training

  - [x] Problem: for the second contrast operation, the image does not change...

    This means some operation does not working properly.

  - [ ] Train it with the ground truth tokens and losses, and just use RL to train the parameter.

  - [x] The vocabulary has problem, even not contain people.

- [x] Fit into one training sample (no leaky ReLU for reward! the positive and negative reward are equally important)
- [x] Write the reward curve.

**Possible way to improve:**

- [ ] normalize the reward in a batch, bs=32.

- [ ] borrow the idea from learning to draw and exposure.

- [ ] the sampling method for operation should change with epsilon.

- [ ] the sampling of the parameter should be be deterministic.

- [ ] make the parameter differentiable with the reward.

- [x] might be the problem of predicting an end token.

  The current situation is because all the reward are negative, but the reward will never go to END token. Hence we need to apply a special reward for the end token. Add a final reward for whether the final image is more similar to the original image.

  But even add the penalty for the `END` token, the rewards are still allways negative.

- [x] Fix the step

- [x] finished the sampling for all the operators parameters

- [x] added the repetitive panelty

- [x] fix the supervision for the operators. But it need to learn the parameter in a certain order. I can fixed the order in dataloader.

  {0: '<NONE>', 1: '<START>', 2: '<END>', 3: '<UNK>', 4: 'brightness', 5: 'contrast', 6: 'hue', 7: 'saturation', 8: 'lightness', 9: 'tint', 10: 'inpaint_obj', 11: 'color_bg', 12: 'rotate', 13: 'flip', 14: 'rotate_obj', 15: 'flip_obj', 16: 'deform_obj', 17: 'crop', 18: 'gaussain_blur', 19: 'radial_blur', 20: 'sharpness', 21: 'denoise', 22: 'dehaze', 23: 'edge', 24: 'facet_blur', 25: 'exposure', 26: 'black&white'}

  Order in the sequence: [10: 'inpaint_obj', 11: 'color_bg', 4: 'brightness', 5: 'contrast', 7: 'saturation', 20: 'sharpness']

  - [x] fix the order of operation
  - [x] use gt operation : [**caution**]: some gt operation is empty 

- [x] add the self-critic

- [x] increase the batch size

- [ ] Inpaint image reward is too small, which might affect the result.

- [x] Train the basenet for visual feature and compute W-gan distance.

  

- [ ] Processing the datas.

  

### Train

```shell
cd $ROOT/core/tools
# REINFORCE
CUDA_VISIBLE_DEVICES=0 python run_rl_train.py --checkpoint_every 1500 --num_iters 7500 --shuffle 1 --print_every 1 --visualize_every 10 --operator_supervise 0 --GT_OP_DEBUG 1 --sc_flag 1 --trial $TRIAL
# DDPG
CUDA_VISIBLE_DEVICES=0 python run_ddpg_train.py --checkpoint_every 1500 --num_iters 7500 --shuffle 0 --print_every 1 --visualize_every 100 --operator_supervise 0 --warmup 8 --rm_batch_size 64 --trial $TRIAL
# Modular
CUDA_VISIBLE_DEVICES=0 python core/tools/run_modular_train.py --checkpoint_every 3000 --num_iters 20000 --shuffle 1 --print_every 50 --visualize_every 100 --operator_supervise 1 --num_workers 16 --batch_size 16 --learning_rate 0.0005 --trial 43 --tri_lam 0.1
```

- Related parameter: `img_loss_lam`, `discount_factor`, `leaky_relu`s negative slop.  
- Have to modify the line 134 of the kornia package `/home/jingshi/anaconda3/lib/python3.7/site-packages/kornia/color/hsv.py`
- Remember to change the default path for loading vocab when change session

### Test

```shell
CUDA_VISIBLE_DEVICES=0 python test_modular.py --num_workers 1 --batch_size 1 --trial 41 
```

### Test Integral

```shell
CUDA_VISIBLE_DEVICES=0 python test_integral.py --num_workers 1 --batch_size 1 --session 3 --trial 41 
```

test_integral:'../output/IER_ground_trial_1/best.pth'

test_op:         '../output/IER_ground_trial_1/best.pth'

--num_workers 1 --id 1 --session 3 --batch_size 1 --trial 44

### Result

- trial 1: no op reward, has step panalty 0.005 the reward increase but only predict one opeartor

- Trial 2: no op reward, no step panelty, fixed 2 step and non repetitive operation panelty.

  The crash is due to too low the probability is so that it cannot run the distribution sample.

- Trial3: no op reward, no step panelty, use gt operation. Overfit to certain operators.

- Trial4: have `iter_size` as 8.

- Trial5: self-critic trainings (Wrong)

- Trial6: add entropy panelty and exploration prob, with bs 1.

- Trial7: smoothed the parameter of the operator. Inspect the selection of entropy panelty.

- Trial8: make the entropy panelty 0.01 (no critic)

- Trial9: make the entropy panelty 0.001 (no critic)

- Trial10: entropy 0.01 with critic

- Trial 11: bs 1, EP 0.05

- Trial 12: 1 batch

- Trial 13: 1 sample, iter_size=1

- Trial 14: DDPG 1 sample L2 reward 2000 iteration (-1,1) normalization

- Trial 15: DDPG 1 sample L2 reward 3000 iteration no normalization

- Trial 16: DDPG 1 sample wan reward 2000 iteration with normalization

- Trial 17: DDPG 1 sample wan reward 2000 iteration without normalization

- Trial 18: DDPG 1 sample wgan reward 1000 iteration without normalization, block the reward of next result, individualy update value function and actor.

- Trial 19: DDPG 1 sample reward 10000 iteration with normalization. entropy factor 0.05, noise factor 0.6

- Trial 20: DDPG 1 sample reward 3000 iteration with normalizaion. entropy factor 0.1, noise factor 0.6

- Trial 21: DDPG 1 sample reward 3000 iteration with normalization. entropy factor 0.05, noise factor 1

- Trial 22: DDPG 1 sample with 1500 iteration with only brightness operation. it can **converge**.

- Trial 23: DDPG 1 sample with 500 iteration with brightness, saturation, hue operation.

- Trial 24: DDPG 1 sample with 500 iteration with brightness, saturation, hue operation with three steps **without repetitition**.

- Trial 25: DDPG 1sample with 500 iteration with brightness, saturation with two steps without repetition.

- Trial 26: DDPG 128 sample, only one step, all operation.

- Trial 27: DDPG 128 sample, reduce episode train times=5.

- Trial 28: DDPG 128 sample, reduce episode train times=5, fix context bug.

- Trial 29: Modular 10 sample, with bn, lr 1e-4.

- Trial 30: Modular 10 sample, with bn, lr 1e-5.

- Trial 31: Modular 10 sample, no final bn, lr 1e-5. **Conclusion**: must use final bn layer

- Trial 32: Modular 128 sample, bs 16, lr 1e-5, no LSTM.

- Trial 33: modular bs 16, lr 1e-5, w/o LSTM, final L1 loss. (**baseline**)

- Trial 34: modular bs 16, lr 1e-5, with LSTM, final L1 loss. (find that one batch has the same output for each operation, but the operation is not saturated.) **HAVE PROBLEM**, no better then baseline

- Trial 36: modular bs 16, lr 1e-5, step triplet loss, final L1 loss.  **BETTER THAN BASELINE** from visual judgement, but the loss is similar.

- Trial 37: modular, bs26, lr 1e-5, step triplet loss, final L1 loss. no non-linear layer for the final output.

- Trial 38: modular, bs16, lr 1e-5, step triplet loss but not update previous step. No non-linera layer for the final output.

- Trial 39: modular, bs16, lr 1e-5, step triplet loss but not update previous step, with margin 0.01. No non-linera layer for the final output.

  ----- Change the operators. Retrain -----

- Trial 40: modular bs 16, lr 1e-5, step triplet loss trip_lam=1

- Trial 41: modular bs 16, lr 1e-5, no step triplet loss. **[where the OMN visual result in paper from]** port: 8108

  Test integral port 8112, integral L1: .1200

- Trial 42: modular bs 16, lr 1e-5, step triplet loss tri_lam=0.1. 

- Trial 44: modular bs 16, lr 1e-5, step triplet loss trip_lam=1.

  test: L1 .0893.  integral L1 .1071

  train: port 8110

  Test integral port: 8111

- Trial 45: modular bs 16, lr 1e-5, step triplet loss trip_lam=0

  test: L1 .0925. port 8109. integeral L1: .1066 not make much change ... not good.

  train: port 

- Trial 46: modular bs 16, lr 1e-5, step triplet loss trip_lam=1, shuffle operation.

- Trial 47: modular bs 16, lr 1e-5, step triplet loss trip_lam=1, all perceptual loss with vgg16 relu4-3. A little better.

- Trial 47: evaluate w.r.t global: L1: 0.1064

- Trial 48: only evaluate global operation, use Perceptual loss. 

  Val L1: 0.1016. Test L1: in dist 0.1109, out dist 0.1125, decre: -0.1025

- Trial 49: only evaluate global operation, use Perceptual loss and L1 loss together, triplet loss trip_lam=1.

  Val L1: 0.1020. Test L1: in dist 0.1109, out dist 0.1168, decre: -0.8273

- Trial 50: only evaluate global operation, use L1 loss only.

  Val L1: 0.0983. Test L1: in dist 0.1109, out dist 0.1083, decre -0.2901

- Trial 51: only evaluate local operation, use L1 Loss only.

  Val L1: 0.0676.

  input L1 dist 0.0721, output L1 dist 0.0569, L1 decre: 0.0152
  input perc dist 0.5083, output perc dist 0.4485, perc decre: 0.0597

- Trial 52: train with new perceptual loss.

  input L1 dist 0.1109, output L1 dist 0.1039, L1 decre: 0.0069
  input perc dist 0.4853, output perc dist 0.5024, perc decre: -0.0171

- Trial 53: only evaluate local operation, use L1 Loss only, reduced lr rate 5e-5, triplet 1

  Val L1: 

- Trial 54: only evaluate local operation, use L1 Loss only, reduced lr rate 5e-5, triplet 0.1

  Val L1: 0.1077, not converge

- Trial 55: only evaluate local operation, use L1 Loss only, reduced lr rate 5e-4, triplet 1

  Val L1: 0.1101, not converge.

- Trial 56: only evaluate local operation, use L1 loss only, reduced lr rate 5e-3, triplet 1

  Val L1: 0.0986, seems converged

- 

Visualize

```shell
# list
aws s3 ls s3://cil-ldie/LDIE/output/
# copy
aws s3 cp localfolder s3://cil-ldie/LDIE/output/ --recursive --grants read=uri=http://acs.amazonaws.com/groups/global/AllUsers
# then there will be s3://cil-ldie/LDIE/output/localfolder in the remote
```



- 8084: supervised on synthesized data

- 8093: grounding result

- 8100: trial 2

- 8096: trial 3

- 8101: trial 12: fix to one batch

- 8102: trial 0

- 8104: trial 14: fix to 8 sample

- 8105: trial 33

### Code

####  Train & Test

1. `parser.py` : class `Seq2seqparser` contains `seq2seq` object

   ##### RL methods

   `reinforce_forward` : call `seq2seq.reinforce_forward()`

   `reinforce_backward`: call `seq2seq.reinforce_backward()`

   ##### Test methods

   `parse`: call `seq2seq.sample_output()`

2. `seq2seq.py`: class `Seq2seq` contains `encoder` and `decoder` 

   ##### RL methods

   `reinforce_forward`: firstly encode and `decoder.forward_sample`

   `reinforce_backward`: take in the reward and the operator sequence probability, and the parameter class probability.

3. `decoder.py`: class `Decoder` is the most critical part in this model. it contains 

   `mattnet_ground` for predicting the grounding result (however it can be replaced by a fixing module)

   `EdExecutor`: for executing each operation in each sample step.

   `reinforce_forward`: it still need 

   `forward_sample`: 



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