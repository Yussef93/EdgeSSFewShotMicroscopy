# EdgeSSFewShotMicroscopy
Repository for [Edge-Based Self-Supervision for Semi-Supervised Few-Shot Microscopy Image Cell Segmentation](https://arxiv.org/abs/2208.02105)
## Abstract
Deep neural networks currently deliver promising results for microscopy image cell segmentation, but they require large-scale labelled databases, which is a costly and time-consuming process. In this work, we relax the labelling requirement by combining self-supervised with semi-supervised learning. We propose the prediction of edge-based maps for self-supervising the training of the unlabelled images, which is combined with the supervised training of a small number of labelled images for learning the segmentation task. In our experiments, we evaluate on a few-shot microscopy image cell segmentation benchmark and show that only a small number of annotated images, e.g. 10% of the original training set, is enough for our approach to reach similar performance as with the fully annotated databases on 1- to 10-shots. Our code and trained models is made publicly available.

![Blockdiagram](https://user-images.githubusercontent.com/57146761/183038077-1dcfb8fc-f84f-4eaa-a786-9e29ff87dee4.png)

##Requirements
### Step 1) Create environment
Install environment from env.yml using conda
```
conda env create -f env.yml
```
### Step 2) Download Datasets
Place each dataset inside 'Datasets/Raw/DATASETNAME/'
Note: Names of datasets are B5, B39, TNBC, EM, ssTEM

### Step 3) Run preprocessing pipeline
Run the code inside run_preprocessing.py to pre-process source and target data.

## Code
### Train with Edges
Example
```
python Learning_main.py  --target 'TNBC' --affine 0 --lr 0.001 --loss 'weightedbce' --name 'miccai_jointOpt_cannyedge_10percentlabelled_100percentunlabelled'
```
### Fine-tune model
Example
```
python Evaluation_main.py  --eval-selections  1 2 3 4 5 6 7 8 9 10   --lr 0.001 --finetune-epochs 20 --finetune-lr 0.0001 --finetune-loss 'weightedbce' --architect 'FCRN' --switchaffine 1 --num-shots 1 3 5 7 10   --statedictepoch 50 --target 'TNBC'   --pretrained-name $YOUR_PRETRAINED_MODEL_NAME --finetune-name $FINETUNED_MODEL_NAME$
```
## Our pretrained models
Will be uploaded soon!

## Cite
Please use the following citation:
```
@inproceedings{dawoud2022edge,
  title={Edge-Based Self-supervision for Semi-supervised Few-Shot Microscopy Image Cell Segmentation},
  author={Dawoud, Youssef and Ernst, Katharina and Carneiro, Gustavo and Belagiannis, Vasileios},
  booktitle={International Workshop on Medical Optical Imaging and Virtual Microscopy Image Analysis},
  pages={22--31},
  year={2022},
  organization={Springer}
}
```

