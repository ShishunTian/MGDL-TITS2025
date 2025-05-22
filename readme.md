# Multi-level Guided Discrepancy Learning for Source-Free Object Detection in Hazy Conditions (T-ITS 2025)
A Pytorch Implementation of Multi-level Guided Discrepancy Learning for Source-Free Object Detection in Hazy Conditions (IEEE T-ITS 2025).

## Get Started
### Datasets Preparation
* Cityscapes and Foggy Cityscapes: Download the [Cityscapes/Foggy Cityscapes](https://www.cityscapes-dataset.com/) dataset.

* RTTS: Download the [RTTS](https://sites.google.com/view/reside-dehaze-datasets/reside-%CE%B2) dataset.

All codes are written to fit for the format of PASCAL_VOC. See dataset preparation code in [DA-Faster RCNN](https://github.com/yuhuayc/da-faster-rcnn/tree/master/prepare_data).

We use [DCP](https://github.com/He-Zhang/image_dehaze) and [ALTM](https://github.com/26hzhang/OptimizedImageEnhance/tree/master/matlab/ALTMRetinex) to generate dehazed images as inputs of teacher network. You can modify the variable `preprocess` in `lib/roi_data_layer/minibatch_mgdl.py` to change the folder path.

### Environment
```
#Prepare the environment via conda
conda create -n MGDL python=3.7
conda activate MGDL
conda install pytorch==1.10.2 torchvision==0.11.3  -c pytorch -c conda-forge

#Install packages
cd ./MGDL
pip install -r requirement.txt

#Setup faster-rcnn
cd ./lib
python setup.py build develop
```

## Models
We use the VGG-16 as the backbone. The pretrained model can be downloaded from this [link](https://drive.google.com/file/d/1yO5zJ1-GCRz45B8idd5AwNt1mmz2aBJw/view?usp=sharing). 

We also provide well-trained models. You can download from here:

* Cityscapes -> Foggy Cityscapes mAP = 39.1 [link](https://drive.google.com/file/d/14biNYqD7N-3-j22lzZRzAjZVY2wz-vf2/view?usp=sharing)

* Cityscapes -> RTTS mAP = 37.0 [link](https://drive.google.com/file/d/1Vll7DpyKPKpe7yYds-PYuCgmKWBGzAs_/view?usp=sharing)

## Folder Structures
```
MDGL
└─ pretrained
   └─ vgg16_caffe.pth 
└─ dataset(VOC format)
   └─ cityscapes
      └─ VOC2007
         └─ JPEGImages
         └─ Annotations
         └─ ImageSets
            └─ train_s.txt
            └─ test_s.txt
            └─ train_t.txt
            └─ test_t.txt
   └─ rtts
      └─ ...
└─ ...
```

## Pretrain on Source Datasets
You should check the consistency of categories between the source dataset and the target dataset before training.
```
CUDA_VISIBLE_DEVICES=$GPU_ID python pretrain_source.py \
       --dataset_t cs --epochs 12 --net vgg16 --log_ckpt_name "source_only" --save_dir "your path"
```

## Train
Cityscapes -> Foggy Cityscapes
```
CUDA_VISIBLE_DEVICES=$GPU_ID python train_net_mgdl.py \
       --dataset_t cs_fg --epochs 30 --net vgg16 --log_ckpt_name "cs_fg" \
       --save_dir "your path" --load_name "xxx.pth(Pretrain on Source Datasets)"
```

Cityscapes -> RTTS
```
CUDA_VISIBLE_DEVICES=$GPU_ID python train_net_mgdl_rtts.py \
       --dataset_t cs_rtts --epochs 40 --net vgg16 --log_ckpt_name "cs_rtts" \
       --save_dir "your path" --load_name "xxx.pth(Pretrain on Source Datasets)"
```

## Test
Cityscapes -> Foggy Cityscapes
```
CUDA_VISIBLE_DEVICES=$GPU_ID python test_net.py \
       --dataset_t cs_fg --net vgg16 --log_ckpt_name "cs_fg_test" \
       --save_dir "your path" --load_name "xxx.pth"
```

Cityscapes -> RTTS
```
CUDA_VISIBLE_DEVICES=$GPU_ID python test_net.py \
       --dataset_t cs_rtts --net vgg16 --log_ckpt_name "cs_rtts_test" \
       --save_dir "your path" --load_name "xxx.pth"
```
## Acknowledgement
This repo is developed based on HTCN and AASFOD. Please check [HTCN](https://github.com/chaoqichen/HTCN) and [AASFOD](https://github.com/ChuQiaosong/AASFOD) for more details.

## Citation
If you think this work is helpful for your project, please give it a star and citation. We sincerely appreciate for your acknowledgments.
```
@article{tian2025multi,
  title={Multi-Level Guided Discrepancy Learning for Source-Free Object Detection in Hazy Conditions},
  author={Tian, Shishun and Wang, Yifan and Zeng, Tiantian and Zou, Wenbin},
  journal={IEEE Transactions on Intelligent Transportation Systems},
  year={2025},
  publisher={IEEE}
}
```
