## Multi-granularity Correspondence Learning from Long-term Noisy Videos


[![Project Homepage](https://img.shields.io/badge/Project-Homepage-green)](https://lin-yijie.github.io/projects/Norton/)
[![arXiv](https://img.shields.io/badge/arXiv-2401.16702-b31b1b.svg)](https://arxiv.org/pdf/2401.16702.pdf)
[![zhihu](https://img.shields.io/badge/-WeChat@机器之心-000000?logo=wechat&logoColor=07C160)](https://mp.weixin.qq.com/s/q0kL62AM3G1wscTq92HIxA)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/multi-granularity-correspondence-learning-1/long-video-retrieval-background-removed-on)](https://paperswithcode.com/sota/long-video-retrieval-background-removed-on?p=multi-granularity-correspondence-learning-1)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/multi-granularity-correspondence-learning-1/zero-shot-video-retrieval-on-youcook2)](https://paperswithcode.com/sota/zero-shot-video-retrieval-on-youcook2?p=multi-granularity-correspondence-learning-1)

Norton (NOise Robust Temporal Optimal traNsport) is a contrastive model for long-term video learning that enjoys zero-shot transfer to retrieval/QA/sequence labeling style tasks, especially for long videos.
> Yijie Lin, Jie Zhang, Zhenyu Huang, Jia Liu, Zujie Wen, Xi Peng, Multi-granularity Correspondence Learning from Long-term Noisy Videos, ICLR 2024 (oral).  [[paper]](https://arxiv.org/pdf/2401.16702.pdf) 


### Background
<img src="docs/observation.jpg" width="70%" class="center">

Existing video-language studies mainly focus on learning short video clips, leaving **long-term temporal dependencies** rarely explored due to over-high computational cost of modeling long videos. To address this issue, one feasible solution is learning the correspondence between video clips and captions, which inevitably encounters the multi-granularity noisy correspondence (MNC) problem as shown in Fig. 1. To be specific, MNC refers to the clip-caption misalignment (coarse-grained) and frame-word misalignment (fine-grained), hindering temporal learning and video understanding. In this paper, we propose NOise Robust Temporal Optimal traNsport (Norton) that addresses MNC in a unified optimal transport (OT) framework. 


### Method
<img src="docs/method.jpg" width="70%" class="center">

We perform video-paragraph contrastive learning to capture long-term temporal correlations from a fine-to-coarse perspective. Specifically, we first utilize the log-sum-exp operator on the frame-word similarity matrix to obtain fine-grained similarity between clip and caption. Additionally, we append an alignable prompt bucket on the clip-caption similarity matrix to filter out the irrelevant clips or captions. By applying Sinkhorn iterations on the clip-caption similarity matrix, we effectively tackle the asynchronous problem and obtain the optimal transport distance as the video-paragraph similarity.


### News
- [2023-4-14] We are pleased to provide the feature for downstream tasks, see [endtask](endtask.md).
- [2023-1-16] Norton is accepted to ICLR 2024 as oral presentation.

### Todo
- [x] Release Norton [checkpoint](https://drive.google.com/file/d/1ovUBCb-XSoD7bAFKAVa5w13yUqXmCpiS/view?usp=share_link).
- [ ] Release pre-training data (30 fps S3D of Howto100M).
- [x] Release downstream data.

## Get Started

### Contribution
**The core components and contribution of Norton are placed in `mmpt/losses/nce.py`, including video-paragraph contrastive loss and clip-caption contrastive loss.**

### File Organization
```
├── data
│   place the data here
│     └── how2
│         place Howto100M data feature and annotation
├── projects
│   the config files for training/evaluation pipeline
├── mmpt
│   the core code of Norton
│     ├── losses
│     │    the loss functions
│     │    └── nce.py
│     │        the core components and contribution of Norton, including video-paragraph contrastive loss and clip-caption contrastive loss
│     ├── models
│     │   backbone models, the same with VideoCLIP
│     ├── modules
│     │   hard negative searching code, the same with VideoCLIP
│     ├── evaluators
│     │    └── metric.py
│     │        the evaluation metrics like long/short video retrieval, QA, sequence labeling
│     ├── processors
│     │   the data processors, including the data sampling of Howto100M and downstream tasks
│     │   ├── how2retriprocessor.py
│     │   │   the data loading of Howto100M
│     │   └── dsprocessor.py
│     │       the data loading of downstream tasks
├── mmpt_cli
│   the job starting code of training/evaluation pipeline
│      ├── localjob.py
│      │   the job start code of training pipeline
│      └── predict.py
│          the job start code of evaluation pipeline
├── scripts
│   the scripts for extracting video and text features, see DATASET.md for details
├── locallaunch.py
│   entry code, launching jobs
└── run.sh
    demo script for training and evaluating Norton
```

### Installation

We use fairseq as the main trainer (no models/datasets dependency on fairseq). Simply using
```pip install fairseq``` or

```
git clone https://github.com/pytorch/fairseq
cd fairseq
pip install -e .  # also optionally follow fairseq README for apex installation for fp16 training.
export MKL_THREADING_LAYER=GNU  # fairseq may need this for numpy.
```


The code is developed under Python=3.8.13, Pytorch=1.11.0, cuda=11.3 with fairseq=0.12.2.
Most models require `transformers==3.4` for API compatibility `pip install transformers==3.4`. 
In addition, some downstream tasks may need `conda install pandas`.  


### Usage

#### Download Checkpoints
We use pre-trained [S3D](https://github.com/antoine77340/S3D_HowTo100M) for video feature extraction. Please place the models as `pretrained_models/s3d_dict.npy` and `pretrained_models/s3d_howto100m.pth`.

Download Norton checkpoint `https://drive.google.com/file/d/1ovUBCb-XSoD7bAFKAVa5w13yUqXmCpiS/view?usp=share_link` to `runs/retri/norton`.

Download VideoCLIP checkpoint `https://dl.fbaipublicfiles.com/MMPT/retri/videoclip/checkpoint_best.pt` to `runs/retri/videoclip` (used for post-pretraining in our work).

```python
import torch

from mmpt.models import MMPTModel


model, tokenizer, aligner = MMPTModel.from_pretrained(
    "projects/retri/norton/how2_pretrain.yaml")

model.eval()


# B, T, FPS, H, W, C (Norton is trained on 30 fps of s3d)
video_frames = torch.randn(1, 2, 30, 224, 224, 3)
caps, cmasks = aligner._build_text_seq(
    tokenizer("some text", add_special_tokens=False)["input_ids"]
)

caps, cmasks = caps[None, :], cmasks[None, :]  # bsz=1

with torch.no_grad():
    output = model(video_frames, caps, cmasks, return_score=True)
print(output["score"])  # dot-product
```


#### Data Preparation
See [dataset](DATASET.md) for each dataset.

#### Global Config for Training Pipeline
We organize the config file for training/testing pipeline under projects `projects/retri/norton`. 

Either training or evaluation process is configed by a concrete config file (we save all complex arguments into the concrete config file for reproducibility, including fairseq args). For example, pretraining of Norton in `projects/retri/norton/how2_pretrain.yaml` and zero-shot on vtt  is in `projects/retri/norton/test_vtt_zs.yaml`.


We wrap all cmds into `locallaunch.py` and `mmpt_cli/localjob.py`. You can check concrete cmds by `--dryrun` and then drop it for actual run.  
For example, run zero-shot evaluation on MSRVTT,
```
python locallaunch.py projects/retri/norton/test_vtt_zs.yaml --jobtype local_predict  # zero-shot evaluation.
python locallaunch.py projects/retri/norton/vttqa_ft.yaml --jobtype local_single --dryrun  # fine-tuning: use --dryrun to check cmds and drop it to make an actual run; local_single will run on one gpu.
python locallaunch.py projects/retri/norton/test_vttqa_ft.yaml --jobtype local_predict  # testing on fine-tuned model.
```

Pretraining can be run as:  
```
python locallaunch.py projects/retri/norton/how2_pretrain.yaml --jobtype local_single --dryrun # check then drop dryrun; paper is ran on local_small as 2 gpus.
```
You may need to change `--jobtype`, check/extend `LocalJob` in `mmpt_cli/localjob.py` for multi-gpu/multi-node pre-training.

For debuging, you could simply using `train.py` and `predict.py` to start the tasks. The following instructions are generated by `locallaunch.py` in Line 133.
```
python -m torch.distributed.launch --nproc_per_node=2 train.py projects/retri/norton/how2-pretrain.yaml --user-dir mmpt --task mmtask --arch mmarch --criterion mmloss --distributed-world-size 2 --log-interval 1000 --fp16 --num-workers 4 --batch-size 1 --lr 1e-05 --clip-norm 2.0 --optimizer adam --adam-betas '(0.9, 0.98)' --lr-scheduler polynomial_decay --total-num-update 1000000 --warmup-updates 1000 --weight-decay 0.0 --ddp-backend no_c10d --max-epoch 6 --restore-file runs/retri/videoclip/checkpoint_best.pt  --reset-optimizer --reset-dataloader --reset-meters --save-dir runs/retri/norton/ --save-interval-updates 1024 --keep-interval-updates 2 --keep-last-epochs 50
python mmpt_cli/predict.py projects/retri/norton/test_vttqa_zs.yaml
```

The detailed instructions of pretraining and fine-tuning can be found at [pretraining instruction](pretraining.md) and [finetuning instruction](endtask.md).


#### Processors
**Multi-modal** research introduces the complexity on modality alignment from different input sources to losses.
This toolkit leverages `mmpt/processors` to handle various needs of data preprocessing and loading, **alleviating** the needs of multiple `torch.data.utils.Dataset` (that can be tricky for ablation study).  
Processors can also be decoupled from `torch.data.utils.Dataset` for offline preprocessing instead of on-the-fly data preprocessing.

The dataset `mmpt.MMDataset` is decoupled as 3 types of processors: `MetaProcessor`, `VideoProcessor`, `TextProcessor` and `Aligner`. They can be configed in `dataset` field of a config file (e.g., see `projects/retri/norton/how2_pretrain.yaml`).  
`MetaProcessor` is used to load the meta data about a dataset, aka, all video_ids of how2 dataset.  
`VideoProcessor` is used to load the video features about a dataset. For example, S3D features for each second of a video.  
`TextProcessor` is used to load the text (feature). For example, BERT pre-tokenized text clips for how2 dataset (with `start`s, `end`s of timestamps and `cap` for `token_ids`).  
`Aligner` is the core class for different baselines that prepares the training data. For example, sampling a clip, masking tokens for MLM, etc.

#### Performance-tuned Components
To speed up pre-training, this toolkit uses sharded features stored in mmaped numpy, backed by `ShardedTensor` in `mmpt/utils/shardedtensor.py` (adopted from MARGE paper). This reduces the loads of IO for multi-GPU training without loading all features for a video into the memory each time and `ShardedTensor` ensure features are stored in continuous disk space for near random access. This is used for both How2 video features and texts in `mmpt/processors/how2processor.py`.


### Citation
If this codebase is useful for your work, please cite the following papers:

```BibTeX
@inproceedings{lin2024norton,
   title={Multi-granularity Correspondence Learning from Long-term Noisy Videos},
   author={Lin, Yijie and Zhang, Jie and Huang, Zhenyu and Liu, Jia and Wen, Zujie and Peng, Xi},
   booktitle={Proceedings of the International Conference on Learning Representations},
   month={May},
   year={2024}
}
```


### Acknowledgement
This repo is built upon the framework of [VideoCLIP](https://github.com/facebookresearch/fairseq/tree/main/examples/MMPT), thanks for their excellent work.