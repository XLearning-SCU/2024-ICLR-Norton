# Pretraining

(If you are new to the ideas of `mmpt.processors`, see [README](README.md) first.)
We mostly use [howto100M](https://github.com/antoine77340/howto100m) dataset for pretraining (other datasets are coming). So you are less likely to write a new `MetaProcessor`, `VideoProcessor` or `TextProcessor` but only working on a new `Aligner`, a new model and loss.

### Data Sharding
Pretraining on Howto100M is heavy on IO since we have millions of videos or captions on the hard disk that cannot be fit into the memory. 
It is desirable to have an optimized preprocessing step before the actual dataloading.  

We support data sharding to pack multiple videos into a shards of training data for both videos and captions. 
(see [dataset](DATASET.md) for preprocessing, we will also release our preprocessed data soon).
These shards will be mapped into memory to reduce the frequency of IO access on millions of files. See (processors starting with `Sharded*`).
This will be the default config for a how2 dataset `projects/retri/norton/how2_pretrain.yaml`.

Great thanks to Dmytro Okhonko for sharing the code from MARGE project.

### Checkpoint Preparation
Download VideoCLIP [checkpoint](https://dl.fbaipublicfiles.com/MMPT/retri/videoclip/checkpoint_best.pt) to `runs/retri/videoclip`.
We initialize our network with VideoCLIP checkpoint with only 1 GPU day of post-training.

### Training
Pre-training on Howto100m is expected on 2 A100 GPUS with 80 GB mem (1 is also possible but slower, actually 2 GPUs with 6 hours training is almost enough).
When training on low-memory GPUs (such as V100), you may need to reduce the batch size `num_video_per_batch` to 32 in `projects/retri/norton/how2_pretrain.yaml` and
the infer_scale (for Retrieval Augmentation below) to 8 in Line 59 of `mmpt/tasks/retritask.py`.

launching a pretraing can be done, via:  
```
python locallaunch.py projects/retri/norton/how2_pretrain.yaml --jobtype local_small
```

using one GPU, you might use
```
python locallaunch.py projects/retri/norton/how2_pretrain.yaml --jobtype local_single
```

### Pre-training with Retrieval Augmentation (following VideoCLIP)
This projects support alternatively run a retrieval-augmented model and pre-training, i.e., **searching the nearest videos as the hard negative samples.**
We implement a basic retrieval model that is built on the hidden states of a video and faiss.

You may need to install faiss via `conda install faiss-cpu -c pytorch`. 
(it's fast enough and we don't need GPU version)

Right now, the hidden states of a video is computed as the average of 16 clips of their pooled visual/text hidden states.
See `mmpt/tasks/retritask.py` for more details.
