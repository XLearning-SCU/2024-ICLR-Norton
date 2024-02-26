# Dataset

The pre-training data structure is the same as [VideoCLIP](https://github.com/facebookresearch/fairseq/tree/main/examples/MMPT). We understand video data are challenging to download and process, and plan to release our processed how2 data soon. 

For videos, we provide our preprocessing scripts under `scripts/video_feature_extractor` (adapted from `https://github.com/antoine77340/video_feature_extractor`); for text, we pre-tokenizing scripts under `scripts/text_token_extractor`.


### S3D Feature Extraction
We use pre-trained [S3D](https://github.com/antoine77340/S3D_HowTo100M) for video feature extraction. Please place the models as `pretrained_models/s3d_dict.npy` and `pretrained_models/s3d_howto100m.pth`.

We implement a `PathBuilder` to automatically track video ids, source video paths to their feature locations (you may need `conda install -c anaconda pandas`). Decoding may need `pip install ffmpeg-python`.

### Howto100M
[Howto100M](https://www.di.ens.fr/willow/research/howto100m/) is a large-scale video pre-training datasets. You may download videos by yourself and run preprocessing of our scripts. 

Highlight of our preprocessing: (1) we use [sentencified_htm_1200k.json](http://www.robots.ox.ac.uk/~htd/tan/sentencified_htm_1200k.json) from [TAN](https://www.robots.ox.ac.uk/~vgg/research/tan/) ; (2) we shard video/text features using `SharedTensor` in `mmpt/utils/shardedtensor.py` for fast loading during training (faster than `h5py`).

#### video
To extract video features: edit and run `bash scripts/video_feature_extractor/how2/s3d.sh`. (consider to run this on multiple machines; by default, we store features in fp16 to save space and also for faster training).

Split available video ids as `data/how2/how2_s3d_train.lst` and `data/how2/how2_s3d_val.lst`. We have provided our splits in `data/how2`.

Lastly, pack video features into `ShardedTensor` using `python scripts/video_feature_extractor/shard_feature.py`.

#### text
Transform `sentencified_htm_1200k.json` into `.kpl` using `python -m mmpt.processors.dedupprocessor`.

Tokenize dedupped captions `data/how2/sentencified_htm_1200k.pkl` into sharded numpy arrays:  
```
python scripts/text_token_extractor/pretokenization.py scripts/text_token_extractor/configs/bert-base-uncased.yaml
```

### Downstream Youcook, MSRVTT etc.
We plan to release our processed data soon. Please download the data to `data/youcook`, `data/coin`, and `data/msrvtt` accordingly. The file name please refer to the yaml file like `test_path: data/msrvtt/MSRVTT_JSFUSION_test.csv` in `projects/retri/norton/test_vtt_zs.yaml`. 


We use the version of Youcook, MSRVTT, and [Coin](https://coin-dataset.github.io) come with Howto100M and MIL-NCE.
MSRVTT-QA annotations can be downloaded [here]((https://drive.google.com/drive/folders/1_Wyr2VEWU4N-OgLBaQDGWXqD2TXXUBaF)), following ActBERT.
Youcook videos can be downloaded [here](https://www.rocq.inria.fr/cluster-willow/amiech/Youcook2_val.zip) and we only use the testing videos, following [MIL-NCE](https://github.com/antoine77340/MIL-NCE_HowTo100M).

We extract video features for Youcook, MSRVTT, and COIN similar to the first step of Howto100M but we read text from meta data directly and perform on-the-fly tokenization during evaluation.
