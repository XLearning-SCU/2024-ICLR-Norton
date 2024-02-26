# Zero-shot Transfer and Finetuning

(If you are new to the ideas of `mmpt.processors`, see [README](README.md) first.)
All finetuning datasets (specifically `processors`) are defined in `mmpt.processors.dsprocessor`.
Given the complexity of different types of finetuning tasks, each task may have their own meta/video/text/aligner processors and `mmpt/evaluators/{Predictor,Metric}`.

### Tasks

Currently, we support 4 end datasets: `MSRVTT`, `Youcook`, `COIN`, and `HTM-Align` with the following tasks:  
long video retrieval (w and w/o background): `Youcook`;  
text-video retrieval: `MSRVTT`, `Youcook`;   
action segmentation: `COIN`;  
Video Question and Answering: `MSRVTT-QA`.  

To add your own dataset, you can specify the corresponding processors and config them in the `dataset` field of a config file.

### Zero-shot Transfer (no Training)
Zero-shot transfer will run the pre-trained model (e.g., Norton) directly on testing data. Configs with pattern: `projects/task/*_zs.yaml` are dedicated for zero-shot transfer.
For example, run zero-shot evaluation on MSRVTT retrieval task:
```
python locallaunch.py projects/retri/norton/test_vtt_zs.yaml --jobtype local_predict
```

#### HTM-Align
[HTM-Align](https://www.robots.ox.ac.uk/~vgg/research/tan/) is a manually annotated 80-video subset of HowTo100M (HTM) dataset, to evaluate the alignment performance. 
It is a test set randomly sampled from Food & Entertaining category of HTM.

For a video from the HTM dataset, the annotators (1) annotate if the sentence from ASR is visually alignable with the video,
(2) if alignable, change the start & end timestamps of the sentence to align with the visual content.

We use this dataset to evaluate the effectiveness of solving noisy correspondence. The evaluation code is borrowed from [TAN](https://github.com/TengdaHan/TemporalAlignNet/blob/main/eval/eval_zeroshot_align.py).
Note that HTM-Align is not supported by fairseq and could simply be evaluated by 
```
python eval_zeroshot_align.py --taskconfig projects/retri/norton/test_how2align_zs.yaml 
```

### Fine-tuning

The training of a downstream task is similar to pretraining, execept you may need to specify the `restore_file` in `fairseq.checkpoint` and reset optimizers, see `projects/retri/norton/vttqa_ft.yaml`.
```
python locallaunch.py projects/retri/norton/vttqa_ft.yaml --jobtype local_single
```
We typically do finetuning on 1 gpus (`local_single`).

#### Testing
For each finetuning dataset, you may need to specify a testing config, similar to `projects/retri/norton/test_vttqa_ft.yaml`.  

We define `mmpt.evaluators.Predictor` for different types of prediction. For example, `MSRVTT` and `Youcook` are video-retrieval tasks and expecting to use `RetrievalPredictor`. You may need to define your new type of predictors and specify that in `predictor` field of a testing config.

Each task may also have their own metric for evaluation. This can be created in `mmpt.evaluators.Metric` and specified in the `metric` field of a testing config.

Launching a testing is as simple as training by specifying the path of a testing config:

```
python locallaunch.py projects/retri/norton/test_vttqa_ft.yaml --jobtype local_predict
```

Testing will be launched locally by default since prediction is computationally less expensive.

