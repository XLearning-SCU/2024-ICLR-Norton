#!/bin/bash
# demo script for running the code, please refer to the pretraining.md and endtask.md for more details

# pre-training
python locallaunch.py projects/retri/norton/how2_pretrain.yaml --jobtype local_small

# evaluation on zero-shot endtask
python locallaunch.py projects/retri/norton/test_vtt_zs.yaml --jobtype local_single
python locallaunch.py projects/retri/norton/test_vttqa_zs.yaml --jobtype local_single
python locallaunch.py projects/retri/norton/test_youcook_zs.yaml --jobtype local_single
python locallaunch.py projects/retri/norton/test_youcook_fullvideo_bg_zs.yaml --jobtype local_single
python locallaunch.py projects/retri/norton/test_youcook_fullvideo_zs.yaml --jobtype local_single
python eval_zeroshot_align.py --taskconfig projects/retri/norton/test_how2align_zs.yaml
