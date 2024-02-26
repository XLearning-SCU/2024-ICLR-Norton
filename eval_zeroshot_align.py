import os
import sys

import numpy as np
from tqdm import tqdm
import json
import matplotlib.pyplot as plt
import pandas as pd
import math
import torch
import argparse

from torch.nn.utils.rnn import pad_sequence
from torch.utils.data.dataloader import default_collate
from sklearn import metrics
from mmpt_cli.predict import load_config


def read_file(path):
    with open(path, 'r') as f:
        content = f.readlines()
    content = [i.strip() for i in content]
    return content


def read_json(path):
    with open(path, 'r') as f:
        content = json.load(f)
    return content


### from: https://github.com/pytorch/pytorch/issues/15849#issuecomment-518126031
class _RepeatSampler(object):
    """ Sampler that repeats forever.
    Args:
        sampler (Sampler)
    """

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)


# https://github.com/pytorch/pytorch/issues/15849#issuecomment-573921048
class DataLoaderFast(torch.utils.data.dataloader.DataLoader):
    '''reusing cpu workers to speed up dataloader when starting new epoch
    from: https://github.com/pytorch/pytorch/issues/15849#issuecomment-518126031 '''

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        object.__setattr__(self, 'batch_sampler', _RepeatSampler(self.batch_sampler))
        # self.batch_sampler = _RepeatSampler(self.batch_sampler)
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for i in range(len(self)):
            yield next(self.iterator)


class HTM_Align():
    """HTM_Align dataset. 
    For each video, return all the visual features and all the texts."""

    def __init__(self,
                 config,
                 source='htm_align.json',
                 video_feature_path=None,
                 num_clips=4,
                 seq_len=64,
                 ds=1):
        self.num_clips = num_clips
        self.seq_len = seq_len
        self.ds = ds

        self.video_feature_path = video_feature_path
        anno_path = config.dataset.anno_path
        with open(anno_path) as fp:
            anno = json.load(fp)
        self.anno = anno

        self.feature_suffix = 'npy'

        for i in self.anno.keys():
            assert os.path.exists(os.path.join(self.video_feature_path, "{}.{}".format(i, self.feature_suffix)))
        self.video_info = sorted(self.anno.keys())

    def __len__(self):
        return len(self.video_info)

    def __getitem__(self, idx):
        vid = self.video_info[idx]
        anno = self.anno[vid]
        text, text_start, text_end, text_aligned = [], [], [], []
        for seg in anno:
            text_aligned.append(seg[0])
            text_start.append(seg[1])
            text_end.append(seg[2])
            text.append(seg[3])

        video = self._get_video_feature(vid, text_start, text_end, self.num_clips)
        return {'video': video,
                'start': torch.tensor(text_start),
                'end': torch.tensor(text_end),
                'vid': vid,
                'str': text,
                'aligned': torch.tensor(text_aligned)}

    def _get_video_feature(self, vid, start, end, num_clips=4):
        path = os.path.join(self.video_feature_path, "{}.{}".format(vid, self.feature_suffix))
        if path.endswith('.npy'):
            feature = torch.from_numpy(np.load(path))
        else:
            feature = torch.load(path)
        vlen = feature.size(0)

        if self.seq_len == -1:  # take full length
            return feature.float()
        else:
            raise NotImplementedError


@torch.no_grad()
def test_alignment_htm(get_text_visual_sim, device, args, config, video_feature_path):
    D = HTM_Align(config, seq_len=-1, video_feature_path=video_feature_path)
    data_loader = DataLoaderFast(D, batch_size=1, num_workers=0)

    recall = []
    total_vlen = []
    total_text_count = []
    total_aligned_count = []

    total_align_sim = []
    total_align_tgt = []

    seq_len = args.seq_len
    method = 'overlap-seq'  # 'overlap-seq' or 'global'
    print(f'Test Alignment with {method} method')

    for input_data in tqdm(data_loader, total=len(data_loader)):
        video = input_data['video'].to(device)
        text_str = [i[0] for i in input_data['str']]
        tgt_aligned = input_data["aligned"][0].tolist()
        vid = input_data['vid'][0]

        text_str_aligned = np.array(text_str)[np.array(tgt_aligned).astype(bool)].tolist()
        start_idx_aligned = input_data['start'][0].cpu().numpy()[np.array(tgt_aligned).astype(bool)]
        end_idx_aligned = input_data['end'][0].cpu().numpy()[np.array(tgt_aligned).astype(bool)]

        vlen = video.size(1)
        abs_text_pos = torch.stack((input_data['start'][0], input_data['end'][0]), -1).div(vlen).to(device)

        # method1: overlapped moving window along the time axis, then stitch
        if method == 'overlap-seq':
            eps = torch.tensor(1e-5, device=device)
            step = np.arange(0, vlen - seq_len // 2, seq_len // 4)

            # to avoid the leakage of the Ground-truth (annotated/shifted) timestamps,
            # we use the timestamps of non-alignable texts (which are their original ASR timestamps) 
            # to determine the temporal windows
            interpolate_text_mid_ts = (input_data['start'] + input_data['end'])[0].cpu().numpy() / 2

            logits = torch.zeros(len(text_str), vlen, device=device)
            logits_dual = torch.zeros(len(text_str), vlen, device=device)
            overlap_counter = torch.zeros(len(text_str), vlen, device=device)
            logits_a_dual = torch.zeros(len(text_str), device=device)
            logits_a_joint = torch.zeros(len(text_str), device=device)
            text_overlap_counter = torch.zeros(len(text_str), device=device)

            for idx, step_ in enumerate(step):
                # the following line leaks GT timestamps (shown here as a reference, it's not used in our paper)
                # active_text_mask = np.logical_and(step_ - seq_len <= interpolate_text_mid_ts, 
                #                                   interpolate_text_mid_ts <= step_+ seq_len + seq_len)

                # default method: avoid leaking GT timestamps
                nonalignable_text_idx = np.arange(len(text_str))[~np.array(tgt_aligned).astype(bool)]
                nonalignable_text_mid_ts = interpolate_text_mid_ts[~np.array(tgt_aligned).astype(bool)]
                nonalignable_text_window_mask = np.logical_and(
                    step_ - seq_len <= nonalignable_text_mid_ts,
                    nonalignable_text_mid_ts <= step_ + seq_len + seq_len)
                active_nonalignable_text_idx = nonalignable_text_idx[nonalignable_text_window_mask]
                if len(active_nonalignable_text_idx) == 0:
                    continue

                text_window_left, text_window_right = (
                    active_nonalignable_text_idx.min(),
                    active_nonalignable_text_idx.max())
                active_text_mask = np.zeros((len(text_str))).astype(bool)
                # handle edge case, otherwise the heading and tailing alignable texts could be missed
                if idx <= 3:
                    text_window_left = 0
                elif idx >= len(step) - 4:
                    text_window_right = vlen
                active_text_mask[text_window_left: text_window_right + 1] = True

                active_text_str = np.array(text_str)[active_text_mask].tolist()
                active_text_mask_tensor = torch.from_numpy(active_text_mask).to(device).bool()
                if abs_text_pos is not None:
                    active_abs_text_pos = abs_text_pos[active_text_mask][None, :]
                else:
                    active_abs_text_pos = None

                if np.sum(active_text_mask) == 0:
                    continue

                logits_ = get_text_visual_sim(video[:, step_:min(vlen, step_ + seq_len)], active_text_str,
                                              abs_text_pos=active_abs_text_pos)

                if args.use_alignability_head:
                    logits_a_dual_ = logits_['alignability-dual']
                    logits_a_joint_ = logits_['alignability-joint']
                    logits_a_dual[active_text_mask_tensor] += logits_a_dual_[0, :, 0]
                    logits_a_joint[active_text_mask_tensor] += logits_a_joint_[0, 2, :, 0]  # we find the 3rd layer works the best
                    text_overlap_counter[active_text_mask_tensor] += 1
                else:
                    # if in this option, the model is not designed for alignment task, 
                    # but still we can use sim to measure alignability
                    logits_a_dual_ = logits_['dual-sim'][0].max(-1).values
                    logits_a_joint_ = logits_['sim'][0].max(-1).values
                    logits_a_dual[active_text_mask_tensor] += logits_a_dual_
                    logits_a_joint[active_text_mask_tensor] += logits_a_joint_
                    text_overlap_counter[active_text_mask_tensor] += 1

                logits[active_text_mask_tensor, step_:min(vlen, step_ + seq_len)] += logits_['sim'][0, :, :min(vlen, step_ + seq_len) - step_]
                logits_dual[active_text_mask_tensor, step_:min(vlen, step_ + seq_len)] += logits_['dual-sim'][0, :,
                                                                                          :min(vlen, step_ + seq_len) - step_]
                overlap_counter[active_text_mask_tensor, step_:min(vlen, step_ + seq_len)] += 1
            logits = logits.div(torch.maximum(overlap_counter, eps))
            logits_dual = logits_dual.div(torch.maximum(overlap_counter, eps))

            logits_a_dual = logits_a_dual.div(torch.maximum(text_overlap_counter, eps))
            logits_a_joint = logits_a_joint.div(torch.maximum(text_overlap_counter, eps))
            sim = (logits + logits_dual) / 2

        # method2: one pass, by interpolating the positional embedding if necessary
        elif method == 'global':
            logits_ = get_text_visual_sim(video, text_str, interpolate_from=seq_len)
            sim = logits_['sim'][0, -1, :]
            if args.use_alignability_head:
                logits_a_dual = logits_['alignability-dual'][0, :, 0]
                logits_a_joint = logits_['alignability-joint'][0, -1, :, 0]
            else:
                logits_a_dual = logits_['dual-sim'][0, -1].max(-1).values
                logits_a_joint = logits_['sim'][0, -1].max(-1).values

        if args.use_alignability_head:
            align_score = logits_a_joint

        sim.masked_fill_(sim == 0, -6e4)
        prob = sim.softmax(-1)
        vlen = sim.size(-1)

        total_align_tgt.append(np.array(tgt_aligned))
        if args.use_alignability_head:
            total_align_sim.append(align_score.cpu().numpy())
        else:
            total_align_sim.append(sim.max(-1)[0].cpu().numpy())

        sim = sim[torch.as_tensor(tgt_aligned).bool(), :]
        prob = prob[torch.as_tensor(tgt_aligned).bool(), :]

        for text_idx in range(sim.size(0)):
            s = math.floor(start_idx_aligned[text_idx])
            e = math.ceil(end_idx_aligned[text_idx])
            recall.append(s <= prob[text_idx].argmax(-1).item() <= e)

        total_vlen.append(vlen)
        total_text_count.append(len(text_str))
        total_aligned_count.append(len(text_str_aligned))

    total_align_sim = np.concatenate(total_align_sim, 0)
    total_align_tgt = np.concatenate(total_align_tgt, 0)
    assert total_align_tgt.shape == total_align_sim.shape

    auc = metrics.roc_auc_score(total_align_tgt, total_align_sim)

    metric = {'Recall': np.mean(recall), 'AUC': auc}
    print(metric)
    return metric


if __name__ == '__main__':
    """mil-nce and clip results could evaluate by "https://github.com/TengdaHan/TemporalAlignNet/blob/main/eval/eval_zeroshot_align.py".
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("taskconfig", type=str)
    args = parser.parse_args()
    config = load_config(args)

    np.random.seed(0)
    torch.manual_seed(0)
    check_baseline = 'norton'  # milnce or clip-B32 in "https://github.com/TengdaHan/TemporalAlignNet/blob/main/eval/eval_zeroshot_align.py"

    if check_baseline.startswith('norton'):
        video_feature_path = config.dataset.vfeat_dir

        from mmpt.models import MMPTModel

        model, tokenizer, aligner = MMPTModel.from_pretrained(args.taskconfig)


        class DummyArgs():
            def __init__(self):
                self.num_workers = 4
                self.model = 'align'
                self.sim = 'dot'
                self.sentence_mode = 'cls'
                self.num_encoder_layers = 0
                self.seq_len = 32
                self.use_alignability_head = False


        args = DummyArgs()
        device = torch.device('cuda')
        model.to(device)


    def get_text_visual_sim(video_embed, text_str, **kwargs):
        """get text-visual similarity matrix designed for S3D-word2vec / CLIP model.
        i.e. NO visual-textual joint modelling."""
        bsz = len(text_str)
        video_embed = video_embed.repeat(bsz, 1, 1)
        seq_len = video_embed.size(1)
        vfeats = video_embed.view(bsz, seq_len, video_embed.size(-1))
        padding = torch.zeros(
            bsz, model.max_video_len - seq_len, vfeats.size(-1)).to(vfeats.device)
        vfeats = torch.cat([vfeats, padding], dim=1)
        vmasks = torch.cat([
            torch.ones((bsz, seq_len), dtype=torch.bool),
            torch.zeros((bsz, model.max_video_len - seq_len), dtype=torch.bool)
        ],
            dim=1
        ).to(vfeats.device)

        caps_all = []
        cmasks_all = []
        for i in range(bsz):
            caps, cmasks = aligner._build_text_seq(
                tokenizer(text_str[i], add_special_tokens=False)["input_ids"]
            )
            caps, cmasks = caps[None, :], cmasks[None, :]  # bsz=1
            caps_all.append(caps)
            cmasks_all.append(cmasks)
        caps = torch.stack(caps_all).squeeze().to(vfeats.device)
        cmasks = torch.stack(cmasks_all).squeeze().to(vfeats.device)
        if bsz == 1:
            caps = caps.unsqueeze(0)
            cmasks = cmasks.unsqueeze(0)
        output = model.model(caps, cmasks, vfeats, vmasks)
        v = output["video_outputs"][0, 1:-1, :]
        t = output["pooled_text"]
        v /= v.norm(dim=-1, keepdim=True)
        t /= t.norm(dim=-1, keepdim=True)

        similarity = torch.matmul(v, t.transpose(0, 1)).transpose(-1, -2).unsqueeze(0)
        output_ = {'sim': similarity, 'dual-sim': similarity}
        return output_


    test_alignment_htm(get_text_visual_sim, device, args, config, video_feature_path)
    sys.exit(0)
