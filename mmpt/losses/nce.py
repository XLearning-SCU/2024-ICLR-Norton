# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
softmax-based NCE loss, used by this project.
"""

import torch

from torch import nn
import numpy as np
from torch.nn import functional as F

from .loss import Loss


class NCE(Loss):
    def __init__(self):
        # TODO (huxu): define temperature.
        self.loss = nn.CrossEntropyLoss()

    def __call__(self, align_scores, **kargs):
        # note: we reuse the same shape as cls head in BERT (batch_size, 2)
        # but NCE only needs one logits.
        # (so we drop all weights in the second neg logits.)
        align_scores = align_scores[:, :1]
        # duplicate negative examples
        batch_size = align_scores.size(0) // 2
        pos_scores = align_scores[:batch_size]
        neg_scores = align_scores[batch_size:].view(1, batch_size).repeat(
            batch_size, 1)
        scores = torch.cat([pos_scores, neg_scores], dim=1)
        return self.loss(
            scores,
            torch.zeros(
                (batch_size,),
                dtype=torch.long,
                device=align_scores.device),
        )


class T2VContraLoss(Loss):
    """NCE for MM joint space, on softmax text2video matrix.
    """

    def __init__(self, config):
        # TODO (huxu): define temperature.
        self.loss = nn.CrossEntropyLoss()

    def __call__(self, pooled_video, pooled_text, **kargs):
        batch_size = pooled_video.size(0)
        logits = torch.mm(pooled_text, pooled_video.transpose(1, 0))
        targets = torch.arange(
            batch_size,
            dtype=torch.long,
            device=pooled_video.device)
        return self.loss(logits, targets)


class V2TContraLoss(Loss):
    """NCE for MM joint space, with softmax on video2text matrix."""

    def __init__(self, config):
        # TODO (huxu): define temperature.
        self.loss = nn.CrossEntropyLoss()

    def __call__(self, pooled_video, pooled_text, **kargs):
        batch_size = pooled_video.size(0)
        logits = torch.mm(pooled_video, pooled_text.transpose(1, 0))
        targets = torch.arange(
            batch_size,
            dtype=torch.long,
            device=pooled_video.device)
        return self.loss(logits, targets)


class MMContraLoss(Loss):
    """Contrstive Loss of Norton"""

    def __init__(self, config):
        """Constructor.
        Args:
            sequenceloss: video-paragraph contrastive loss.
            beta: the weight of fault negative exploitation.
            sinkhorn_iterations: the number of iterations for Sinkhorn Normalization in Eq. (7).
        """
        self.loss = nn.CrossEntropyLoss()
        if config.loss.sequence_contrast is not None:
            # sequence contrast loss
            self.sequenceloss = SequenceContrastLoss(config)
        self.sequence_contrast_weight = config.loss.sequence_contrast.sequence_contrast_weight
        self.beta = config.loss.beta
        self.sinkhorn_iterations = config.loss.sinkhorn_iterations

    def log_sinkhorn_iterations(self, Z: torch.Tensor, log_mu: torch.Tensor, log_nu: torch.Tensor) -> torch.Tensor:
        """ Perform Sinkhorn Normalization in Log-space for stability"""
        u, v = torch.zeros_like(log_mu), torch.zeros_like(log_nu)
        for _ in range(self.sinkhorn_iterations):
            u = log_mu - torch.logsumexp(Z + v.unsqueeze(0), dim=1)
            v = log_nu - torch.logsumexp(Z + u.unsqueeze(1), dim=0)
        return Z + u.unsqueeze(1) + v.unsqueeze(0)

    @torch.no_grad()
    def log_optimal_transport(self, scores: torch.Tensor) -> torch.Tensor:
        """ Perform Differentiable Optimal Transport in Log-space for stability, following ``SuperGlue: Learning Feature Matching with Graph Neural Networks`` """
        m, n = scores.shape
        one = scores.new_tensor(1)
        ms, ns = (m * one).to(scores), (n * one).to(scores)

        norm = - (ms + ns).log()
        log_mu = norm.expand(m)
        log_nu = norm.expand(n)

        Z = self.log_sinkhorn_iterations(scores, log_mu, log_nu)
        Z = Z - norm  # multiply probabilities by M+N
        return Z

    def __call__(self, pooled_video, pooled_text, **kwargs):
        # get video-paragraph contrastive loss
        if self.sequence_contrast_weight is not None:
            sequence_contrast_loss = self.sequence_contrast_weight * self.sequenceloss(pooled_video, pooled_text, **kwargs)
        else:
            sequence_contrast_loss = 0

        # get clip-caption contrastive loss, we do not normalize the video/text features here following VideoCLIP
        logits_per_video = pooled_video @ pooled_text.t()
        logits_per_text = pooled_text @ pooled_video.t()

        if self.sinkhorn_iterations is not None:
            # Faulty Negative Exploitation by optimal transport
            Q = self.log_optimal_transport(logits_per_video).exp()  # $epsilons$ set to 1 for calculating Eq. (7)
            assert torch.inf not in Q or torch.nan not in Q
            # rectify the target matrix
            sim_targets = torch.zeros(logits_per_video.size()).to(logits_per_video.device)
            sim_targets.fill_diagonal_(1)
            sim_v2t_targets = self.beta * Q + (1 - self.beta) * sim_targets
            sim_t2v_targets = self.beta * Q.T + (1 - self.beta) * sim_targets
            # compute loss
            loss_video = -torch.sum(F.log_softmax(logits_per_video, dim=1) * sim_v2t_targets, dim=1).mean()
            loss_text = -torch.sum(F.log_softmax(logits_per_text, dim=1) * sim_t2v_targets, dim=1).mean()
        else:
            # standard clip-caption contrastive loss
            targets = torch.arange(
                pooled_video.size(0),
                dtype=torch.long,
                device=pooled_video.device)
            loss_video = self.loss(logits_per_video, targets)
            loss_text = self.loss(logits_per_text, targets)
        return loss_video + loss_text + sequence_contrast_loss


class SequenceContrastLoss(Loss):
    """Video-paragraph contrastive loss of Norton"""

    def __init__(self, config):
        """Constructor.
        Args:
          length: the sequence length, we form the long video sequence with consecutive 8 clips/captions defaultly
          clip_per_video: the number of clips sampled from each video
          logit_scale: temperature $\tau$ for the softmax in Eq. (4)
          scale: smoothness parameter $\varepsilon$ In Eq. (2-3)
          alpha_lse: the weight of log-sum-exp in Eq. (5)
          prompt_ratio: select the bottom x% similarity of the original aligned clip-caption pairs as the value of bucket in Eq. (6)
          fusion_weight: the weight of fusion mean average feature with fine-grained log-sum-exp feature to obtain the clip/caption feature
          sinkhorn_iterations: number of iterations for Sinkhorn Normalization in Eq. (3)
        """
        self.loss = nn.CrossEntropyLoss()
        self.length = config.loss.sequence_contrast.threshold  # the sequence length
        self.clip_per_video = config.dataset.clip_per_video
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))  # learnable temperature $\tau$ initialized as 0.07
        self.scale = config.loss.sequence_contrast.scale  # smoothness parameter $\varepsilon$ In Eq. (2-3)
        self.alpha_lse = config.loss.alpha_lse
        self.prompt_ratio = config.loss.sequence_contrast.prompt_ratio
        self.fusion_weight = config.loss.fusion_weight
        self.sinkhorn_iterations = config.loss.sinkhorn_iterations
        assert self.clip_per_video % self.length == 0  # the number of clips sampled from each video should be divisible by the sequence length

    def __call__(self, pooled_video, pooled_text, **kwargs):
        '''video-paragraph contrastive loss from a fine-to-coarse granularity'''

        # Compute similarity based on average pooling of frames/words feature
        pooled_video = pooled_video / pooled_video.norm(dim=-1, keepdim=True)
        pooled_text = pooled_text / pooled_text.norm(dim=-1, keepdim=True)
        clip_logit = torch.mm(pooled_text, pooled_video.T)
        clip_logit = (clip_logit - torch.min(clip_logit)) / (torch.max(clip_logit) - torch.min(clip_logit))  # normalize the logits

        # Compute fine-grained similarity based on frame/word feature
        video_feat = kwargs['video_outputs']
        text_feat = kwargs['text_outputs']
        video_mask_outputs = kwargs['video_mask_outputs']
        text_mask_outputs = kwargs['text_mask_outputs']
        video_feat = video_feat / video_feat.norm(dim=-1, keepdim=True)
        text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)
        retrieve_logits = torch.einsum('atd,bvd->abtv', [text_feat, video_feat])  # a,b: num_clips, t: num_words, v: num_frames
        retrieve_logits = torch.einsum('abtv,at->abtv', [retrieve_logits, text_mask_outputs])
        retrieve_logits = torch.einsum('abtv,bv->abtv', [retrieve_logits, video_mask_outputs])

        # Fine-grained Alignment through log-sum-exp
        t2v_logits = (retrieve_logits / self.alpha_lse).exp()
        t2v_logits = torch.sum(t2v_logits, dim=3) - torch.sum((video_mask_outputs == 0), dim=-1).unsqueeze(0).unsqueeze(2)
        t2v_logits = torch.log(t2v_logits) * self.alpha_lse
        t2v_logits = t2v_logits.sum(dim=2) / (text_mask_outputs.sum(-1).unsqueeze(1))
        t2v_logits = (t2v_logits - torch.min(t2v_logits)) / (torch.max(t2v_logits) - torch.min(t2v_logits))  # normalize the logits

        v2t_logits = (retrieve_logits / self.alpha_lse).exp()
        v2t_logits = torch.sum(v2t_logits, dim=2) - torch.sum((text_mask_outputs == 0), dim=-1).unsqueeze(1).unsqueeze(2)
        v2t_logits = torch.log(v2t_logits) * self.alpha_lse
        v2t_logits = v2t_logits.sum(dim=2) / (video_mask_outputs.sum(-1).unsqueeze(0))
        v2t_logits = (v2t_logits - torch.min(v2t_logits)) / (torch.max(v2t_logits) - torch.min(v2t_logits))
        logsum_logits = (v2t_logits + t2v_logits) / 2

        # enhance the average pooling similarity by incorporating the proposed fine-grained log-sum-exp similarity measure
        clip_logit = (self.fusion_weight * clip_logit + logsum_logits) / (self.fusion_weight + 1)

        long_video_size = pooled_text.shape[0] // self.length

        if self.prompt_ratio is not None:
            # select the Alignable Prompt Bucket value p
            prompt_token_clip, _ = torch.kthvalue(torch.diag(clip_logit), int(self.prompt_ratio * pooled_text.shape[0]))
        else:
            prompt_token_clip = None

        distance = clip_logit.view(long_video_size, self.length, long_video_size, self.length) \
            .transpose(1, 2).reshape(-1, self.length, self.length)
        # calculate the video-paragraph similarity matrix
        logits_prompt = self.calculate_video_logits(distance, prompt_token_clip, long_video_size)

        targets = torch.arange(
            logits_prompt.size(0),
            dtype=torch.long,
            device=logits_prompt.device)

        loss_t = self.loss(logits_prompt, targets)
        loss_v = self.loss(logits_prompt.T, targets)
        return loss_t + loss_v

    def log_sinkhorn_iterations(self, Z: torch.Tensor, log_mu: torch.Tensor, log_nu: torch.Tensor,
                                sinkhorn_iterations: int) -> torch.Tensor:
        """ Perform Sinkhorn Normalization in Log-space for stability"""
        u, v = torch.zeros_like(log_mu), torch.zeros_like(log_nu)
        for _ in range(sinkhorn_iterations):
            u = log_mu - torch.logsumexp(Z + v.unsqueeze(1), dim=2)
            v = log_nu - torch.logsumexp(Z + u.unsqueeze(2), dim=1)
        return Z + u.unsqueeze(2) + v.unsqueeze(1)

    @torch.no_grad()
    def log_optimal_transport_prompt(self, scores: torch.Tensor, sinkhorn_iterations: int, P: torch.Tensor) -> torch.Tensor:
        """ Perform Differentiable Optimal Transport in Log-space for stability, with prompt bucket

        Args:
          scores: similarity matrix
          sinkhorn_iterations: number of iterations to run sinkhorn
          P: value of prompt bucket
        Returns:
            Z: transport assignment matrix (log probabilities)
        """
        b, m, n = scores.shape
        one = scores.new_tensor(1)
        ms, ns = (m * one).to(scores), (n * one).to(scores)

        bins0 = P.expand(b, m, 1)
        bins1 = P.expand(b, 1, n)
        P = P.expand(b, 1, 1)

        # append the prompt bucket to the scores matrix
        couplings = torch.cat([torch.cat([scores, bins0], -1),
                               torch.cat([bins1, P], -1)], 1)

        norm = - (ms + ns).log()
        log_mu = torch.cat([norm.expand(m), ns.log()[None] + norm])
        log_nu = torch.cat([norm.expand(n), ms.log()[None] + norm])
        log_mu, log_nu = log_mu[None].expand(b, -1), log_nu[None].expand(b, -1)

        Z = self.log_sinkhorn_iterations(couplings, log_mu, log_nu, sinkhorn_iterations)
        Z = Z - norm  # multiply probabilities by M+N

        # remove the prompt bucket from the OT assignment matrix
        Z = Z[:, :m, :n]
        return Z

    def calculate_video_logits(self, distance, prompt_token, long_video_size):
        """output logits of video-paragraph similarity matrix
        Args:
          distance: clip-caption similarity matrix
          prompt_token: value of the prompt bucket
          long_video_size: number of video clips per sequence (8 in default)
        """

        with torch.no_grad():
            self.logit_scale.clamp_(0, 4.6052)  # $\tau$ between 0.01, 1
        if self.prompt_ratio is not None:
            # re-align clip with caption through optimal transport, align_logit is the transport assignment matrix
            align_logit = self.log_optimal_transport_prompt(distance / self.scale, self.sinkhorn_iterations, prompt_token / self.scale)
            # compute moving mass of optimal transport for each video-paragraph pair
            mass = align_logit.exp().sum(dim=[1, 2], keepdim=True)
            total_mass = self.length
            # keep the mass of each video-paragraph pair the same
            align_logit = align_logit.exp() * (total_mass / mass)

        # accumulate the similarity scores of clip-caption pairs to obtain the video-paragraph logits
        logits_prompt = self.logit_scale.exp() * torch.sum(align_logit * distance, dim=[1, 2]).view(long_video_size, -1)

        return logits_prompt
