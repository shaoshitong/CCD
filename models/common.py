import random

import numpy as np
import torch, math
import torch.nn as nn


def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.

    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding
# Copyright (c) OpenMMLab. All rights reserved.
from typing import Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


class PKDLoss(nn.Module):
    """PyTorch version of `PKD: General Distillation Framework for Object
    Detectors via Pearson Correlation Coefficient.

    <https://arxiv.org/abs/2207.02039>`_.

    Args:
        loss_weight (float): Weight of loss. Defaults to 1.0.
        resize_stu (bool): If True, we'll down/up sample the features of the
            student model to the spatial size of those of the teacher model if
            their spatial sizes are different. And vice versa. Defaults to
            True.
    """

    def __init__(self, loss_weight=1.0, resize_stu=True):
        super(PKDLoss, self).__init__()
        self.loss_weight = loss_weight
        self.resize_stu = resize_stu

    def norm(self, feat: torch.Tensor) -> torch.Tensor:
        """Normalize the feature maps to have zero mean and unit variances.

        Args:
            feat (torch.Tensor): The original feature map with shape
                (N, C, H, W).
        """
        assert len(feat.shape) == 4
        N, C, H, W = feat.shape
        feat = feat.permute(1, 0, 2, 3).reshape(C, -1)
        mean = feat.mean(dim=-1, keepdim=True)
        std = feat.std(dim=-1, keepdim=True)
        feat = (feat - mean) / (std + 1e-6)
        return feat.reshape(C, N, H, W).permute(1, 0, 2, 3)

    def forward(self, preds_S: Union[torch.Tensor, Tuple],
                preds_T: Union[torch.Tensor, Tuple]) -> torch.Tensor:
        """Forward computation.

        Args:
            preds_S (torch.Tensor | Tuple[torch.Tensor]): The student model
                prediction. If tuple, it should be several tensors with shape
                (N, C, H, W).
            preds_T (torch.Tensor | Tuple[torch.Tensor]): The teacher model
                prediction. If tuple, it should be several tensors with shape
                (N, C, H, W).

        Return:
            torch.Tensor: The calculated loss value.
        """
        if isinstance(preds_S, torch.Tensor):
            preds_S, preds_T = (preds_S, ), (preds_T, )

        loss = 0.

        for pred_S, pred_T in zip(preds_S, preds_T):
            size_S, size_T = pred_S.shape[2:], pred_T.shape[2:]
            if size_S[0] != size_T[0]:
                if self.resize_stu:
                    pred_S = F.interpolate(pred_S, size_T, mode='bilinear')
                else:
                    pred_T = F.interpolate(pred_T, size_S, mode='bilinear')
            assert pred_S.shape == pred_T.shape

            norm_S, norm_T = self.norm(pred_S), self.norm(pred_T)

            # First conduct feature normalization and then calculate the
            # MSE loss. Methematically, it is equivalent to firstly calculate
            # the Pearson Correlation Coefficient (r) between two feature
            # vectors, and then use 1-r as the new feature imitation loss.
            loss += F.mse_loss(norm_S, norm_T) / 2
        return loss * self.loss_weight
class ResBlock(nn.Module):
    def __init__(self,channel):
        super().__init__()
        self.block = nn.Sequential(
            nn.GroupNorm(1,channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, channel, (3, 3), (1, 1), (1, 1), bias=True))
        # self.res_block = nn.Sequential(
        #     nn.Conv2d(channel, channel, (3, 3), (1, 1), (1, 1), bias=True))
    def forward(self,x):
        y = self.block(x)
        return y


class FlowAlignModule(nn.Module):
    def __init__(self, teacher_channel, student_channel, teacher_size, student_size, sampling=16,dirac_ratio=1.,weight=1.0):
        super().__init__()
        self.teacher_channel = teacher_channel
        self.student_channel = student_channel
        self.teacher_size = teacher_size
        self.student_size = student_size
        self.time_embedding = student_channel
        self.sampling = sampling
        self.weight = weight
        print("dirac ratios is:",dirac_ratio)
        self.dirac_ratio = 1 - dirac_ratio
        self.align_loss = PKDLoss()
        if isinstance(teacher_size, tuple):
            teacher_size = teacher_size[0]
        if isinstance(student_size, tuple):
            student_size = student_size[0]
        d = int(teacher_size // student_size)
        self.lowermodule = nn.Sequential(
            nn.BatchNorm2d(self.teacher_channel),
            nn.Conv2d(self.teacher_channel, self.student_channel, (1,1), (d, d), (0,0), bias=False))
        self.studentmodule = nn.Identity() #nn.BatchNorm2d(self.student_channel)
        self.flowembedding = ResBlock(self.student_channel)
        self.time_embed = nn.Sequential(
            nn.Linear(self.student_channel,self.student_channel),
        )

        self.iteration = 0

    def forward(self, student_feature, teacher_feature, inference_sampling=4):
        def append_dims(x, target_dims):
            dims_to_append = target_dims - x.ndim
            if dims_to_append < 0:
                raise ValueError(f"input has {x.ndim} dims but target dim is {target_dims}")
            return x[(...,) + (None,) * dims_to_append]
        student_feature =  self.studentmodule(student_feature)
        if teacher_feature is not None:
            _len_dirac = int(self.dirac_ratio*teacher_feature.shape[0])
            teacher_feature[:_len_dirac][torch.randperm(_len_dirac,device=student_feature.device)] \
                = teacher_feature[:_len_dirac].clone()
            teacher_feature = teacher_feature.contiguous()
        if self.training:
            """
            Random Sampling Aware
            """
            inference_sampling = [1,2,4,8,16]
            inference_sampling = np.random.choice(inference_sampling,1)[0]
            indices = reversed(range(1, inference_sampling + 1))
            x = student_feature
            total_velocity = []
            loss = 0.
            if self.iteration<0:
                _weight = 0
            else:
                _weight = self.weight
            self.iteration+=1
            t_output_feature = self.lowermodule(teacher_feature)
            for i in indices:
                _t = torch.ones(student_feature.shape[0],device=student_feature.device) * i / inference_sampling
                _t_embed = self.time_embed(timestep_embedding(_t, self.time_embedding)).view(_t.shape[0],self.time_embedding, 1, 1)
                _velocity = self.flowembedding(x + _t_embed)
                x = x - _velocity / inference_sampling
                total_velocity.append(_velocity)
                loss += (self.align_loss(student_feature - t_output_feature, _velocity)).mean() / inference_sampling * _weight
            return loss, x #student_feature - torch.stack(total_velocity,0).mean(0)
        else:
            x = student_feature
            indices = reversed(range(1, inference_sampling + 1))
            for i in indices:
                _t = torch.ones(student_feature.shape[0], device=student_feature.device) * i / inference_sampling
                _t_embed =  self.time_embed(timestep_embedding(_t, self.time_embedding))
                _t_embed = _t_embed.view(student_feature.shape[0], self.time_embedding, 1, 1)
                _velocity = self.flowembedding(x + _t_embed)
                x = x - _velocity / inference_sampling
            return torch.Tensor([0.]).to(x.device), x