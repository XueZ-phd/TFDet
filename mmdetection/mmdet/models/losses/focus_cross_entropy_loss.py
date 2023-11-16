# Copyright (c) OpenMMLab. All rights reserved.
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from ..builder import LOSSES
from .focal_loss import FocalLoss, py_sigmoid_focal_loss, py_focal_loss_with_prob, sigmoid_focal_loss

def select_triplets(embeddings, labels):
    anchors = []
    positives = []
    negatives = []
    batch, c = embeddings.shape
    pairwise_distances = torch.cdist(embeddings, embeddings, p=2)

    num_samples = embeddings.shape[0]

    for i in range(num_samples):
        anchor = embeddings[i]
        anchor_label = labels[i]

        # 找到正样本（与锚样本属于同一类别的样本）
        positive_mask = labels == anchor_label
        positive_mask[i] = False  # 排除掉anchor本身
        positive_distances = pairwise_distances[i, positive_mask]

        if len(positive_distances) == 0:
            continue

        positive_index = torch.argmin(positive_distances)
        positive_embedding = embeddings[positive_mask][positive_index]

        # 找到负样本（与锚样本属于不同类别的样本）
        negative_mask = labels != anchor_label
        negative_distances = pairwise_distances[i, negative_mask]

        if len(negative_distances) == 0:
            continue

        negative_index = torch.argmax(negative_distances)
        negative_embedding = embeddings[negative_mask][negative_index]

        anchors.append(anchor.unsqueeze(0))
        positives.append(positive_embedding.unsqueeze(0))
        negatives.append(negative_embedding.unsqueeze(0))
    if len(anchors) > 0:
        return torch.cat(anchors, 0), torch.cat(positives, 0), torch.cat(negatives, 0)
    else:
        return None, None, None


def select_triplets_faster(embeddings, labels):
    device = embeddings.device  # 获取嵌入向量所在的设备

    pairwise_distances = torch.cdist(embeddings, embeddings, p=2)

    positive_mask = labels.unsqueeze(1) == labels.unsqueeze(0)
    positive_mask.fill_diagonal_(0)  # 排除掉anchor本身
    positive_distances = torch.where(positive_mask, pairwise_distances, torch.tensor(float('inf')).to(device))

    positive_indices = torch.argmin(positive_distances, dim=1)
    positive_embeddings = torch.gather(embeddings, 0, positive_indices.unsqueeze(1).expand(-1, embeddings.shape[1]))

    negative_mask = labels.unsqueeze(1) != labels.unsqueeze(0)
    negative_distances = torch.where(negative_mask, pairwise_distances, torch.tensor(float('-inf')).to(device))

    negative_indices = torch.argmax(negative_distances, dim=1)
    negative_embeddings = torch.gather(embeddings, 0, negative_indices.unsqueeze(1).expand(-1, embeddings.shape[1]))

    mask = (positive_indices != negative_indices)  # 排除掉没有符合条件的样本
    anchors = embeddings[mask]
    positives = positive_embeddings[mask]
    negatives = negative_embeddings[mask]

    if anchors.shape[0] > 0:
        return anchors, positives, negatives
    else:
        return None, None, None


class TripletLoss(nn.Module):
    '''
    Compute normal triplet loss or soft margin triplet loss given triplets
    '''
    def __init__(self, margin=None):
        super(TripletLoss, self).__init__()
        self.margin = margin
        if self.margin is None:  # if no margin assigned, use soft-margin
            self.Loss = nn.SoftMarginLoss()
        else:
            self.Loss = nn.TripletMarginLoss(margin=margin, p=2)

    def forward(self, anchor, pos, neg):
        if self.margin is None:
            num_samples = anchor.shape[0]
            y = torch.ones((num_samples, 1)).view(-1)
            if anchor.is_cuda: y = y.cuda()
            ap_dist = torch.norm(anchor-pos, 2, dim=1).view(-1)
            an_dist = torch.norm(anchor-neg, 2, dim=1).view(-1)
            loss = self.Loss(an_dist - ap_dist, y)
        else:
            loss = self.Loss(anchor, pos, neg)

        return loss


@LOSSES.register_module()
class FocusClassificationLoss(nn.Module):

    def __init__(self,
                 use_sigmoid=True,
                 gamma=2.0,
                 alpha=0.25,
                 focal_loss_weight=1.0,
                 triplet_loss_weight=1.0,
                 triplet_margin=0.5,
                 reduction='mean',
                 activated=False):

        super(FocusClassificationLoss, self).__init__()
        assert use_sigmoid is True, 'Only sigmoid focal loss supported now.'
        self.use_sigmoid = use_sigmoid
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.focal_loss_weight = focal_loss_weight
        self.activated = activated

        self.triplet_loss_weight = triplet_loss_weight
        self.triplet_loss = TripletLoss(triplet_margin)

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        if self.use_sigmoid:
            if self.activated:
                calculate_loss_func = py_focal_loss_with_prob
            else:
                if torch.cuda.is_available() and pred.is_cuda:
                    calculate_loss_func = sigmoid_focal_loss
                else:
                    num_classes = pred.size(1)
                    target = F.one_hot(target, num_classes=num_classes + 1)
                    target = target[:, :num_classes]
                    calculate_loss_func = py_sigmoid_focal_loss

            focal_loss_cls = self.focal_loss_weight * calculate_loss_func(
                pred,
                target,
                weight,
                gamma=self.gamma,
                alpha=self.alpha,
                reduction=reduction,
                avg_factor=avg_factor)

        else:
            raise NotImplementedError

        anchors, positives, negatives = select_triplets_faster(pred, target)
        if anchors is not None:
            loss_cls_triplet = self.triplet_loss_weight * self.triplet_loss(anchors, positives, negatives)
        else:
            loss_cls_triplet = torch.tensor(0.0).to(focal_loss_cls.device)

        return {'loss_cls_ce': focal_loss_cls, 'loss_cls_triplet': loss_cls_triplet}
