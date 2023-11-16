import os

import numpy as np
import torch
from cv2 import cv2
os.environ['DISPLAY']='localhost:11.0'
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from torchvision.utils import make_grid


def plot_pred_mask(pred_mask, title=None):
    plt.figure()
    plt.imshow(pred_mask.squeeze(0).squeeze(0).detach().cpu().numpy())
    if title:
        plt.title(title)
    plt.show()


def plot_similarity(similarity_value, transformed_similarity=None):
    if transformed_similarity is None:
        plt.figure()
        plt.subplot(1, 2, 1)
        plt.scatter([x for x in range(similarity_value.shape[-1])], similarity_value.squeeze(0).detach().cpu().numpy())
        plt.title('cosine similarity')
        plt.subplot(1, 2, 2)
        plt.hist(similarity_value.squeeze(0).detach().cpu().numpy())
        plt.title('cosine similarity')
        plt.show()
    else:
        plt.figure()
        plt.subplot(2, 2, 1)
        plt.scatter([x for x in range(similarity_value.shape[-1])], similarity_value.squeeze(0).detach().cpu().numpy())
        plt.title('cosine similarity')
        plt.subplot(2, 2, 2)
        plt.hist(similarity_value.squeeze(0).detach().cpu().numpy())
        plt.title('cosine similarity')
        plt.subplot(2, 2, 3)
        plt.scatter([x for x in range(transformed_similarity.shape[1])], torch.sigmoid(transformed_similarity).squeeze(0).squeeze(-1).squeeze(-1).detach().cpu().numpy())
        plt.title('transformed cosine similarity')
        plt.subplot(2, 2, 4)
        plt.hist(torch.sigmoid(transformed_similarity).squeeze(0).squeeze(-1).squeeze(-1).detach().cpu().numpy())
        plt.title('transformed cosine similarity')
        plt.show()


def plot_features(features, normalize=True, pad_value=0.2, scale_each=False, title=None):
    feat = make_grid(features.permute([1, 0, 2, 3]), nrow=30, normalize=normalize, pad_value=pad_value, scale_each=scale_each).permute([1, 2, 0]).detach().cpu().numpy()
    plt.figure()
    plt.imshow(feat)
    if title:
        plt.title(title)
    plt.show()


def plot_attention(images, features, alpha=0.3, vmin=0.0, vmax=2.0, title=None):
    images = images if isinstance(images, list) else list(images)
    image_np = []
    ren = lambda x: (x - np.min(x)) / (np.max(x) - np.min(x))
    for image in images:
        img = image[0].permute([1, 2, 0]).detach().cpu().numpy()
        image_np.append(ren(img))
    features = features if isinstance(features, list) else list(features)
    assert len(images) == len(features)
    h, w = images[0].shape[2:]
    attns = []
    for feat in features:
        attn = torch.sum(torch.softmax(feat, 1) * feat, dim=1)[0].detach().cpu().numpy()
        attn = cv2.resize(attn, (w, h), interpolation=cv2.INTER_LINEAR)
        attns.append(attn)
    images = np.hstack(image_np)
    attns = np.hstack(attns)
    plt.figure()
    plt.imshow(images)
    plt.imshow(attns, alpha=alpha, vmin=vmin, vmax=vmax)
    if title:
        plt.title(title)
    plt.show()


def plot_image(images, title=None):
    images = images if isinstance(images, list) else list(images)
    ren = lambda x: (x - np.min(x)) / (np.max(x) - np.min(x))
    res = []
    for img in images:
        img = ren(img[0].permute([1, 2, 0]).detach().cpu().numpy())
        res.append(img)
    plt.figure()
    plt.imshow(np.hstack(res))
    if title:
        plt.title(title)
    plt.show()


def plot_single_attention(features, alpha=1.0, vmin=0.0, vmax=2.0, title=None):
    attn = torch.sum(torch.softmax(features, 1) * features, dim=1)[0].detach().cpu().numpy()
    plt.figure()
    plt.imshow(attn, alpha=alpha, vmin=vmin, vmax=vmax)
    if title:
        plt.title(title)
    plt.show()


def plot_proposals(image, proposals, attention=None, color=(255, 255, 0), thickness=1, proposal_score=0.0, title=None):
    h, w = image.shape[2:]
    if attention is not None:
        attn = torch.sum(torch.softmax(attention, 1) * attention, dim=1)[0].detach().cpu().numpy()
        attn = cv2.resize(attn, (w, h), interpolation=cv2.INTER_LINEAR)
    ren = lambda x: (x - np.min(x)) / (np.max(x) - np.min(x))
    image = ren(image.squeeze(0).permute([1, 2, 0]).detach().cpu().numpy())
    image_ori = image.copy()
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if isinstance(proposals[0], torch.Tensor):
        proposals = proposals[0].detach().cpu().numpy()
    elif isinstance(proposals[0], np.ndarray):
        proposals = proposals[0]
    for box in proposals:
        x0, y0, x1, y1 = list(map(int, box[:4]))
        score = box[-1]
        if score<proposal_score:
            continue
        cv2.rectangle(image, (x0, y0), (x1, y1), color, thickness)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.figure()
    plt.imshow(np.hstack([image, image_ori]))
    if attention is not None:
        plt.imshow(np.hstack([attn, attn]), alpha=0.3, vmax=2.0, vmin=0.0)
    if title:
        plt.title(title)
    plt.show()


def plot_imshow(x, size=(640, 512), title=None):
    plt.figure()
    if x.shape[0] != 1 or x.shape[0] !=3:
        x = torch.mean(x, dim=0)
    if x.shape[0] == 3:
        x = x.permute([1, 2, 0]).detach().cpu().numpy()
    else:
        x = x.squeeze().detach().cpu().numpy()
    x = cv2.resize(x, size, interpolation=cv2.INTER_LINEAR)
    plt.imshow(x)
    if title:
        plt.title(title)
    plt.show()

