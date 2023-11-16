import numpy as np

from mmdet.datasets import build_dataset
from mmcv import Config
import matplotlib.pyplot as plt
import os
os.environ['DISPLAY'] = 'localhost:12.0'
import matplotlib
matplotlib.use('TkAgg')
from scipy import optimize

cfg = Config.fromfile('../configs/faster_rcnn/faster_rcnn_vgg16_fpn_sanitized-kaist.py')
train_set = build_dataset(cfg.data.train)
h, w = None, None
for img_idx in train_set.img_ids:
    annos = train_set.get_ann_info(img_idx)
    bboxes = annos['bboxes']
    labels = annos['labels']
    if len(bboxes)>0:
        assert set(labels)=={0}
        if h is None and w is None:
            h, w = bboxes[:, 3] - bboxes[:, 1]+1.0, bboxes[:, 2] - bboxes[:, 0]+1.0
        else:
            h = np.hstack([h, bboxes[:, 3] - bboxes[:, 1]+1.0])
            w = np.hstack([w, bboxes[:, 2] - bboxes[:, 0]+1.0])

plt.figure()
plt.scatter(h, w)


def f_1(x, A, B):
  return A*x + B

A1, B1 = optimize.curve_fit(f_1, h, w)[0]
x = np.arange(np.min(h), np.max(h))
y1 = A1 * x + B1
y2 = 0.462 * x + 6.418
y3 = 0.296 * x - 0.592
plt.plot(x, y1, 'r')
plt.plot(x, y2, 'r')
plt.plot(x, y3, 'r')
print('y1, w/h: ', 1./A1)
print('y2, w/h: ', 1./0.462)
print('y3, w/h: ', 1./0.296)
print(np.linspace(1./0.462, 1./0.296, 5))
plt.figure()
plt.hist2d(h, w, bins=100, cmap='gray')
plt.colorbar()
plt.show()