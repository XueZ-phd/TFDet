import os
import shutil

os.environ['DISPLAY']='localhost:12.0'
import pickle
from cv2 import cv2
from tqdm import tqdm
from mmdet.datasets import build_dataset
from mmcv import Config
import os.path as osp
from glob import glob
import matplotlib.pyplot as plt

cfg = Config.fromfile('../configs/faster_rcnn/faster_rcnn_r50_fpn_llvip.py')
show_epoch = not 'epoch_4.pth'
pkl_root = '../runs_llvip/FasterRCNN_r50wMask_ROIFocalLoss5_CIOU20_cosineSE_notDetach_negEntropy1'
save_root = osp.join(pkl_root, 'detection_results')
if osp.exists(save_root):
    shutil.rmtree(save_root)
os.makedirs(osp.join(save_root, 'lwir'), exist_ok=True)
os.makedirs(osp.join(save_root, 'visible'), exist_ok=True)
score_thres = 0.5
nms_thres = None

# init detection results
test_names = sorted(glob(osp.join('/home/zx/cross-modality-det/datasets/LLVIP/LLVIP/lwir/test', '*.jpg')))


pkl_file = osp.join(pkl_root, 'epoch_7.pkl')
with open(pkl_file, 'rb') as f:
    det_bboxes = pickle.load(f)
# test_mode表示是否过滤没有标注GT的图片，True表示不过滤，False表示过滤。
dataset = build_dataset(cfg.data.test, dict(test_mode=True))
data_infos = dataset.data_infos
assert len(data_infos) == len(det_bboxes)

for idx, (info, det) in tqdm(enumerate(zip(data_infos, det_bboxes)), total=len(data_infos)):
    lwir_file = info['file_name']
    rgb_file = lwir_file.replace('lwir', 'visible')
    lwir_img = cv2.imread(lwir_file)
    rgb_img = cv2.imread(rgb_file)
    # ground-truth boxes
    for anno in dataset.get_ann_info(idx)['bboxes']:
        x0, y0, x1, y1 = list(map(int, anno))
        cv2.rectangle(rgb_img, (x0, y0), (x1, y1), (0, 255, 0), 5)
        cv2.rectangle(lwir_img, (x0, y0), (x1, y1), (0, 255, 0), 5)
    # detection boxes
    det_boxes = det[0]
    for box in det_boxes:
        (x0, y0, x1, y1), s = list(map(int, box[:4])), box[4]
        if s<score_thres:
            continue
        cv2.rectangle(rgb_img, (x0, y0), (x1, y1), (0, 0, 255), 5)
        cv2.rectangle(lwir_img, (x0, y0), (x1, y1), (0, 0, 255), 5)
    cv2.imwrite(osp.join(save_root, 'lwir', osp.basename(lwir_file)), lwir_img)
    cv2.imwrite(osp.join(save_root, 'visible', osp.basename(rgb_file)), rgb_img)


