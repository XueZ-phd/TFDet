import copy
import os
import shutil

import torchvision.ops
import torch

os.environ['DISPLAY']='localhost:11.0'
import pickle
from cv2 import cv2
import numpy as np

from mmdet.datasets import build_dataset
from mmcv import Config
import os.path as osp
from glob import glob

cfg = Config.fromfile('../configs/faster_rcnn/faster_rcnn_vgg16_fpn_sanitized-kaist_v5.py')
show_epoch = not 'epoch_3.pth'
pkl_root = '../runs/FasterRCNN_vgg16_channelRelation_dscSEFusion_similarityMax_1'
score_thres = 0.0
aspect_ratio_thres = 1.0
start_epoch = 3
end_epoch = 4
nms_thres = None

test_dict = {}
test_dict['names'] = ['test-all', 'test-day', 'test-night']
test_dict['setIds'] = [[6, 7, 8, 9, 10, 11], [6, 7, 8], [9, 10, 11]]
test_vidId = {}
test_vidId[6] = [0, 1, 2, 3, 4]
test_vidId[7] = [0, 1, 2]
test_vidId[8] = [0, 1, 2]
test_vidId[9] = [0]
test_vidId[10] = [0, 1]
test_vidId[11] = [0, 1]

# init detection results
out_dir = osp.join(pkl_root, 'epoch_')
if osp.isdir(out_dir):
    shutil.rmtree(out_dir)
os.makedirs(out_dir, exist_ok=True)
test_names = sorted(glob(osp.join('/home/zx/cross-modality-det/datasets/zx-sanitized-kaist-keepPerson-fillNonPerson/images/test_lwir', '*.png')))
# test_names = sorted(glob(osp.join('/home/zx/cross-modality-det/datasets/test_images/lwir/test', '*.jpg')))

for epo in range(start_epoch, min(end_epoch+1, 130, start_epoch+len(glob(osp.join(pkl_root, '*.pkl'))))):
    if show_epoch==f'epoch_{epo}.pth':
        cv2.namedWindow('detected bboxes')
    print('-'*100)
    pkl_file = osp.join(pkl_root, f'epoch_{epo}.pkl')
    with open(pkl_file, 'rb') as f:
        det_bboxes = pickle.load(f)
    # test_mode表示是否过滤没有标注GT的图片，True表示不过滤，False表示过滤。
    dataset = build_dataset(cfg.data.test, dict(test_mode=True))
    data_infos = dataset.data_infos
    assert len(data_infos) == len(det_bboxes)
    # 将所有的txt整合到一起
    for name, setId in zip(test_dict['names'], test_dict['setIds']):
        now_test_names = list(filter(lambda n: int(osp.basename(n).split('_', 1)[0].split('set')[-1]) in setId, test_names))
        print(f'{name}: {len(now_test_names)} images')
        name2id_map = {}
        for idx, tname in enumerate(now_test_names):
            assert osp.realpath(tname) not in list(name2id_map.keys())
            name2id_map[osp.realpath(tname)] = idx + 1

        # 因为classes = ['person', 'person?', 'people', 'cyclist']，所以0=person
        save_bboxes = []
        save_img_ids = []
        tmp_count = 0
        for info, bboxes1img in zip(data_infos, det_bboxes,):
            sid = int(osp.basename(info['filename']).split('_', 1)[0].split('set')[-1])
            if sid not in setId:
                continue
            tmp_count+=1
            img_id = name2id_map[info['filename']]
            assert img_id == tmp_count

            person_bbox = (bboxes1img[0]).copy()
            if person_bbox.shape[0]:
                person_scores = (bboxes1img[0][:, -1]).copy()
                person_bbox[:, 2:4] = bboxes1img[0][:, 2:4] - bboxes1img[0][:, 0:2] + 1.0
                person_bbox[:, 0:2] = person_bbox[:, 0:2] + 0.5
                person_bbox = person_bbox[person_scores >= score_thres]  # 过滤这张图中所有框的score confidence
                aspect_ratio = person_bbox[:, 2] / person_bbox[:, 3]
                person_bbox = person_bbox[aspect_ratio <= aspect_ratio_thres]
            else:
                continue

            if show_epoch==f'epoch_{epo}.pth':
                img = cv2.imread(info['filename'])
                for box_id, box in enumerate(person_bbox):
                    cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]+box[0]), int(box[3]+box[1])), (255, 255, 0), 1)
                cv2.imshow('detected bboxes', img)
                if cv2.waitKey(100)==27: break
                # cv2.destroyAllWindows()

            save_img_ids.extend([[img_id]]*len(person_bbox))
            save_bboxes.extend(person_bbox.tolist())

        assert tmp_count == len(now_test_names)
        out_data = np.hstack([np.asarray(save_img_ids, str), np.asarray(save_bboxes, str)])
        with open(osp.join(out_dir, f'epoch_{epo}-{name}.txt'), 'w') as f:
            np.savetxt(f, out_data, ['%s']*6, ',')

    if show_epoch == f'epoch_{epo}.pth':
        cv2.destroyAllWindows()


from mr_evaluation_script.evaluation_script import evaluate
annFile = '/home/zx/cross-modality-det/code/mmdetection/mr_evaluation_script/KAIST_annotation.json'
phase = "Multispectral"
results = []
for epo in range(start_epoch, min(end_epoch+1, 130, start_epoch+len(glob(osp.join(pkl_root, '*.pkl'))))):
    rstFile = osp.join(out_dir, f'epoch_{epo}-test-all.txt')
    print('-'*100)
    results.append(evaluate(annFile, rstFile, phase))
    print(f'epoch {epo}')
    print('^' * 100)





















