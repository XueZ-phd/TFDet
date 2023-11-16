import argparse
import os
import os.path as osp
import shutil
import warnings
from glob import glob
from PIL import Image
import mmcv
import numpy as np
from tqdm import tqdm

data_root = '/home/zx/cross-modality-det/datasets/zx-sanitized-kaist-keepPerson-fillNonPerson'
out_root = osp.join(data_root, 'coco_format')
if osp.isdir(out_root):
    shutil.rmtree(out_root)
os.makedirs(out_root)

cls2idx = {'person': 0}

train_annos_root = osp.join(data_root, 'annotations', 'train')
test_annos_root = osp.join(data_root, 'annotations', 'test')

with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    for mode in ['train', 'test']:
        # coco format
        image_id = 0
        annotation_id = 0
        coco = dict()
        coco['images'] = []
        coco['annotations'] = []
        coco['categories'] = [{'id': v, 'name': k, 'supercategory': None} for k, v in cls2idx.items()]
        image_set = set()
        # convert data to coco format
        if mode == 'train':
            annos_root = train_annos_root
        else:
            annos_root = test_annos_root
        # read all images and the corresponding labels
        annos = sorted(glob(osp.join(annos_root, '*.txt')))
        for anf in tqdm(annos, total=len(annos)):
            # image
            ir_img_name = anf.replace('annotations', 'images').replace(mode, f'{mode}_lwir').replace('.txt', '.png')
            assert osp.isfile(ir_img_name) and (ir_img_name not in image_set)
            image = Image.open(ir_img_name)
            image_item = dict()
            image_item['id'] = int(image_id)
            image_item['file_name'] = str(ir_img_name)
            image_item['height'] = int(image.height)
            image_item['width'] = int(image.width)
            coco['images'].append(image_item)
            image_set.add(ir_img_name)
            assert image_item['height'] == 512 and image_item['width'] == 640
            # annotations
            head = ' '.join(np.loadtxt(anf, dtype=str, ndmin=1, max_rows=1))
            assert head == '% bbGt version=3', print(head)
            labels = np.loadtxt(anf, dtype=str, ndmin=2, skiprows=1)
            for label in labels:
                assert label[0] == 'person'
                x, y, w, h = list(map(int, label[1:5]))
                assert x >= 0 and y >= 0 and w > 0 and h > 0 and (x + w) <= 640 - 1 and (y + h) <= 512 - 1
                xywh = [x, y, w, h]
                bbox = list([x, y, x + w, y + h])
                annotation_item = dict()
                annotation_item['segmentation'] = []

                seg = []
                # bbox[] is x1,y1,x2,y2
                # left_top
                seg.append(int(bbox[0]))
                seg.append(int(bbox[1]))
                # left_bottom
                seg.append(int(bbox[0]))
                seg.append(int(bbox[3]))
                # right_bottom
                seg.append(int(bbox[2]))
                seg.append(int(bbox[3]))
                # right_top
                seg.append(int(bbox[2]))
                seg.append(int(bbox[1]))

                annotation_item['segmentation'].append(seg)
                annotation_item['area'] = int(w * h)
                annotation_item['ignore'] = 0
                annotation_item['iscrowd'] = 0
                annotation_item['image_id'] = int(image_id)
                annotation_item['bbox'] = xywh
                annotation_item['category_id'] = int(cls2idx[label[0]])
                annotation_item['id'] = int(annotation_id)
                coco['annotations'].append(annotation_item)
                annotation_id = annotation_id + 1
            image_id += 1

        mmcv.dump(coco, osp.join(out_root, f'lwir_{mode}.json'))
        print(f'{mode.capitalize()} Done!')








