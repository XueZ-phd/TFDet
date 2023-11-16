import argparse
import copy
import os.path as osp
from glob import glob
import numpy as np
import os
import warnings

import mmcv
from PIL import Image

''''
LWIR和VISIBLE共享同一套Annotations
该代码根据Annotations从文件中索引LWIR的图片，并根据路径索引相匹配的VISIBLE图片
最后只保存LWIR的json文件，即coco['file_name'] = path/to/lwir/image
在mmdetection/mmdet/datasets/pipelines/my_load_rgbt_pipeline.py中，我根据file_name索引lwir image，并匹配相应的visible图片
'''

parser = argparse.ArgumentParser()
parser.add_argument('--train_image_root', type=str,
                    default='/home/zx/cross-modality-det/datasets/sanitized_train_images',
                    help='the root path to training image path',
                    choices=[
                        # 2070
                        '/home/ivlab/new_home/zx/cross-modality-det/datasets/KAIST-Sanitized/sanitized_train_images',
                        # 3060
                        '/UsrFile/Usr3/zx/KAIST-Sanitized/sanitized_train_images',
                        # 3090
                        '/home/zx/cross-modality-det/datasets/sanitized_train_images'
                    ])
parser.add_argument('--test_image_root', type=str,
                    default='/home/zx/cross-modality-det/datasets/test_images',
                    help='the root path to training image path',
                    choices=[
                        # 2070
                        '/home/ivlab/new_home/zx/cross-modality-det/datasets/KAIST-Sanitized/test_images',
                        # 3060
                        '/UsrFile/Usr3/zx/KAIST-MBNet/images',
                        # 3090
                        '/home/zx/cross-modality-det/datasets/test_images'
                    ])
parser.add_argument('--annotation_root', type=str,
                    default='/home/zx/cross-modality-det/datasets/annotations',
                    help='the root path to train (sanitized) and test (improved) annotations',
                    choices=[
                        # 2070
                        '/home/ivlab/new_home/zx/cross-modality-det/datasets/KAIST-Sanitized/annotations',
                        # 3060
                        '/UsrFile/Usr3/zx/KAIST-Sanitized/annotations',
                        # 3090
                        '/home/zx/cross-modality-det/datasets/annotations'
                    ])
parser.add_argument('--out_path', type=str,
                    default='/home/zx/cross-modality-det/datasets/coco_format',
                    help='the root path to output json files',
                    choices=[
                        # 2070
                        '/home/ivlab/new_home/zx/cross-modality-det/datasets/KAIST-Sanitized/coco_format',
                        # 3060
                        '/UsrFile/Usr3/zx/KAIST-Sanitized/coco_format',
                        # 3090
                        '/home/zx/cross-modality-det/datasets/coco_format'
                    ])
args = parser.parse_args()

cls2id_dict = {'person': 0, 'people': 1, 'person?': 2, 'cyclist': 3, 'person?a': 4}


def get_images_generator(mode):
    if mode == 'train':
        # train
        train_annotation_names = sorted(glob(osp.join(args.annotation_root, 'train', '*/*/*.txt')))
        assert len(train_annotation_names) == 7601
        train_image_names = []
        for train_anno_name in train_annotation_names:
            train_anno_name = osp.basename(train_anno_name)
            set_id, v_id, I_id = train_anno_name.split('_')
            train_image_name = osp.join(args.train_image_root, 'lwir_train', f'{set_id}_{v_id}_lwir_{I_id}').replace(
                'txt', 'jpg')
            assert osp.isfile(train_image_name) and osp.isfile(train_image_name.replace('lwir', 'visible')), print(
                train_image_name)
            train_image_names.append(train_image_name)
        return train_image_names
    elif mode == 'test':
        test_annotation_names = sorted(glob(osp.join(args.annotation_root, 'test', '*/*.txt')))
        assert len(test_annotation_names) == 2252
        # test
        test_image_names = []
        for test_anno_name in test_annotation_names:
            test_anno_name = osp.basename(test_anno_name)
            set_id, v_id, I_id = test_anno_name.split('_')
            test_image_name = osp.join(args.test_image_root, 'lwir', 'test', f'{set_id}_{v_id}_lwir_{I_id}').replace(
                'txt', 'jpg')

            assert osp.isfile(test_image_name) and osp.isfile(test_image_name.replace('lwir', 'visible')), print(
                test_image_name)
            test_image_names.append(test_image_name)
        return test_image_names


def parse_txt(image_path, img_w, img_h):
    if osp.basename(osp.dirname(image_path)) == 'lwir_train':
        mode = 'train'
        sep = 'sanitized_annotations/sanitized_annotations'
    elif osp.basename(osp.dirname(image_path)) == 'test':
        mode = 'test'
        sep = 'annotations_KAIST_test_set'
    anno_path = ''.join(osp.basename(image_path).split('_lwir')).replace('.jpg', '.txt')
    anno_path = osp.join(args.annotation_root, mode, sep, anno_path)
    assert osp.isfile(anno_path)

    # load annotations
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        raw_bboxes = np.loadtxt(anno_path, skiprows=1, usecols=[1, 2, 3, 4], ndmin=2)
        raw_labels = np.loadtxt(anno_path, skiprows=1, usecols=0, dtype=str, ndmin=1)
        num_annos = raw_bboxes.shape[0]
        assert raw_labels.shape[0] == num_annos

    # save bboxes, labels, bboxes_ignore, labels_ignore
    bboxes = []
    bboxes_ignore = []

    if num_annos > 0:
        # assert all valid classes and select all valid annotations
        assert (_ in list(cls2id_dict.keys()) for _ in raw_labels)
        area = raw_bboxes[:, 2] * raw_bboxes[:, 3]
        raw_bboxes = raw_bboxes[area > 0]
        raw_labels = raw_labels[area > 0]

        all_bboxes = raw_bboxes.copy()
        bbox_heights = raw_bboxes[:, 3].copy()
        all_bboxes[:, 2:4] = (raw_bboxes[:, 0:2] + raw_bboxes[:, 2:4]).copy()   # xywh 2 xyxy

        condition = []
        # ignore losses generated by categories 'people', 'person?', 'person?a'
        assert len(raw_labels) == len(all_bboxes) == len(bbox_heights)
        for _lbl, _bbox, box_h in zip(raw_labels, all_bboxes, bbox_heights):
            valid_xrange = _bbox[0] >= 0 and _bbox[1] >= 0 and _bbox[2] <= img_w and _bbox[3] <= img_h
            if cls2id_dict[_lbl] == cls2id_dict['person'] and box_h > 0 and valid_xrange:
                condition.append(True)
            else:
                condition.append(False)

        assert len(condition) == len(raw_labels)
        labels = [lbl for cond, lbl in zip(condition, raw_labels) if cond]
        bboxes = [box.tolist() for cond, box in zip(condition, all_bboxes) if cond]

        labels_ignore = [lbl for cond, lbl in zip(condition, raw_labels) if not cond]
        bboxes_ignore = [box.tolist() for cond, box in zip(condition, all_bboxes) if not cond]


    if len(bboxes) == 0:
        bboxes = np.zeros((0, 4))
        labels = np.zeros(0)
    else:
        bboxes = np.array(bboxes, ndmin=2)
        labels = np.array([cls2id_dict[cls] for cls in labels])

    if len(bboxes_ignore) == 0:
        bboxes_ignore = np.zeros((0, 4))
        labels_ignore = np.zeros(0)
    else:
        bboxes_ignore = np.array(bboxes_ignore, ndmin=2)
        bboxes_ignore[:, 0] = np.maximum(bboxes_ignore[:, 0], 0)
        bboxes_ignore[:, 1] = np.maximum(bboxes_ignore[:, 1], 0)
        bboxes_ignore[:, 2] = np.minimum(bboxes_ignore[:, 2], img_w)
        bboxes_ignore[:, 3] = np.minimum(bboxes_ignore[:, 3], img_h)
        labels_ignore = np.array([cls2id_dict[cls] for cls in labels_ignore])

    assert bboxes.shape[0] == labels.shape[0] and bboxes_ignore.shape[0] == labels_ignore.shape[0]
    return bboxes.astype(np.float32), labels.astype(np.int64), \
           bboxes_ignore.astype(np.float32), labels_ignore.astype(np.int64)


def collect_image_infos(mode, exclude_extensions=None):
    img_infos = []
    images_generator = get_images_generator(mode)

    for image_path in mmcv.track_iter_progress(list(images_generator)):
        if exclude_extensions is None or (
                exclude_extensions is not None
                and not image_path.lower().endswith(exclude_extensions)):
            img_pillow = Image.open(image_path)
            img_info = {
                'filename': image_path,
                'width': img_pillow.width,
                'height': img_pillow.height,
            }
            img_infos.append(img_info)
    return img_infos


def cvt_to_coco_json(img_infos, classes, mode):
    image_id = 0
    annotation_id = 0
    coco = dict()
    coco['images'] = []
    coco['type'] = 'instance'
    coco['categories'] = []
    coco['annotations'] = []
    image_set = set()

    def addAnnItem(annotation_id, image_id, category_id, bbox, difficult_flag):
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

        xywh = np.array(
            [bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]])
        annotation_item['area'] = int(xywh[2] * xywh[3])
        if difficult_flag == 1:
            annotation_item['ignore'] = 0
            annotation_item['iscrowd'] = 1
        else:
            annotation_item['ignore'] = 0
            annotation_item['iscrowd'] = 0
        annotation_item['image_id'] = int(image_id)
        annotation_item['bbox'] = xywh.astype(int).tolist()
        annotation_item['category_id'] = int(category_id)
        annotation_item['id'] = int(annotation_id)
        coco['annotations'].append(annotation_item)
        return annotation_id + 1

    for category_id, name in enumerate(classes):
        category_item = dict()
        category_item['supercategory'] = str('none')
        category_item['id'] = int(category_id)
        category_item['name'] = str(name)
        coco['categories'].append(category_item)

    for img_dict in img_infos:
        file_name = img_dict['filename']
        assert file_name not in image_set

        bboxes, labels, bboxes_ignore, labels_ignore = parse_txt(file_name, int(img_dict['width']), int(img_dict['height']))
        # if bboxes.shape[0] == 0 and mode=='train':
        #     continue

        image_item = dict()
        image_item['id'] = int(image_id)
        image_item['file_name'] = str(file_name)
        image_item['height'] = int(img_dict['height'])
        image_item['width'] = int(img_dict['width'])
        coco['images'].append(image_item)
        image_set.add(file_name)

        # bboxes, labels, bboxes_ignore, labels_ignore = parse_txt(file_name)
        for bbox_id in range(len(bboxes)):
            bbox = bboxes[bbox_id]
            label = labels[bbox_id]
            annotation_id = addAnnItem(
                annotation_id, image_id, label, bbox, difficult_flag=0)
        for bbox_id in range(len(bboxes_ignore)):
            bbox = bboxes_ignore[bbox_id]
            label = labels_ignore[bbox_id]
            annotation_id = addAnnItem(
                annotation_id, image_id, label, bbox, difficult_flag=1)

        image_id += 1
    return coco


def main():
    for mode in ['train', 'test']:
        # 1 load image list info
        img_infos = collect_image_infos(mode)

        # 2 convert to coco format data
        coco_info = cvt_to_coco_json(img_infos, list(cls2id_dict.keys()), mode)

        # 3 dump
        save_dir = os.path.join(args.out_path)
        mmcv.mkdir_or_exist(save_dir)
        save_path = osp.join(save_dir, f'lwir_{mode}.json')
        mmcv.dump(coco_info, save_path)
        print(f'save json file: {save_path}')
if __name__ == '__main__':
    main()