'''
这个程序输入是txt存储的标注文件，输出是将txt标注文件中非person的类别用标注框内的均值填充。
涉及到以下三点细节：
1. 将非person类的标注框内的像素用数据集均值填充，visible的数据集rgb均值为：[91, 85, 75], lwir的数据集rgb均值为：[45, 43, 45]
2. 确保person类不会被其它类别遮挡，即person类不应该被非person类的目标遮挡时被填充
3.
'''
import copy
import os
import shutil
import warnings

os.environ['DISPLAY'] = 'localhost:12.0'
import numpy as np
from cv2 import cv2
import os.path as osp
from glob import glob
from tqdm import tqdm

data_root = '/home/zx/cross-modality-det/datasets'
out_data_root = osp.join(data_root, 'zx-sanitized-kaist-keepPerson-fillNonPerson-add')
if osp.isdir(out_data_root):
    shutil.rmtree(out_data_root)
out_train_annos_root = osp.join(out_data_root, 'annotations', 'train')
out_train_irImg_root = osp.join(out_data_root, 'images', 'train_lwir')
out_train_viImg_root = osp.join(out_data_root, 'images', 'train_visible')
out_test_annos_root = osp.join(out_data_root, 'annotations', 'test')
out_test_irImg_root = osp.join(out_data_root, 'images', 'test_lwir')
out_test_viImg_root = osp.join(out_data_root, 'images', 'test_visible')
for path in [out_train_annos_root, out_train_irImg_root, out_train_viImg_root, out_test_annos_root, out_test_irImg_root, out_test_viImg_root]:
    os.makedirs(path)
# train image root
train_img_root = osp.join(data_root, 'sanitized_train_images')
lwir_train_img_root = osp.join(train_img_root, 'lwir_train')
vis_train_img_root = osp.join(train_img_root, 'visible_train')
# train annotation root
train_anno_root = osp.join(data_root, 'annotations/train/sanitized_annotations/sanitized_annotations')
# test image root
test_img_root = osp.join(data_root, 'test_images')
lwir_test_img_root = osp.join(test_img_root, 'lwir', 'test')
vis_test_img_root = osp.join(test_img_root, 'visible', 'test')
# test annotation root
# test_anno_root = osp.join(data_root, 'annotations/test/annotations_KAIST_test_set_add')
test_anno_root = osp.join(data_root, 'annotations/test/annotations_KAIST_test_set')

# train files
train_lwir_imgs = sorted(glob(osp.join(lwir_train_img_root, '*')))
train_vis_imgs = sorted(glob(osp.join(vis_train_img_root, '*')))
train_annos = sorted(glob(osp.join(train_anno_root, '*')))
assert len(train_lwir_imgs) == len(train_vis_imgs) == len(train_annos)
print(f'train files: {len(train_annos)}')
# test files
test_lwir_imgs = sorted(glob(osp.join(lwir_test_img_root, '*')))
test_vis_imgs = sorted(glob(osp.join(vis_test_img_root, '*')))
test_annos = sorted(glob(osp.join(test_anno_root, '*')))
assert len(test_lwir_imgs) == len(test_vis_imgs) == len(test_annos)
print(f'test files: {len(test_annos)}')

classes = {'person': 0, 'people': 1, 'person?': 2, 'cyclist': 3, 'person?a': 4}
#
with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    for mode in ['train', 'test']:
        if mode is 'train':
            lwir_imgs = train_lwir_imgs.copy()
            vis_imgs = train_vis_imgs.copy()
            annos = train_annos.copy()
            out_anno_root = out_train_annos_root
            out_irImg_root = out_train_irImg_root
            out_viImg_root = out_train_viImg_root
        else:
            lwir_imgs = test_lwir_imgs.copy()
            vis_imgs = test_vis_imgs.copy()
            annos = test_annos.copy()
            out_anno_root = out_test_annos_root
            out_irImg_root = out_test_irImg_root
            out_viImg_root = out_test_viImg_root

        for idx, (irf, vif, anf) in tqdm(enumerate(zip(lwir_imgs, vis_imgs, annos)), total=len(lwir_imgs), desc=f'processing {mode}...'):
            s, v, m, i = osp.basename(irf).split('_')
            s1, v1, m1, i1 = osp.basename(vif).split('_')
            s2, v2, i2 = osp.basename(anf).split('_')
            assert ((s == s1 == s2) and (v == v1 == v2) and (i.rsplit('.')[0] == i1.rsplit('.')[0] == i2.rsplit('.')[0]))

            labels = np.loadtxt(anf, ndmin=2, skiprows=1, dtype=str)
            renew_labels = []
            labels = sorted(labels, key=lambda x: classes[x[0]], reverse=True)
            ir_img = cv2.imread(irf)
            vi_img = cv2.imread(vif)
            assert ir_img.shape[:2] == (512, 640)
            ir_img_ori = copy.deepcopy(ir_img)
            vi_img_ori = copy.deepcopy(vi_img)
            # print(labels)
            # print('='*100)
            for l in labels:
                x, y, w, h = list(map(int, l[1: 5]))
                if w * h <= 0:
                    # print(anf)
                    continue
                x = np.maximum(x - 1, 0)
                x = np.minimum(x, 640 - 1)
                y = np.maximum(y - 1, 0)
                y = np.minimum(y, 512 - 1)

                w = np.minimum(640 - 1 - x, w)
                h = np.minimum(512 - 1 - y, h)

                if not l[0] == 'person':
                    # ir_img[y: y + h, x: x + w, :] = np.mean(ir_img_ori[y: y + h, x: x + w, :], axis=(0, 1), keepdims=True)
                    # vi_img[y: y + h, x: x + w, :] = np.mean(vi_img_ori[y: y + h, x: x + w, :], axis=(0, 1), keepdims=True)
                    ir_img[y: y + h, x: x + w, :] = np.array([45, 43, 45]).astype(ir_img.dtype).reshape((1, 1, 3))
                    vi_img[y: y + h, x: x + w, :] = np.array([75, 85, 91]).astype(vi_img.dtype).reshape((1, 1, 3))
                else:
                    ir_img[y: y + h, x: x + w, :] = ir_img_ori[y: y + h, x: x + w, :]
                    vi_img[y: y + h, x: x + w, :] = vi_img_ori[y: y + h, x: x + w, :]
                    renew_labels.append(f'{l[0]} {x} {y} {w} {h} ' + ' '.join(l[5:]) + '\n')
            # save
            with open(osp.join(out_anno_root, osp.basename(anf)), 'w') as f:
                f.write('% bbGt version=3\n')
                f.writelines(renew_labels)
            cv2.imwrite(osp.join(out_irImg_root, osp.basename(anf).replace('.txt', '.png')), ir_img)
            cv2.imwrite(osp.join(out_viImg_root, osp.basename(anf).replace('.txt', '.png')), vi_img)

            # aug night train images
            if mode is 'train' and s in ['set03', 'set04', 'set05']:
                # source
                source_ir_image_name = osp.join(out_irImg_root, osp.basename(anf).replace('.txt', '.png'))
                source_vi_image_name = osp.join(out_viImg_root, osp.basename(anf).replace('.txt', '.png'))
                source_anno_name = osp.join(out_anno_root, osp.basename(anf))
                # target
                target_ir_image_name = source_ir_image_name.replace('.png', '_1.png')
                target_vi_image_name = source_vi_image_name.replace('.png', '_1.png')
                target_anno_name = source_anno_name.replace('.txt', '_1.txt')
                assert not (osp.isfile(target_vi_image_name) or osp.isfile(target_ir_image_name)
                            or osp.isfile(target_anno_name))
                if not (idx % 3):
                    print(f'copying {osp.basename(source_anno_name)}')
                    shutil.copy(source_ir_image_name, target_ir_image_name)
                    shutil.copy(source_vi_image_name, target_vi_image_name)
                    shutil.copy(source_anno_name, target_anno_name)


        del annos, labels, ir_img, vi_img, x, y, w, h
        # save annotation video
        annos = sorted(glob(osp.join(out_anno_root, '*')))
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        att_ir_img = cv2.imread(lwir_imgs[0])
        out_video = cv2.VideoWriter(osp.join(out_data_root, f'{mode}.avi'), fourcc, 20.0, (640*2, 512))
        for anno in tqdm(annos, total=len(annos), desc=f'video writing. {mode}...'):
            head = ' '.join(np.loadtxt(anno, dtype=str, ndmin=1, max_rows=1))
            assert head == '% bbGt version=3', print(head)
            labels = np.loadtxt(anno, dtype=str, ndmin=2, skiprows=1)
            ir_img = cv2.imread(anno.replace(out_anno_root, out_irImg_root).replace('.txt', '.png'))
            vi_img = cv2.imread(anno.replace(out_anno_root, out_viImg_root).replace('.txt', '.png'))
            for label in labels:
                assert label[0] == 'person'
                x, y, w, h = list(map(int, label[1:5]))
                assert x >= 0 and y >= 0 and w > 0 and h > 0 and (x + w) <= 640 - 1 and (y + h) <= 512 - 1
                cv2.rectangle(ir_img, (x, y), (x + w, y + h), (0, 255, 0), 1)
                cv2.rectangle(vi_img, (x, y), (x + w, y + h), (0, 255, 0), 1)
            out_video.write(np.hstack([ir_img, vi_img]))
        out_video.release()
