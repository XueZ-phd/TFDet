from  glob import glob
import os.path as osp
import numpy as np

annos_root = '/UsrFile/Usr3/zx/KAIST-Sanitized/annotations'
train_annos = osp.join(annos_root, 'train', 'sanitized_annotations', 'sanitized_annotations')
val_annos = osp.join(annos_root, 'test', 'annotations_KAIST_test_set')
assert osp.isdir(train_annos) and osp.isdir(val_annos)

# train annotations statics
train_anno_names = sorted(glob(osp.join(train_annos, '*.txt')))
train_labels_dict = dict()
for train_name in train_anno_names:
    crt_labels = np.loadtxt(train_name, str, skiprows=1, usecols=0, ndmin=1)
    if crt_labels.size == 0:
        continue
    for l in crt_labels:
        if l not in list(train_labels_dict.keys()):
            train_labels_dict[l] = 0
        train_labels_dict[l]+=1

# val annotations statics
val_anno_names = sorted(glob(osp.join(val_annos, '*.txt')))
val_labels_dict = dict()
for val_name in val_anno_names:
    crt_labels = np.loadtxt(val_name, str, skiprows=1, usecols=0, ndmin=1)
    if crt_labels.size == 0:
        continue
    for l in crt_labels:
        if l not in list(val_labels_dict.keys()):
            val_labels_dict[l] = 0
        val_labels_dict[l]+=1

print(f'train statics:\n{train_labels_dict}')
print(f'val statics:\n{val_labels_dict}')