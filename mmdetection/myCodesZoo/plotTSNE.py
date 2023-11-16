import os
os.environ['DISPLAY']='localhost:13.0'
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
from PIL import Image
import pandas as pd
import seaborn as sns
from sklearn import manifold
from tqdm import tqdm

matplotlib.rc('font', **{'family': 'normal', 'size': 18})

for i in range(2):
    filename = 'set08_V000_I02519'
    anno_file = os.path.join(
    '/home/zx/cross-modality-det/datasets/zx-sanitized-kaist-keepPerson-fillNonPerson/annotations/test/', filename+'.txt'
    )
    seed = 0
    gtBBoxes = np.loadtxt(anno_file, dtype=int, delimiter=' ', skiprows=1, usecols=[1, 2, 3, 4], ndmin=2)
    gtClasses = np.loadtxt(anno_file, dtype=str, delimiter=' ', skiprows=1, usecols=0, ndmin=1)
    assert len(gtClasses) == len(gtBBoxes) and np.unique(gtClasses) == 'person'

    gt = np.zeros((512, 640), dtype='uint8')
    for box in gtBBoxes:
        x, y, w, h = box
        gt[y:y+h, x:x+w] = 1
    if i == 0:
        fea_file = f'/home/zx/cross-modality-det/code/mmdetection/runs/FasterRCNN_vgg16_catConvCSPLayer/t-sne/{filename}_feature.npy'
    else:
        fea_file = f'/home/zx/cross-modality-det/code/mmdetection/runs/FasterRCNN_vgg16_channelRelation_dscSEFusion_similarityMax_1/t-sne/{filename}_feature.npy'
    feature = np.load(fea_file)
    title = fea_file.split('runs/')[-1].split('/t-sne')[0]
    feature = np.transpose(feature, [1, 2, 0])
    h, w, c = feature.shape
    print('Feature shape:', feature.shape)

    gt = Image.fromarray(gt)
    gt = np.asarray(gt.resize((w, h), Image.Resampling.NEAREST))
    print('GT shape:', gt.shape, 'Labels:', np.unique(gt))

    feature = feature.reshape((h*w, c))
    targets = gt.reshape(h*w).astype(int)

    idxes1 = np.where(targets==1)[0]
    idxes0 = np.where(targets==0)[0]
    np.random.seed(0)
    rdn = np.random.randint(0, len(idxes0), 5*len(idxes1), dtype=int)
    targets = np.concatenate([targets[idxes1], targets[idxes0[rdn]]])
    feature = np.concatenate([feature[idxes1], feature[idxes0[rdn]]])
    print(feature.shape)
    tsne = manifold.TSNE(n_components=2, random_state=seed, init='pca')
    transformed_data = tsne.fit_transform(feature)
    transformed_data[:, 0] = (transformed_data[:, 0] - np.min(transformed_data[:, 0])) / (np.max(transformed_data[:, 0]) - np.min(transformed_data[:, 0]))
    transformed_data[:, 1] = (transformed_data[:, 1] - np.min(transformed_data[:, 1])) / (np.max(transformed_data[:, 1]) - np.min(transformed_data[:, 1]))
    tsne_df = pd.DataFrame(
    np.column_stack((transformed_data, targets)),
    columns=["x", "y", "targets"]
    )
    tsne_df.loc[:, "targets"] = tsne_df.targets.astype(int)
    grid = sns.FacetGrid(tsne_df, hue="targets", height=8, palette=['b', 'r'])
    grid.map(plt.scatter, "x", "y")
    # grid.add_legend()
    plt.axis('off')
    ax = plt.gca()
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(True)
    # plt.title(title)
    plt.savefig(f'/home/zx/cross-modality-det/code/mmdetection/runs/FasterRCNN_vgg16_channelRelation_dscSEFusion_similarityMax_1/t-sne/comparisons/seed{seed}_{title.split("FasterRCNN_vgg16_")[1]}.svg',\
                bbox_inches='tight', pad_inches=0.0)