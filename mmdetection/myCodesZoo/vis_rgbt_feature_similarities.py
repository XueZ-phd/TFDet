import os
os.environ['DISPLAY'] = 'localhost:10.0'
import matplotlib
matplotlib.use('TkAgg')
import numpy as np
import os.path as osp
import matplotlib.pyplot as plt
from glob import glob
from tqdm import tqdm

np_file_root = '/home/zx/cross-modality-det/code/mmdetection/runs_llvip/FasterRCNN_r50wMask_ROIFocalLoss5_CIOU20_cosineSE_notDetach_negEntropy1/rgbt_feature_cosine_similarities/scale0'
# np_file_root = '/home/zx/cross-modality-det/code/mmdetection/runs/FasterRCNN_vgg16_channelRelation_dscSEFusion_similarityMax_1/rgbt_feature_cosine_similarities/scale1'
nps = sorted(glob(osp.join(np_file_root, '*.npy')))[:1000]
max_ratios = []
mean_ratios = []
medium_ratios = []
diags = []
non_diags = []

c = 512
all_sim = np.zeros((c, c))
pbar = tqdm(total=len(nps)*c, initial=1)
for f in nps:
    sim = np.load(f)
    all_sim += sim
    for j in range(c):
        pbar.update(1)
        dig = sim[j, j]
        non_dig = sim[j, :].tolist().copy()
        non_dig.pop(j)
        assert len(non_dig) == c-1
        max_ratios.append(dig / (np.max(non_dig)+1e-7))
        mean_ratios.append(dig / (np.mean(non_dig)+1e-7))
        medium_ratios.append(dig / (np.median(non_dig)+1e-7))
        diags.append(np.abs(dig))
        non_diags.append(np.mean(np.abs(non_dig)))
pbar.close()
print('mean max ratio:', np.mean(np.clip(max_ratios, 0.0, 20.0)))
print('mean mean ratio:', np.mean(np.clip(mean_ratios, 0.0, 20.0)))
print('mean meadium ratio:', np.mean(np.clip(medium_ratios, 0.0, 20.0)))
print('diags mean:', np.mean(np.clip(diags, 0.0, 20.0)))
print('non_diags mean:', np.mean(np.clip(non_diags, 0.0, 20.0)))
# plt.imshow((all_sim-np.min(all_sim))/(np.max(all_sim)-np.min(all_sim)))
# plt.show()