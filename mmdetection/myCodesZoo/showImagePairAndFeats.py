import shutil

import matplotlib.pyplot as plt
import torchvision
import numpy as np
import os.path as osp
import os

def reN(x): return  (x-np.min(x))/(np.max(x)-np.min(x))

def save_img_and_feats(rgb_img, lwir_img, rgb_x, lwir_x, taf_x, fpn_x, save_dir):

    save_dir = osp.join(save_dir, 'feats')
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    os.makedirs(save_dir, exist_ok=True)
    # image
    np_rgb = reN(rgb_img[0].permute(1, 2, 0).detach().cpu().numpy())
    np_lwir = reN(lwir_img[0].permute(1, 2, 0).detach().cpu().numpy())

    plt.imsave(f'{osp.join(save_dir, "img_rgb.png")}', np_rgb, vmin=0.0, vmax=1.0)
    plt.imsave(f'{osp.join(save_dir, "img_lwir.png")}', np_lwir, vmin=0.0, vmax=1.0)

    # backbone feats
    for name in ['rgb_x', 'lwir_x', 'taf_x', 'fpn_x']:
        feats = eval(name)
        for f_idx, feat in enumerate(feats):
            maps = torchvision.utils.make_grid(feat.permute(1, 0, 2, 3), 30, normalize=True, pad_value=0.5).permute(1, 2, 0).detach().cpu().numpy()
            plt.imsave(f'{osp.join(save_dir, f"feat_{name}{f_idx}.png")}', maps)
