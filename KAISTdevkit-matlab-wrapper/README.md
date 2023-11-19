## Matlab Wrapper For KAIST Evaluation

Written by Chengyang Li based on the original demo code from Soonmin Hwang.

See demo_test.m for an example.

**Note**: We provide evalutaion results in terms of log-average miss rate as well as recall on all 9 different settings, i.e. Reasonable-all, Reasonable-day, Reasonable-night, Scale=near, Scale=medium, Scale=far, Occ=none, Occ=partial, Occ=heavy (see CVPR15 paper for details). Results are provided both using the origninal and the improved test annotaions and are printed in the format of "original value (improved value)". However we strongly suggest you report results using the improved test annotaions only to enable a reliable comparision.


## Acknowledgement
We thank Chengyang Li for providing the matlab evaluation code. If you find this useful, please cite their paper as well. 

**GitHub link: [https://github.com/Li-Chengyang/MSDS-RCNN/tree/master/lib/datasets/KAISTdevkit-matlab-wrapper](https://github.com/Li-Chengyang/MSDS-RCNN/tree/master/lib/datasets/KAISTdevkit-matlab-wrapper)**

**Citation**
	
	@InProceedings{li_2018_BMVC,
	author = {Li, Chengyang and Song, Dan and Tong, Ruofeng and Tang, Min},
	title = {Multispectral Pedestrian Detection via Simultaneous Detection and Segmentation},
	booktitle = {British Machine Vision Conference (BMVC)},
	month = {September}
	year = {2018}
	} 