<div align='left'>
<img src='/misc/scene1.jpg' height=115cm />
<img src='/misc/scene2.jpg' height=115cm />
<img src='/misc/scene3.png' height=115cm />
</div>

# TFDet: Target-aware Fusion for RGB-T Pedestrian Detection

<div align='left'>
<img src='/misc/fusionMethodAndFeatureComparison.png' width=90% />
</div>

**This is the official repository for our paper "TFDet: Target-aware Fusion for RGB-T Pedestrian Detection" (<u>[arxiv paper link](https://arxiv.org/abs/2305.16580)</u>).**

**Our main contributions are summarized as follows:**

- We comprehensively analyze the adverse impact of false positives on detection performance and identify the noisy feature map as a key factor contributing to these false positives.
- To address the noisy feature map issue, we propose the target-aware fusion strategy, which can effectively fuse complementary features from both modalities and highlight the feature representation in pedestrian areas while suppressing the representations in the background areas.
- Experiments show that our TFDet generates discriminative feature maps, significantly reducing false positives. Our TFDet achieves state-of-the-art performance on two challenging multi-spectral pedestrian detection benchmarks: KAIST and LLVIP. Additionally, it is computationally effective and has a comparable inference time to previous state-of-the-art approaches.
- Notably, TFDet performs especially well under challenging night scenes.


<h2 id="1">Dataset and Models</h2>
- Datasets and model checkpoints can be downloaded from this <u>[cloud link](https://pan.baidu.com/s/1ut90qAY9RAo34EHo52hRUw)</u>, extractor code: `tfde`.
- Since the KAIST dataset has been updated by several previous works, such as <u>[Hwang et al.](https://openaccess.thecvf.com/content_cvpr_2015/papers/Hwang_Multispectral_Pedestrian_Detection_2015_CVPR_paper.pdf)</u>, <u>[Li et al.](https://github.com/Li-Chengyang/MSDS-RCNN)</u>, <u>[Liu et al.](https://arxiv.org/pdf/1611.02644.pdf)</u>, and <u>[Zhang et al.](https://openaccess.thecvf.com/content_ICCV_2019/papers/Zhang_Weakly_Aligned_Cross-Modal_Learning_for_Multispectral_Pedestrian_Detection_ICCV_2019_paper.pdf)</u>, we upload this dataset for your convenience in using our code.
- The LLVIP dataset can be downloaded from its official repository.


**In the <u>[cloud link](https://pan.baidu.com/s/1ut90qAY9RAo34EHo52hRUw)</u>, files are organized as follow:**

	TFDet----------------------------------
		|--yolov5-master
			|--runs								# For yolov5-llvip
				|--train
					|--modifiedDCN_MaskSup_negCorr_1024
						|--weights
							|--best.pt
		|--mmdetection
			|--runs_llvip						# For faster-rcnn-r50-llvip
				|--FasterRCNN_r50wMask_ROIFocalLoss5_CIOU20_cosineSE_dcnGWConvGlobalCC_1024x1280
					|--epoch_7.pth
					|--20230825_171907.log
					|--20230825_171907.log.json
			|--runs								# For faster-rcnn-v16-kaist
				|--FasterRCNN_vgg16_channelRelation_dscSEFusion_similarityMax_1
					|--epoch_
						|--epoch_3-test-all.txt
						|--epoch_3-test-day.txt
						|--epoch_3-test-night.txt
					|--epoch_3.pth
					|--epoch_3.pkl
		|--datasets
			|--kaist
				|--zx-sanitized-kaist-keepPerson-fillNonPerson
					|--images
						|--test.zip
						|--train_lwir.zip
						|--train_visible.zip
					|--annotations.zip
					|--coco_format.zip
					|--test.avi
					|--train.avi
			|--LLVIP
				|--LLVIP
					|--coco_format
					|--lwir
					|--visible
				|--yolov5_format
					|--images
						|--lwir
							|--train
							|--test
						|--visible
							|--train
							|--test
					|--labels
						|--lwir
							|--train
							|--test
						|--visible
							|--train
							|--test
	


## KAIST
### Environmental Requirements
- We use Faster R-CNN implemented by the [MMDetection](https://github.com/open-mmlab/mmdetection) toolbox to detect pedestrians on the KAIST dataset. Please follow the [MMDetection](https://github.com/open-mmlab/mmdetection) documents to install environments.

In our environment, we use:
	
	python==3.7.13
	torch==1.10.1+cu111
	torchvision==0.11.2+cu111
	mmcv==1.6.0
	mmdet==2.24.1
	 

### Dataset
- Please download the KAIST dataset and checkpoint from the above cloud link, and save them following the structure shown in the **[Dataset and Models](#1)** section.
- If you solely intend to assess the inference results, feel free to download the following files: ``images/test.zip``, ``annotations.zip`` and ``coco_format.zip``.

### Inference
Since the KAIST dataset is evaluated by the `log-average miss rate` metric, we should run three files: ``mmdetection/tools/test.py``, ``mmdetection/myCodesZoo/cvtpkl2txt_person.py``, and ``KAISTdevkit-matlab-wrapper/demo_test.m``.

We should first run the ``tools/test.py`` to generate the detection results.
	
	cd mmdetection

	# generate detection result in pkl format
	python tools/test.py configs/faster_rcnn/faster_rcnn_vgg16_fpn_sanitized-kaist_v5.py runs/FasterRCNN_vgg16_channelRelation_dscSEFusion_similarityMax_1/epoch_3.pth --work-dir runs/FasterRCNN_vgg16_channelRelation_dscSEFusion_similarityMax_1 --gpu-id 7 --eval bbox --out runs/FasterRCNN_vgg16_channelRelation_dscSEFusion_similarityMax_1/epoch_3.pkl

A few minutes later, you will obtain one pkl file named ``runs/FasterRCNN_vgg16_channelRelation_dscSEFusion_similarityMax_1/epoch_3.pkl``. Then, run the ``myCodesZoo/cvtpkl2txt_person.py`` to parse the pkl file

	cd myCodesZoo
	
	# convert detections to txt format
	python cvtpkl2txt_person.py

Then, you will obtain a folder ``runs/FasterRCNN_vgg16_channelRelation_dscSEFusion_similarityMax_1/epoch_``, which includes ``epoch_3-test-all.txt``, ``epoch_3-test-day.txt``, ``epoch_3-test-night.txt``. Meanwhile, we use the python code provided by [MLPD](https://github.com/sejong-rcv/MLPD-Multi-Label-Pedestrian-Detection/tree/main/evaluation_script) to compute the ``log-average miss rate``, and get a state-of-the-art performance: ``MR_all: 4.37, MR_day: 5.08, and MR_night: 3.36``， which is significantly better than previous state-of-the-art approaches. Nevertheless, we use the commomly-used matlab code to evaluate the ``log-average miss rate`` for fair comparisons with other approaches.

	cd ../../KAISTdevkit-matlab-wrapper

	run demo_test.m
 
Finally, you will get the result: ``MR_all: 4.47, MR_day: 5.22, and MR_night: 3.36``

<div align='left'>
<img src='misc/fppi_wrt_mr_curve.png' />
</div>

## LLVIP
### Environmental Requirements
We use the official repositories of [MMDetection](https://github.com/open-mmlab/mmdetection) and [YOLOv5 official repository](https://github.com/ultralytics/yolov5) in our experiments. Please configure your environment following the official documentation. For MMDetection, we employ the same environment as in the experiments on the KAIST dataset. For YOLOv5, some dependencies in our environment include:
	
	python==3.7.16
	torch==1.12.1
	torchvision==0.13.1

### Dataset
- Please download the LLVIP dataset from this [link](https://github.com/bupt-ai-cz/LLVIP/blob/main/previous%20annotations.md).

- For MMDetection, you should convert the annotation into coco-format following this [document](https://mmdetection.readthedocs.io/en/latest/user_guides/train.html#train-with-customized-datasets).
- For YOLOv5, you should convert the annotation into yolo-format follow this [document](https://docs.ultralytics.com/yolov5/tutorials/train_custom_data/).

### Inference
For MMDetection, we evaluate TFDet at two resolutions. Since the LLVIP dataset have a large number of multispectral images, we use distributed inference code. 

	# 640 x 512 resolution
	CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8 PORT=29500 bash tools/zx_dist_test_llvip_640x512.sh 8 --eval bbox
	# 1280 x 1024 resolution
	CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8 PORT=29500 bash tools/zx_dist_test_llvip_1024x1280.sh 8 --eval bbox
Then, you will get:

	|Resolution|AP.50|AP|
	|:---------|:---|:--|
	|640 x 512|95.7|56.1|
	|1280 x 1024|96.0|59.4|

For YOLOv5, 

	python val.py --device 0 --data LLVIP.yaml --weights runs/train/modifiedDCN_MaskSup_negCorr_1024/weights/best.pt --batch-size 32 --img 1024 --conf-thres 0.008 --iou-thres 0.4 --exist-ok
	
Finally, you will get:
	
	|AP.50|AP.75|AP|
	|:----|:----|:--|
	|97.9|83.4|71.1|


## Citation
If you find our TFDet useful, please cite this paper:

	@article{tfdet,
	title={TFDet: Target-aware Fusion for RGB-T Pedestrian Detection},
	author={Zhang, Xue and Zhang, Xiaohan and Sheng, Zehua and Shen, Hui-Liang},
	journal={arXiv preprint arXiv:2305.16580},
	year={2023}}
