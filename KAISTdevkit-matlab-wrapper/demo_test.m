function demo_test()
clear; 
clc;
for epoch=3
    % mmdetection
    dtDir = sprintf('%s%d','../mmdetection/runs/FasterRCNN_vgg16_channelRelation_dscSEFusion_similarityMax_1/epoch_/epoch_', epoch);
    %% specify path of groundtruth annotaions
    gtDir = './annotations_KAIST_test_set';
%     gtDir = 'E:\pyDemo\cross-modality-det\mmdetection\runs_llvip\FasterRCNN_r50wMask_ablation_onlyFFM_ROIFocalLoss5_CIOU20_cosineSE_dcnGWConvGlobalCC_640x512';
    %% evaluate detection results
    kaist_eval_full(dtDir, gtDir, false, false);
end
end
