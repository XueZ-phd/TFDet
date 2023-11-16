import numpy as np
import torch


def bayesian_fusion_multiclass(match_score_vec, pred_class):
    scores = np.zeros((match_score_vec.shape[0], 4))
    scores[:, :3] = match_score_vec
    scores[:, -1] = 1 - np.sum(match_score_vec, axis=1)
    log_scores = np.log(scores)
    sum_logits = np.sum(log_scores, axis=0)
    exp_logits = np.exp(sum_logits)
    out_score = exp_logits[pred_class] / np.sum(exp_logits)
    return out_score


def weighted_box_fusion(bbox, score):
    weight = score / np.sum(score)
    out_bbox = np.zeros(4)
    for i in range(len(score)):
        out_bbox += weight[i] * bbox[i]
    return out_bbox


def nms_bayesian(det1, det2, scores, classes, thresh, method):
    dets = torch.cat([det1, det2], 0).detach().cpu().numpy()

    x1 = dets[:, 0] + classes * 640
    y1 = dets[:, 1] + classes * 512
    x2 = dets[:, 2] + classes * 640
    y2 = dets[:, 3] + classes * 512

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    match_scores = []
    match_bboxs = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        match = np.where(ovr > thresh)[0]
        match_ind = order[match + 1]

        match_score = list(scores[match_ind])
        match_bbox = list(dets[match_ind][:, :4])

        original_score = scores[i].tolist()
        original_bbox = dets[i][:4]

        # If some boxes are matched
        if len(match_score) > 0:
            match_score += [original_score]
            # Try with different fusion methods
            final_score = bayesian_fusion_multiclass(np.asarray(match_score), classes[i])
            match_bbox += [original_bbox]
            final_bbox = weighted_box_fusion(match_bbox, match_score)

            match_scores.append(final_score)
            match_bboxs.append(final_bbox)
        else:
            match_scores.append(original_score)
            match_bboxs.append(original_bbox)

        order = order[inds + 1]

    assert len(keep) == len(match_scores)
    assert len(keep) == len(match_bboxs)

    match_bboxs = match_bboxs
    match_scores = torch.Tensor(match_scores)

    return match_scores, match_bboxs

