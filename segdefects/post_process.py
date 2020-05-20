import os
import cv2
import numpy as np


def _evaluate(true_mask, pred_mask, same_channel):
    """ Calculate the overall precision, recall, per channel recall,
    and per channel precision and iou if pred_mask has the equivalent channels.
    @params
        true_mask: ground truth mask, HxWx3
        pred_mask: predicted mask, HxWx3
        same_channel: if pred_mask has the same channels as true_mask (if pred_mask is grayscale)
    """
    # same_channel = (len(true_mask.shape) == len(pred_mask.shape))
    # calculate overall precision and recall
    intersect = np.where(true_mask&pred_mask, 1, 0)
    true_mask_b = np.sum(true_mask, axis=2)
    pred_mask_b = np.sum(pred_mask, axis=2)
    intersect_b = np.sum(intersect, axis=2)
    precision = np.count_nonzero(intersect_b) / np.count_nonzero(pred_mask_b)
    recall = np.count_nonzero(intersect_b) / np.count_nonzero(true_mask_b)
    # calculate per channel recall
    ch_recalls = []
    for i in range(3):
        if np.count_nonzero(true_mask[:,:,i]) != 0:
            r = np.count_nonzero(intersect[:,:,i]) / np.count_nonzero(true_mask[:,:,i])
        else:
            r = 0
        ch_recalls.append(r)
    # if pred_mask comes in RGB, calculate per channel precision and iou
    ch_precisions = []
    ch_ious = []
    if same_channel:
        union = np.where(true_mask|pred_mask, 1, 0)
        for i in range(3):
            if np.count_nonzero(pred_mask[:,:,i]) != 0:
                p = np.count_nonzero(intersect[:,:,i]) / np.count_nonzero(pred_mask[:,:,i])
            else:
                p = 0
            ch_precisions.append(p)
            iou = np.count_nonzero(intersect[:,:,i]) / np.count_nonzero(union[:,:,i])
            ch_ious.append(iou)
    return precision, recall, ch_recalls, ch_precisions, ch_ious

def evaluate(true_dir, pred_dir, same_channel=False):
    true_masks = os.listdir(true_dir)
    pred_masks = os.listdir(pred_dir)
    print('# true files in {}:{}\n# pred files in {}:{}'.format(true_dir, len(true_masks), pred_dir, len(pred_masks)))
    true_masks.sort()
    pred_masks.sort()
    precision, recall = [], []
    ch_recalls, ch_precisions = [], []
    ch_ious = []
    for name in pred_masks:
        t = cv2.imread(os.path.join(true_dir, name))
        p = cv2.imread(os.path.join(pred_dir, name))
        pr, rc, ch_rcs, ch_prs, ious = _evaluate(t, p, same_channel)
        precision.append(pr)
        recall.append(rc)
        ch_recalls.append(ch_rcs)
        ch_precisions.append(ch_prs)
        ch_ious.append(ious)
    precision = np.mean(precision)
    recall = np.mean(recall)
    ch_recalls = np.mean(ch_recalls, axis=0)
    print('precision: {:.4f}, recall: {:.4f}'.format(precision, recall))
    for r in ch_recalls:
        print('channel recall: {:.4f}'.format(r))
    if same_channel:
        for p, iou in zip(ch_precisions, ch_ious):
            print('channel precision: {:.4f}, channel iou: {:.4f}'.format(p, iou))


if __name__ == '__main__':
    true_dir = '../datadefects/highquality/labels'
    pred_dir = '../datadefects/exps/exp5/bce_dice_5highq_joint'
    same_channel = False
    evaluate(true_dir, pred_dir, same_channel)