import os
import cv2
import argparse
import numpy as np


def joint(patch_dir, save_dir):
    """ Joint mask patches into big masks
    Assume step_size = patch_size
    @params
        patch_dir: directory with patches of predicted masks
        save_dir: directory to save big masks
    """
    # collect mask names associated with one image
    file_dict = {}
    files = os.listdir(patch_dir)
    for f in files:
        tokens = f[:-4].rsplit('_', 4)
        basename = tokens[0]
        H = int(tokens[1][1:])
        W = int(tokens[2][1:])
        h = int(tokens[3][1:])
        w = int(tokens[4][1:])
        if basename not in file_dict:
            file_dict[basename] = {'H':H, 'W':W, 'patches':[]}
        file_dict[basename]['patches'].append([h, w, f])
    # put patches back into a big image
    os.makedirs(save_dir, exist_ok=True)
    for basename in file_dict:
        save_name = os.path.join(save_dir, basename+'.png')
        H, W = file_dict[basename]['H'], file_dict[basename]['W']
        mask = np.zeros((H, W, 3))
        patches = file_dict[basename]['patches']
        for h, w, f in patches:
            patch_name = os.path.join(patch_dir, f)
            patch = cv2.imread(patch_name)
            ph, pw, _ = patch.shape
            mask[h:h+ph, w:w+pw] = patch
        cv2.imwrite(save_name, mask)
        print('Finished jointing', basename)

def joint2(patch_dir, save_dir):
    """ Joint mask patches into big masks
    Assume step_size = patch_size
    @params
        patch_dir: directory with patches of predicted masks
        save_dir: directory to save big masks
    """
    # collect mask names associated with one image
    file_dict = {}
    files = os.listdir(patch_dir)
    for f in files:
        tokens = f[:-4].rsplit('_', 4)
        basename = tokens[0]
        H = int(tokens[1][1:])
        W = int(tokens[2][1:])
        h = int(tokens[3][1:])
        w = int(tokens[4][1:])
        if basename not in file_dict:
            file_dict[basename] = {'H':H, 'W':W, 'patches':[]}
        file_dict[basename]['patches'].append([h, w, f])
    # put patches back into a big image
    os.makedirs(save_dir, exist_ok=True)
    for basename in file_dict:
        save_name = os.path.join(save_dir, basename+'.png')
        H, W = file_dict[basename]['H'], file_dict[basename]['W']
        mask = np.zeros((H, W, 1))
        pred_cnt = np.zeros((H, W, 1))
        patches = file_dict[basename]['patches']
        for h, w, f in patches:
            patch_name = os.path.join(patch_dir, f)
            patch = np.load(patch_name)
            patch = patch.transpose((1, 2, 0))
            ph, pw, _ = patch.shape
            mask[h:h+ph, w:w+pw] += patch
            pred_cnt[h:h+ph, w:w+pw] += 1
        mask = np.nan_to_num(mask / pred_cnt, nan=0, posinf=0, neginf=0)
        mask = ((mask > 0.5) * 255).astype(int)
        cv2.imwrite(save_name, mask)
        print('Finished jointing', basename)

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
    union = np.where(true_mask|pred_mask, 1, 0)
    true_mask_b = np.sum(true_mask, axis=2)
    pred_mask_b = np.sum(pred_mask, axis=2)
    intersect_b = np.sum(intersect, axis=2)
    union_b = np.sum(union, axis=2)
    
    try:
        iou = np.count_nonzero(intersect_b) / np.count_nonzero(union_b)
        precision = np.count_nonzero(intersect_b) / np.count_nonzero(pred_mask_b)
        recall = np.count_nonzero(intersect_b) / np.count_nonzero(true_mask_b)
    except:
        iou = 1.0
        precision = 1.0
        recall = 1.0
    # calculate per channel recall
    ch_recalls = []
    for i in range(3):
        if np.count_nonzero(true_mask[:,:,i]) != 0:
            r = np.count_nonzero(intersect[:,:,i]) / np.count_nonzero(true_mask[:,:,i])
        else:
            r = 1.0
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
                p = 1.0
            ch_precisions.append(p)
            iou = np.count_nonzero(intersect[:,:,i]) / np.count_nonzero(union[:,:,i])
            ch_ious.append(iou)
    return iou, precision, recall, ch_recalls, ch_precisions, ch_ious

def evaluate(true_dir, pred_dir, same_channel=False, whole_dir=None):
    true_masks = os.listdir(true_dir)
    pred_masks = os.listdir(pred_dir)
    print('# true files in {}:{}\n# pred files in {}:{}'.format(true_dir, len(true_masks), pred_dir, len(pred_masks)))
    pred_masks.sort()
    IoU, precision, recall = [], [], []
    ch_recalls, ch_precisions = [], []
    ch_ious = []
    for name in pred_masks:
        if not os.path.isfile(os.path.join(true_dir, name)):
            continue
        t = cv2.imread(os.path.join(true_dir, name))
        p = cv2.imread(os.path.join(pred_dir, name))
        if whole_dir is not None: # calculate ratio of defects over ship surface
            w = cv2.imread(os.path.join(whole_dir, name))
            t_b = (np.sum(t, axis=2) > 0).astype(np.uint8)
            p_b = (np.sum(p, axis=2) > 0).astype(np.uint8)
            w_b = (np.sum(w, axis=2) > 0).astype(np.uint8)
            t_b_in_w = np.where(w_b == 1, t_b, 0)
            p_b_in_w = np.where(w_b == 1, p_b, 0)
            t_ratio = np.sum(t_b_in_w) / np.sum(w_b)
            p_ratio = np.sum(p_b_in_w) / np.sum(w_b)
            print('name+tpratio {} {:.4f} {:.4f}'.format(name, t_ratio, p_ratio))
        iou, pr, rc, ch_rcs, ch_prs, ious = _evaluate(t, p, same_channel)
        print('name+chn+recall {} {:.4f} {:.4f} {:.4f}'.format(name, *ch_rcs))
        print('name+iou+p+r+f1 {} {:.4f} {:.4f} {:.4f} {:.4f}'.format(name, iou, pr, rc, 2*pr*rc/(pr+rc+0.01)))
        IoU.append(iou)
        precision.append(pr)
        recall.append(rc)
        ch_recalls.append(ch_rcs)
        ch_precisions.append(ch_prs)
        ch_ious.append(ious)
    IoU = np.mean(IoU)
    precision = np.mean(precision)
    recall = np.mean(recall)
    ch_recalls = np.mean(ch_recalls, axis=0)
    print('iou: {:.4f}, precision: {:.4f}, recall: {:.4f}'.format(IoU, precision, recall))
    for r in ch_recalls:
        print('channel recall: {:.4f}'.format(r))
    if same_channel:
        for p, iou in zip(ch_precisions, ch_ious):
            print('channel precision: {:.4f}, channel iou: {:.4f}'.format(p, iou))


if __name__ == '__main__':
    # work by passing arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--patch_dir', type=str, default='path/to/patches/for/joint')
    parser.add_argument('--joint_dir', type=str, default='path/to/dir/saving/joint/images')
    parser.add_argument('--pred_type', type=str, default='png', choices=['png', 'npy'])
    parser.add_argument('--true_dir', type=str, default='path/to/true/images')
    parser.add_argument('--pred_dir', type=str, default='path/to/pred/images')
    parser.add_argument('--whole_dir', type=str, default='path/to/whole/images')
    cfg = parser.parse_args()

    # joint patches back to big images
    patch_dir = cfg.patch_dir
    joint_dir = cfg.joint_dir
    pred_type = cfg.pred_type
    if not patch_dir.startswith('path'):
        if pred_type == 'png':
            joint(patch_dir, joint_dir)
        elif pred_type == 'npy':
            joint2(patch_dir, joint_dir)

    # calculate precision and recall
    true_dir = cfg.true_dir
    pred_dir = cfg.pred_dir
    same_channel = False
    if not true_dir.startswith('path'):
        whole_dir = cfg.whole_dir
        if whole_dir.startswith('path'):
            evaluate(true_dir, pred_dir, same_channel)
        else:
            evaluate(true_dir, pred_dir, same_channel, whole_dir)
