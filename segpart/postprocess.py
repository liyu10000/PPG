import os
import cv2
import numpy as np
from .utils import read_txt


def gen_mask_from_lines(label_layout_dir, pred_line_dir, pred_mask_dir):
    os.makedirs(pred_mask_dir, exist_ok=True)
    names = [f[:-4] for f in os.listdir(pred_line_dir) if f.endswith('.txt')]
    for name in names:
        print('processing', name)
        whole_mask = cv2.imread(os.path.join(label_layout_dir, name+'.png')) # cropped, 640x480
        txt_content = read_txt(os.path.join(label_layout_dir, name+'.txt'))
        pred_bon = np.array(read_txt(os.path.join(pred_line_dir, name+'.txt'))['bon']) # 640x2
        
        # generate parts seg image
        h, w = txt_content['h'], txt_content['w']
        for i in range(w):
            h1, h2 = pred_bon[i]
            h1 = int(h1) if h1 > 0 else 0
            h2 = max(h1, int(h2)) # h2 should be at least no smaller than h1
            whole_mask[h1:, i, 0] = 0 # TS
            whole_mask[:h1, i, 1] = 0 # BT
            whole_mask[h2:, i, 1] = 0 # BT
            whole_mask[:h2, i, 2] = 0 # VS
        
        # resize parts seg and put back to the original image
        H, W = txt_content['H'], txt_content['W']
        h1, w1 = txt_content['h1'], txt_content['w1']
        h2, w2 = txt_content['h2'], txt_content['w2']
        part_mask = cv2.resize(whole_mask, (w2-w1+1, h2-h1+1))
        whole_mask = np.zeros((H, W, 3), dtype=part_mask.dtype)
        whole_mask[h1:h2+1, w1:w2+1] = part_mask
        # save mask
        cv2.imwrite(os.path.join(pred_mask_dir, name+'.png'), whole_mask)