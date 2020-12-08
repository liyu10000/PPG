import os
import cv2
import numpy as np


def resize(image_dir, save_dir, wh=(640,480)):
    os.makedirs(save_dir, exist_ok=True)
    names = [f for f in os.listdir(image_dir) if f.endswith('.png')]
    for name in names:
        image_name = os.path.join(image_dir, name)
        save_name = os.path.join(save_dir, name)
        img = cv2.imread(image_name)
        img = cv2.resize(img, wh)
        cv2.imwrite(save_name, img)