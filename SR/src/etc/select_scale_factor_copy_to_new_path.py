import os
import shutil

scale_factor = 4
src = "E:/PPG/results-PPGScrub/"
dst = "E:/PPG/results-PPGScrub-splitted/x{}/".format(scale_factor)

os.makedirs(dst, exist_ok=True)
for root, dirs, files in os.walk(src):
    for file in files:
        if "x{}".format(scale_factor) in file.split("_")[-2]:
            base = os.path.basename(root)
            print("{} ||| {}".format(file, base))
            dst_path = os.path.join(dst, base)
            os.makedirs(dst_path, exist_ok=True)
            shutil.copy2(os.path.join(root, file), dst_path)