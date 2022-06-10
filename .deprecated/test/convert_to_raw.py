from PIL import Image
import numpy as np
import glob
import os
import sys
target_dir = "/workspace/research/frames"

if __name__ == '__main__':
    pngs = sorted(glob.glob(os.path.join(target_dir, "*.png")))
    for png in pngs:
        pil_img = Image.open(png)
        png_name = os.path.basename(png)
        raw_name = png_name.split('.')[0] + ".raw"
        np_img = np.array(pil_img)
        np_img.tofile(os.path.join(target_dir, raw_name))
