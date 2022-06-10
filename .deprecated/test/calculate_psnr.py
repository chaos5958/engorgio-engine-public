#!/usr/bin/env python
import sys
from PIL import Image
import numpy as np
def calculate_psnr(img1, img2, max_value=255):
    """"Calculating peak signal-to-noise ratio (PSNR) between two images."""
    mse = np.mean((np.array(img1, dtype=np.float32) - np.array(img2, dtype=np.float32)) ** 2)
    if mse == 0:
        return 100
    return 20 * np.log10(max_value / (np.sqrt(mse)))

if __name__ == "__main__":
    print(calculate_psnr(np.asarray(Image.open(sys.argv[1])), np.asarray(Image.open(sys.argv[2]))))