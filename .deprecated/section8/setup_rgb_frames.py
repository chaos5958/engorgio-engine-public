import os
import argparse
import glob
import json
from PIL import Image
import numpy as np

def save_png(video_path, image_dir):
    os.makedirs(image_dir, exist_ok=True)
    cmd = f"/usr/bin/ffmpeg -i {video_path} -vf fps=1 -ss 00:00:00 -to 00:01:00 {image_dir}/%d.png"
    os.system(cmd)


def png_to_rgb(image_dir):
    png_files = sorted(glob.glob(os.path.join(image_dir, "*.png")))
    for png_file in png_files:
        png_img = Image.open(png_file)
        np_img = np.array(png_img)
        raw_img = os.path.join(image_dir, os.path.basename(png_file).split('.')[0] + '.rgb')
        np_img.tofile(raw_img)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/workspace/research/engorgio/dataset')
    parser.add_argument('--result_dir', type=str, default='/workspace/research/engorgio/result')
    parser.add_argument('--content', type=str, default='lol')
    args = parser.parse_args()

    video_dir = os.path.join(args.data_dir, 'lol0', 'video')
    video_name = '2160p_d600.webm'
    video_path = os.path.join(video_dir, video_name)
    image_dir = os.path.join(args.data_dir, 'lol0', 'image')

    save_png(video_path, image_dir)
    png_to_rgb(image_dir)
