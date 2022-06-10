import glob
import os 
from PIL import Image


if __name__ == "__main__":
    # img_dir = "/workspace/research/engorgio-dataset/game1/image/720p_3960kbps_s960_d60.webm/key"
    img_dir = "/workspace/research/engorgio-engine/build/controller/test"

    paths = glob.glob(os.path.join(img_dir, "*.rgb"))
    size = (1280, 720)
    for path in paths:
        with open(path, 'rb') as f:
            raw = f.read()
        name = os.path.basename(path).split('.')[0]
        img = Image.frombytes('RGB', size, raw)
        # b, g, r = img.split()
        # img = Image.merge("RGB", (r, g, b))
        img.save(os.path.join(img_dir, '{}_yuv2rgb.png'.format(name)))

    # paths = glob.glob(os.path.join(img_dir, "*.y"))
    # size = (1280, 720)
    # for path in paths:
    #     with open(path, 'rb') as f:
    #         raw = f.read()
    #     name = os.path.basename(path).split('.')[0]
    #     img = Image.frombytes('L', size, raw)
    #     img.save(os.path.join(img_dir, '{}_y.png'.format(name)))

    # paths = glob.glob(os.path.join(img_dir, "*.u"))
    # size = (640, 360)
    # for path in paths:
    #     with open(path, 'rb') as f:
    #         raw = f.read()
    #     name = os.path.basename(path).split('.')[0]
    #     img = Image.frombytes('L', size, raw)
    #     img.save(os.path.join(img_dir, '{}_u.png'.format(name)))

    # paths = glob.glob(os.path.join(img_dir, "*.v"))
    # size = (640, 360)
    # for path in paths:
    #     with open(path, 'rb') as f:
    #         raw = f.read()
    #     name = os.path.basename(path).split('.')[0]
    #     img = Image.frombytes('L', size, raw)
    #     img.save(os.path.join(img_dir, '{}_v.png'.format(name)))