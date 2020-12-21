"""
This script is used for transform a gray image to a rgba img based on 
the color standard of SZ weather bureau.
Control flow should be specified.
"""
import scipy.misc
from PIL import Image
import numpy as np
import time
from data_provider.CIKM.map_of_color import color_map
from concurrent.futures import ThreadPoolExecutor, wait
import os
from multiprocessing import Pool


color = {
    0: [0, 0, 0, 0],
    1: [0, 236, 236, 255],
    2: [1, 160, 246, 255],
    3: [1, 0, 246, 255],
    4: [0, 239, 0, 255],
    5: [0, 200, 0, 255],
    6: [0, 144, 0, 255],
    7: [255, 255, 0, 255],
    8: [231, 192, 0, 255],
    9: [255, 144, 2, 255],
    10: [255, 0, 0, 255],
    11: [166, 0, 0, 255],
    12: [101, 0, 0, 255],
    13: [255, 0, 255, 255],
    14: [153, 85, 201, 255],
    15: [255, 255, 255, 255],
    16: [0, 0, 0, 0]
}

gray_cursor = [-1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 1000]
_imread_executor_pool = ThreadPoolExecutor(max_workers=16)



def mapping(img):
    """Map each gray level pixel in origin image to RGBA space
    Parameter
    ---------
    img : ndarray (a gray level image)

    Returns
    ---------
    img : An Image object with RGBA mode

    """
    # color_map = form_color_map()
    # print('123')
    # print(img.shape)
    h, w = img.shape
    new_img = np.zeros((h, w, 4), dtype=np.int8)
    for i in range(h):
        for j in range(w):
            new_img[i, j] = color_map[img[i, j]]
    img = new_img
    img = Image.fromarray(img, mode="RGBA")
    return img


def form_color_map():
    """
    This function forms a color map due to the pseudo color segmentation 
    given by weather bureau.
    :return: 
    """
    color_map = {}
    cursor = 0
    for i in range(256):
        while not gray_cursor[cursor + 1] > i >= gray_cursor[cursor]:
            cursor += 1
        color_map[i] = color[cursor]
    print("color map")
    return color_map


def transfer(img_path, des_path):
    """Transform the input image to rgba mode and save to a specified destination
    Parameters
    ----------
    img_path : path to a image, this image must be a gray level image.
    des_path : the destination path. 
    
    """

    img = scipy.misc.imread(img_path)
    img = mapping(img)

    img.save(des_path, "PNG")


def multi_thread_transfer(dir_path, des_path):
    """
    Transfer all the gray level images into rgba color images from dir_path to des_path.
    Theoretically, this function can parallels transfer multiple images.
    :param dir_path: input image directory
    :param des_path: output directory
    :return: 
    """
    imgs = os.listdir(dir_path)
    origin_paths = []
    des_paths = []
    if not os.path.exists(des_path):
        os.mkdir(des_path)
    for img in imgs:
        origin_paths.append(os.path.join(dir_path, img))
        des_paths.append(os.path.join(des_path, img))

    future_objs = []
    for i in range(len(imgs)):
        obj = _imread_executor_pool.submit(transfer, origin_paths[i], des_paths[i])
        future_objs.append(obj)
    wait(future_objs)


def multi_process_transfer(dir_path, des_path):
    """
    Transfer all the gray level images into rgba color images from dir_path to des_path
    using multiprocess which can highly speed up the transfer process.
    :param dir_path: 
    :param des_path: 
    :return: 
    """
    imgs = os.listdir(dir_path)
    origin_paths = []
    des_paths = []
    if not os.path.exists(des_path):
        os.makedirs(des_path)
    for img in imgs:
        origin_paths.append(os.path.join(dir_path, img))
        des_paths.append(os.path.join(des_path, img))

    p = Pool()
    for i in range(len(imgs)):
        p.apply_async(transfer, args=(origin_paths[i], des_paths[i], ))
    p.close()
    p.join()


def control_flow1(dir_path, out_path):
    """
    Simultaneously transform all gray level images in a datetime directory
    to rgba png images and keep the directory structure.
    :param dir_path:
    :param out_path:
    :return:
    """
    for dir in os.listdir(dir_path):
        in_p = os.path.join(dir_path, dir)
        out_p = os.path.join(out_path, dir+"_colored")
        multi_process_transfer(in_p, out_p)


if __name__ == "__main__":
    img_path = "/root/extend/result/conv_lstm_generator/conv_lstm_generator_epoch_3/19"
    des_path = "/root/extend/result/conv_lstm_rgb4_generator/conv_lstm_generator_epoch_3/19"
    a = time.time()
    multi_process_transfer(img_path, des_path)
    # transfer(img_path, des_path)
    # control_flow1(img_path, des_path)
    print(time.time()-a)
