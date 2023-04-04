import os
import numpy as np

from skimage import io, img_as_ubyte

from shutil import copy
from collections import Container
from argparse import ArgumentParser
from random import random


# I would like to take a minute to express that relative imports in Python are horrible.
# Although this function is implemented somewhere else, it cannot be imported, since its
# folder is in the parent folder of this. Relative imports result in ValueErrors. The
# design choice behind this decision eludes me. The only way to circumvent this is either
# make this package installable, add the parent folder to PATH or implement it again.
# I went with the latter one.
#
# If you are reading this and you also hate the relative imports in Python, cheers!
# You are not alone.
def chk_mkdir(*paths: Container) -> None:
    """
    Creates folders if they do not exist.

    Args:
        paths: Container of paths to be created.
    """
    for path in paths:
        if not os.path.exists(path):
            os.makedirs(path)


def merge_masks(masks_folder):
    masks = list()
    for mask_img_filename in os.listdir(masks_folder):
        mask_img = io.imread(os.path.join(masks_folder, mask_img_filename))
        masks.append(mask_img)

    merged_mask = np.sum(masks, axis=0)
    merged_mask[merged_mask > 0] = 1

    return img_as_ubyte(merged_mask)


if __name__ == '__main__':

    parser = ArgumentParser()
    # parser.add_argument('--dataset_path', required=True, type=str)
    # parser.add_argument('--export_path', required=True, type=str)
    parser.add_argument('--dataset_path', default='/gpfs/usr/vadim/my_work/Data/Carvana/carvana-image-masking-challenge', type=str)
    parser.add_argument('--export_path', default='/gpfs/usr/vadim/my_work/Data/Carvana/u_net_train/', type=str)
    args = parser.parse_args()
    # args.dataset_path = '/gpfs/usr/vadim/my_work/Data/kaggle neuron/data-science-bowl-2018/stage1_train/'
    # args.export_path = '/gpfs/usr/vadim/my_work/Data/kaggle neuron/data-science-bowl-2018/u_net_train/'
    new_train_images_folder = os.path.join(args.export_path,'train', 'images')
    new_train_masks_folder = os.path.join(args.export_path,'train', 'masks')
    new_test_images_folder = os.path.join(args.export_path, 'test', 'images')
    new_test_masks_folder = os.path.join(args.export_path, 'test', 'masks')
    new_val_images_folder = os.path.join(args.export_path, 'val', 'images')
    new_val_masks_folder = os.path.join(args.export_path, 'val', 'masks')

    data_dir = os.path.join(args.dataset_path, 'train')
    mask_dir = os.path.join(args.dataset_path, 'train_masks')

    chk_mkdir(args.export_path, new_train_images_folder, new_train_masks_folder, new_test_images_folder,new_test_masks_folder, new_val_images_folder, new_val_masks_folder)

    for image_name in os.listdir(data_dir):
        image_name = image_name.split('.')[0]
        image_f = os.path.join(data_dir, image_name+'.jpg')
        mask_f = os.path.join(mask_dir, image_name+'_mask.gif')

        tmp = random()
        if tmp <0.7:
            new_images_folder = new_train_images_folder
            new_masks_folder = new_train_masks_folder
        elif tmp > 0.85:
            new_images_folder = new_test_images_folder
            new_masks_folder = new_test_masks_folder
        else:
            new_images_folder = new_val_images_folder
            new_masks_folder = new_val_masks_folder
        # copy the image
        # img = io.imread(image_f)
        # io.imsave(os.path.join(new_images_folder, image_name + '.png'), img)
        # msk = io.imread(mask_f)
        # io.imsave(os.path.join(new_masks_folder, image_name + '.png'), msk)
        copy(src=image_f,
             dst=os.path.join(new_images_folder, image_name+'.jpg'))
        copy(src=mask_f,
             dst=os.path.join(new_masks_folder, image_name+'.gif'))

