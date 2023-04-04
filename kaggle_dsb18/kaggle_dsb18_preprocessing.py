import os
import numpy as np
from skimage import io
from six.moves import collections_abc
from argparse import ArgumentParser
from random import random
import torchvision.transforms as T
from MGUnet.unet.dataset import correct_dims
#from monai.losses.dice import *
#from monai.losses.dice import DiceLoss

# I would like to take a minute to express that relative imports in Python are horrible.
# Although this function is implemented somewhere else, it cannot be imported, since its
# folder is in the parent folder of this. Relative imports result in ValueErrors. The
# design choice behind this decision eludes me. The only way to circumvent this is either
# make this package installable, add the parent folder to PATH or implement it again.
# I went with the latter one.
#
# If you are reading this and you also hate the relative imports in Python, cheers!
# You are not alone.
def chk_mkdir(*paths: collections_abc) -> None:
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
    return merged_mask


def enlarge_image(a,M=1024, N=1024, L=4 ):
    p, q, r = a.shape
    out = np.zeros((M, N, L),dtype=a.dtype)
    out[:p, :q, :r] = a
    return out.astype('uint8')


if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('--dataset_path', required=True, type=str)
    parser.add_argument('--export_path', required=True, type=str)
    args = parser.parse_args()
    # args.dataset_path = '/gpfs/usr/vadim/my_work/Data/kaggle neuron/data-science-bowl-2018/stage1_train/'
    # args.export_path = '/gpfs/usr/vadim/my_work/Data/kaggle neuron/data-science-bowl-2018/u_net_train/'
    new_train_images_folder = os.path.join(args.export_path,'train', 'images')
    new_train_masks_folder = os.path.join(args.export_path,'train', 'masks')
    new_test_images_folder = os.path.join(args.export_path, 'test', 'images')
    new_test_masks_folder = os.path.join(args.export_path, 'test', 'masks')
    new_val_images_folder = os.path.join(args.export_path, 'val', 'images')
    new_val_masks_folder = os.path.join(args.export_path, 'val', 'masks')

    chk_mkdir(args.export_path, new_train_images_folder, new_train_masks_folder, new_test_images_folder,new_test_masks_folder, new_val_images_folder, new_val_masks_folder)
    d1 = d2 = 0

    for image_name in os.listdir(args.dataset_path):
        image = io.imread(os.path.join(args.dataset_path, image_name, 'images', image_name + '.png'))
        if (image.shape)[0] > d1:
            d1 = (image.shape)[0]
        if (image.shape)[1] > d2:
            d2 = (image.shape)[1]

    for image_name in os.listdir(args.dataset_path):
        images_folder = os.path.join(args.dataset_path, image_name, 'images')
        masks_folder = os.path.join(args.dataset_path, image_name, 'masks')
        image = io.imread(os.path.join(args.dataset_path, image_name, 'images', image_name +'.png'))

        tmp = random()
        if tmp < 0.7:
            new_images_folder = new_train_images_folder
            new_masks_folder = new_train_masks_folder
        elif tmp > 0.85:
            new_images_folder = new_test_images_folder
            new_masks_folder = new_test_masks_folder
        else:
            new_images_folder = new_val_images_folder
            new_masks_folder = new_val_masks_folder

        #save the image sample
        io.imsave(os.path.join(new_images_folder, image_name + '.png'), image)

        #merging all masks together
        mask_img = merge_masks(masks_folder)
        '''
        #m1, i1 = correct_dims(mask_img, image)
        #print("dice loss :: ", dl(torch.from_numpy(mask_img), torch.from_numpy(image)))
        #padding and saving the merged mask
        enlarged_mask = np.zeros((512, 512), dtype=mask_img.dtype)
        enlarged_mask[:mask_img.shape[0], :mask_img.shape[1]] = mask_img'''
        io.imsave(os.path.join(new_masks_folder, image_name + '.png'), mask_img.astype('uint8'))



