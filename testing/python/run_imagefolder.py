import argparse
import cv2 as cv
import numpy as np
import numpy.linalg as LA
import cPickle as pk
from os import makedirs, remove
from os.path import join, exists, abspath, dirname, basename, isfile

from main import main


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description="Run Openpose on a image folder with still images and image sequences,"
        "e.g., a folder with image_1.jpg, image_1.png, video/frame_{1-100}.png ..."
        "The image_folder is supposed to have a data_info.pkl file.")
    parser.add_argument(
        'image_folder', type=str, metavar='DIR',
        help="Path to the image folder.")
    parser.add_argument(
        'vis_folder', type=str, metavar='DIR',
        help="Location of a folder for saving visualization images.")
    parser.add_argument(
        'save_path', type=str, metavar='DIR',
        help='Location of the output 2D joint locations.')
    args = parser.parse_args()
    image_folder = args.image_folder
    vis_folder = args.vis_folder
    save_path = args.save_path

    # ------------------------------------------------------------
    # Load data info
    # ------------------------------------------------------------

    print("Loading image folder info from {0:s} ...".format(
        join(image_folder, "data_info.pkl")))
    with open(join(image_folder, "data_info.pkl"), 'r') as f:
        data_info = pk.load(f)
        image_names = data_info["image_names"]
        item_names = data_info["item_names"]
        item_lengths = data_info["item_lengths"]
        item_to_image = data_info["item_to_image"]
        #image_to_itemframe = data_info["image_to_itemframe"]

    # ------------------------------------------------------------
    # Run Openpose on each of the items (i.e., video frames and
    # individual images) within image_folder
    # ------------------------------------------------------------

    save_dict = dict()
    for i,item_name in enumerate(item_names):
        print("  Running Openpose on item #{0:d}: {1:s} ...".format(i, item_name))
        item_length = item_lengths[i]

        # Get images_paths from data_info
        image_paths = [None]*item_length
        for k in range(item_length):
            image_paths[k] = join(image_folder, image_names[item_to_image[i] + k])

        if item_length > 1: # video frames
            vis_dir = join(vis_folder, item_name)
        else: # individual images
            vis_dir = vis_folder

        save_dict[item_name] = main(image_paths, vis_dir, save_path=None)

    # Save estimated joint 2D positions
    with open(save_path, 'w') as f:
        pk.dump(save_dict, f)
