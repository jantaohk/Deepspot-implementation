import openslide
import pandas as pd
import numpy as np
from openslide import OpenSlide
from openslide.lowlevel import OpenSlideUnsupportedFormatError
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
from multiprocessing import Pool, Value, Lock
import os
from skimage.color import rgb2hsv
from skimage.filters import threshold_otsu
from skimage.io import imsave, imread
from skimage.exposure.exposure import is_low_contrast
from skimage.transform import resize
from scipy.ndimage import binary_dilation, binary_erosion
import argparse
import logging
import h5py
from tqdm import tqdm
import pickle
import re
import pdb
import pandas as pd

def get_mask_image(img_RGB, RGB_min=50):
    # img_RGB: slide pixel information. later converted to lowest resolution in get_mask
    # RGB_min: decides which dark pixels to ignore
    img_HSV = rgb2hsv(img_RGB)

    background_R = img_RGB[:, :, 0] > threshold_otsu(img_RGB[:, :, 0]) # otsu thresholding for each colour channel, separating background and foreground
    background_G = img_RGB[:, :, 1] > threshold_otsu(img_RGB[:, :, 1]) # [height, width, colour channels]
    background_B = img_RGB[:, :, 2] > threshold_otsu(img_RGB[:, :, 2]) # the arrow compares to the threshold determined by otsu, and asks if the colour intensity of each channel is higher than that threshold. this returns a bunch of booleans
    tissue_RGB = np.logical_not(background_R & background_G & background_B) # merge the dark bits (tissue) together. logical_not essentially takes the inverse of each background
    tissue_S = img_HSV[:, :, 1] > threshold_otsu(img_HSV[:, :, 1]) # otsu thresholding for saturation (colour intensity)
    min_R = img_RGB[:, :, 0] > RGB_min # eliminates pixels that are too dark
    min_G = img_RGB[:, :, 1] > RGB_min
    min_B = img_RGB[:, :, 2] > RGB_min

    mask = tissue_S & tissue_RGB & min_R & min_G & min_B # combines everything together to get binary image
    return mask
    
def get_mask(slide, slide_type, level='max', RGB_min=50):
    resized= None
    if level == 'max': # keep this line even if openslide is problematic
        if slide_type in "openslide":
            level = len(slide.level_dimensions) - 1 # picks the lowest resolution
        else:
            level= 3 # doesnt seem important for non-openslide
            downsample= 32

    if slide_type== "openslide":
        print("Using OpenSlide")
        print("Level dimensions (before transpose):", slide.level_dimensions[level])
        raw_img = slide.read_region((0, 0), level, slide.level_dimensions[level]).convert('RGB')
        print("Raw OpenSlide image shape:", raw_img.size)
        img_RGB = np.transpose(np.array(slide.read_region((0, 0),level,slide.level_dimensions[level]).convert('RGB')), # maintains formatting and axes consistency for the lowest resolution image###
                           axes=[1, 0, 2]) # note the shape of img_RGB is the transpose of slide.level_dimensions.###
        print("img_RGB shape after transpose:", img_RGB.shape)
    else:
        print("Using PIL (not OpenSlide)")
        print("dimensions before transpose:", slide.size)
        resized= slide.resize((round(slide.width// downsample), round(slide.height// downsample)))
        print("Resized PIL image size:", resized.size)
        img_RGB = np.transpose(np.array(resized.convert("RGB")), axes= [1, 0, 2])
        print("img_RGB shape from PIL:", img_RGB.shape)
    tissue_mask = get_mask_image(img_RGB, RGB_min)
    return tissue_mask, level, resized

def extract_patches(slide_path, mask_path, patch_size, patches_output_dir, slide_id, max_patches_per_slide=2000):
    # downsampling is used because areas which are tissue will be seen in this downsampled image. rather than going through the entire proper image, this saves time

    patch_folder = os.path.join(patches_output_dir, slide_id) # making folders to store patches and masks
    if not os.path.isdir(patch_folder):
        os.makedirs(patch_folder)

    patch_folder_mask = os.path.join(mask_path, slide_id)
    if not os.path.isdir(patch_folder_mask):
        os.makedirs(patch_folder_mask)

    if os.path.exists(os.path.join(patch_folder, "complete.txt")): # checks if a patch has already been made for slides of a certain id
        print(f'{slide_id}: patches have already been extracted')
        return

    path_hdf5 = os.path.join(patch_folder, f"{slide_id}.hdf5") # stores in hdf5 format, efficient way of storing whole slide images
    hdf = h5py.File(path_hdf5, 'w')

    try:
        slide = OpenSlide(slide_path) # loads the slide, or alternatively:
        slide_type= "openslide"
        
    except Exception as e:
        print("Slide could not be opened with OpenSlide. Exception:", e)
        slide= Image.open(slide_path)
        slide_type= "pil"
    
    mask, mask_level, resized = get_mask(slide, slide_type)
    mask = binary_dilation(mask, iterations=3) # fills holes and smoothes edges
    mask = binary_erosion(mask, iterations=3) # the opposite, reaches the same goal
    np.save(os.path.join(patch_folder_mask, "mask.npy"), mask) # save as a binary numpy format for easy access

    if slide_type in "openslide":
        mask_level = len(slide.level_dimensions) - 1 # lowest resolution
    else:
        mask_level= 3 # could change

    PATCH_LEVEL = 0 # highest resolution
    BACKGROUND_THRESHOLD = .2

    try:
        if slide_type in "openslide":
            ratio_x = slide.level_dimensions[PATCH_LEVEL][0] / slide.level_dimensions[mask_level][0] # dimension at full resolution/ lowest
            ratio_y = slide.level_dimensions[PATCH_LEVEL][1] / slide.level_dimensions[mask_level][1]
            xmax, ymax = slide.level_dimensions[PATCH_LEVEL] # width and height at full resolution
            ###xmax= xmax// 2 # crucial line to half the image if needed
            
            # some slides are 20x and some are 40x, this adjusts patch size accordingly
            resize_factor = float(slide.properties.get('aperio.AppMag', 20)) / 20.0
            if not slide.properties.get('aperio.AppMag', 20): print(f"magnifications for {slide_id} is not found, using default magnification 20X")
                
        else:
            xmax, ymax = slide.size
            width_r, height_r= resized.size
            ratio_x= xmax/ width_r
            ratio_y= ymax/ height_r
            resize_factor= 1 ######################

        patch_size_resized = (int(resize_factor * patch_size[0]), int(resize_factor * patch_size[1]))
        print(f"patch size for {slide_id}: {patch_size_resized}")

        i = 0
        indices = [(x, y) for x in range(0, xmax, patch_size_resized[0]) for y in
                    range(0, ymax, patch_size_resized[0])] # creates grid positions

        # here, we generate all the patches with valid mask
        if max_patches_per_slide is None:
            max_patches_per_slide = len(indices)

        np.random.seed(5)
        np.random.shuffle(indices) # randomises grid positions, so not all of them will be in the top left of the image (number of patches is capped)

        for x, y in indices:
            # check if in background mask
            x_mask = int(x / ratio_x)
            y_mask = int(y / ratio_y)
            if slide_type!= "openslide":
                x_mask = min(int(x / ratio_x), mask.shape[0] - 1)
                y_mask = min(int(y / ratio_y), mask.shape[1] - 1)
            if mask[x_mask, y_mask] == 1: # only selects tissues and not background
                if slide_type in "openslide":
                    patch = slide.read_region((x, y), PATCH_LEVEL, patch_size_resized).convert('RGB') # collect patch
                else:
                    crop_box= (x, y, x+ patch_size_resized[0], y+ patch_size_resized[1])
                    patch= slide.crop(crop_box).convert("RGB") # might check if this works properly or not. area outside image could be cropped by mistake
                try:
                    mask_patch = get_mask_image(np.array(patch)) # checks again that this is tissue, particularly for edge cases where downsampling may cause issues
                    mask_patch = binary_dilation(mask_patch, iterations=3)
                except Exception as e:
                    print("error with slide id {} patch {}".format(slide_id, i))
                    print(e)
                if (mask_patch.sum() > BACKGROUND_THRESHOLD * mask_patch.size) and not (is_low_contrast(patch)):
                    if resize_factor != 1.0: # saving
                        patch = patch.resize(patch_size)
                    patch = np.array(patch)
                    tile_name = f"{x}_{y}"
                    hdf.create_dataset(tile_name, data=patch)
                    i = i + 1
            if i >= max_patches_per_slide:
                break

        hdf.close()

        if i == 0:
            print("no patch extracted for slide {}".format(slide_id))
        else:
            with open(os.path.join(patch_folder, "complete.txt"), 'w') as f: # ensures this script isn't run again
                f.write('Process complete!\n')
                f.write(f"Total n patch = {i}")
                print(f"{slide_id} complete, total n patch = {i}")

    except Exception as e: # error handling
        print("error with slide id {} patch {}".format(slide_id, i))
        print(e)

def get_slide_id(slide_name): # extracts slide ID from file name
    return slide_name.split('.')[0]

def process(opts): # helps with parelleling processing
    slide_path, patch_size, patches_output_dir, mask_path, slide_id, max_patches_per_slide = opts
    extract_patches(slide_path, mask_path, patch_size,
                    patches_output_dir, slide_id, max_patches_per_slide)
    
parser = argparse.ArgumentParser(description='Generate patches from a given folder of images')
parser.add_argument('--ref_file', default= None, required=False, metavar='ref_file', type=str,
                    help='Path to the ref_file, if provided, only the WSIs in the ref file will be processed') ### changed the default option to none
parser.add_argument("--lookup_table", type= str, help= "Path to csv file containing metadata")
parser.add_argument('--patch_path', default="examples/Patches_hdf5" ,metavar='PATCH_PATH', type=str,
                    help='Path to the output directory of patch images')
parser.add_argument('--mask_path', default="examples/Patches_hdf5", metavar='MASK_PATH', type=str,
                    help='Path to the  directory of numpy masks')
parser.add_argument('--patch_size', default=256, type=int, help='patch size, '
                                                                'default 256')
parser.add_argument('--start', type=int, default=0,
                    help='Start slide index for parallelization') # start and end apparently relates to the images listed in reference.csv
parser.add_argument('--end', type=int, default=None,
                    help='End slide index for parallelization') # index of last slide to be processed
parser.add_argument('--max_patches_per_slide', default=None, type=int)
parser.add_argument('--debug', default=0, type=int,
                    help='whether to use debug mode') # 1= debug mode
parser.add_argument('--parallel', default=1, type=int,
                    help='whether to use parallel computation') # 1= run in parellel

if __name__ == '__main__':

    args = parser.parse_args() # reads command line arguments
    lookup_table= pd.read_csv(args.lookup_table)
    slide_list= lookup_table["image_path"].to_list()
    if args.ref_file: # there might be an optional csv file containing slide files, this makes sures only those files are processed
        ref_file = pd.read_csv(args.ref_file)
        selected_slides = list(ref_file['wsi_file_name'])
        wsi_files = [f'{s}.svs' for s in selected_slides]+ [f"{s}.png" for s in selected_slides]+ [f'{s}.tif' for s in selected_slides]+ [f"{s}.tiff" for s in selected_slides]
        slide_list = list(set(slide_list) & set(wsi_files))
        slide_list = sorted(slide_list)
    if args.start is not None and args.end is not None: # links back to the parser stuff earlier
        slide_list = slide_list[args.start:args.end] # start and end slides
        print(slide_list)
    elif args.start is not None:
        slide_list = slide_list[args.start:]
    elif args.end is not None:
        slide_list = slide_list[:args.end]

    if args.debug: # debug mode- tests only 5 slides and up to 20 patches
        slide_list = slide_list[0:5]
        args.max_patches_per_slide = 20

    print(f"Found {len(slide_list)} slides")
    opts = [
        (
            slide_path,  # full path already from lookup
             (args.patch_size, args.patch_size),
             args.patch_path,
             args.mask_path,
             get_slide_id(os.path.basename(slide_path)),  # extract slide ID from filename only
             args.max_patches_per_slide
        )
        for slide_path in slide_list
    ]

    if args.parallel: # whether to run in parallel, or each slide one by one
        pool = Pool(processes=4)
        pool.map(process, opts)
    else:
        for opt in opts:
            process(opt)