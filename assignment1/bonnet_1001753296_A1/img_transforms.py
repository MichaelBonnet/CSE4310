from change_hsv import hsv_to_rgb, rgb_to_hsv

from random import randint
import random
import sys
import numpy as np
from numpy.lib import stride_tricks
import PIL
from PIL import Image
from matplotlib import pyplot as plt

# WORKING
#   random_crop
#   resize_img
#   extract_patch

# NOT WORKING
#   ?

# UNTESTED
#   color_jitter


# Generates a random square crop of an image given the following inputs:
# • An image as a numpy array.
# • An integer reflecting the size.
# Should check to make sure the crop size is feasible given the input image size. 
#   For example, if the image size is [w, h], 
#   then the input size s must be in range s ∈ (0, min(w, h)]. 
# Should pick a random center location from which to crop and then return the cropped image.
def random_crop(img, size):

    height, width, c = img.shape
    if size not in range(0, (min(height, width)+1) ):
        print("improper size given")
        return

    # Generate random center coordinates for the crop
    column = randint( (0 + size/2), ((width - size/2)+1)  )
    row    = randint( (0 + size/2), ((height - size/2)+1) )

    col_lo_bound = int( column - size/2 )
    col_hi_bound = int( column + size/2 )
    row_lo_bound = int( row    - size/2 )
    row_hi_bound = int( row    + size/2 )

    random_cropped = img[row_lo_bound:row_hi_bound, col_lo_bound:col_hi_bound]
    
    return random_cropped

# Testing above
# Status: WORKING
def test_random_crop(img, size):

    random_cropped = random_crop(img, size)

    plt.imshow(random_cropped)
    plt.show()


# Patch extraction
# Following the instructions given here (https://twitter.com/MishaLaskin/status/1478500251376009220), 
# create a function that returns n^2 non-overlapping patches given an input image, as
# a numpy array, as well as an integer n.
# You may assume that the input image is square.
def extract_patch(img, num_patches):

    # non-overlapping patches
    H, W, D = img.shape
    size = int(H / (num_patches / 2))
    shape = [H // num_patches, W // num_patches] + [size, size, 3]

    # (row, col, patch_row, patch_col)
    strides = [size * s for s in img.strides[:2]] + list(img.strides)

    # extract patches
    patches = stride_tricks.as_strided(img, shape=shape, strides=strides) 
    
    # return patches as array as well as an integer n
    return patches

# Testing above
# Status: WOKRING
def test_extract_patch(img, num_patches):

    patches = extract_patch( img, num_patches )

    nrows = int( num_patches / 2 )
    ncols = int( num_patches / 2 )

    fig = plt.figure()
    for row in range(nrows):
        for col in range(ncols):
            index = row * ncols + col
            ax = fig.add_subplot(nrows, ncols, index+1)
            ax.imshow(patches[row,col,:,:])
            ax.tick_params(left=False,
                        bottom=False,
                        labelleft=False,
                        labelright=False)

    fig.savefig("patches.png")

    plt.show()


# Resizing function that resizes an image given an input image as a numpy array
# and an integer representing the desired scale factor. 
# The image should be resized using nearest neighbor interpolation.
# Modified from answer at:
# https://stackoverflow.com/questions/69728373/resize-1-channel-numpy-image-array-with-nearest-neighbour-interpolation
def resize_img(img, scale_factor):
    
    shape = [img.shape[0], img.shape[1]]
    
    def per_axis(in_sz, out_sz):
        ratio = 0.5 * in_sz / out_sz
        return np.round(np.linspace(ratio - 0.5, in_sz - ratio - 0.5, num=out_sz)).astype(int)

    return img[per_axis(img.shape[0], (shape[0]*scale_factor) )[:, None],
               per_axis(img.shape[1], (shape[1]*scale_factor) )]

# Testing above
# Status: WORKING
def test_resize_img(img, scale_factor):
    
    resized_image = resize_img(img, scale_factor)

    plt.imshow(resized_image)
    plt.show()


# Randomly perturbs the HSV values on an input image by an amount 
#   no greater than the given input value. 
# This should use your code from the first part of the assignment to modify the HSV channels.
def color_jitter(img, hue, saturation, value):

    hsv_image = rgb_to_hsv(img)
    
    hsv_image[:, 0] = hsv_image[:, 0] + randint(        ( -1 * hue ),        hue        )  # Perturbing H channel
    hsv_image[:, 1] = hsv_image[:, 1] + random.uniform( ( -1 * saturation ), saturation )  # Perturbing S channel
    hsv_image[:, 2] = hsv_image[:, 2] + random.uniform( ( -1 * value ),      value      )  # Perturbing V channel

    # Catch out-of-bounds for H channel
    # hsv_image[:, 0] = 60[ hsv_image[:, 0] > 60 ]
    # hsv_image[:, 0] = 0[ hsv_image[:, 0] < 0 ]
    hsv_image[:, 0][ hsv_image[:, 0] > 60 ] = 60
    hsv_image[:, 0][ hsv_image[:, 0] < 0  ] = 0

    # Catch out-of-bounds for S channel
    hsv_image[:, 1][ hsv_image[:, 1] > 1 ] = 1
    hsv_image[:, 1][ hsv_image[:, 1] < 0 ] = 0

    # Catch out-of-bounds for V channe
    hsv_image[:, 2][ hsv_image[:, 2] > 1 ] = 1
    hsv_image[:, 2][ hsv_image[:, 2] < 0 ] = 0

    # Get newly-perturbed image
    return hsv_to_rgb( hsv_image )

# Testing above
# Status: UNTESTED
def test_color_jitter(img, hue, saturation, value):

    jittered_image = color_jitter(img, hue, saturation, value)

    plt.imshow(jittered_image)
    plt.show()


# Main function
def main(argv, argc):

    filename = argv[1]
    option   = argv[2]

    image = np.asarray( Image.open(filename) )

    if option == "random_crop":
        test_random_crop( image, int(argv[3]) )
    elif option == "extract_patch":
        test_extract_patch( image, int(argv[3]) )
    elif option == "resize_img":
        test_resize_img( image, int(argv[3]) )
    elif option == "color_jitter":
        test_color_jitter( image, int(argv[3]), int(argv[4]), int(argv[5]) )


# Calling main()
if __name__ == "__main__":
    main( sys.argv, len(sys.argv) )