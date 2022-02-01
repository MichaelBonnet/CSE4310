from random import randint
import sys
import numpy as np
from numpy.lib import stride_tricks
import PIL
from PIL import Image
from matplotlib import pyplot as plt

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

    column = randint( (0 + size/2), (width - size/2)+1 )
    row    = randint( (0 + size/2), (height - size/2)+1 )

    col_lo_bound = column - size/2
    col_hi_bound = column + size/2
    row_lo_bound = row    - size/2
    row_hi_bound = row    + size/2

    random_cropped = img[row_lo_bound:row_hi_bound, col_lo_bound:col_hi_bound]
    
    return random_cropped


# Patch extraction
# Following the instructions given here (https://twitter.com/MishaLaskin/status/1478500251376009220), 
# create a function that returns n^2 non-overlapping patches given an input image, as
# a numpy array, as well as an integer n.
# You may assume that the input image is square.
def extract_patch(img, num_patches):
    
    # non-overlapping patches of size (num_patches)
    size = num_patches
    H, W = img.shape
    shape = [H // size, W // size] + [size, size]

    # (row, col, patch_row, patch_col)
    strides = [size * s for s in img.strides] + list(img.strides)
    # extract patches
    patches = stride_tricks.as_strided(img, shape=shape, strides=strides)

    # not sure what the integer n is supposed to be?
    n = size * size
    
    # return patches as array as well as an integer n
    return patches, n


# Resizing function that resizes an image given an input image as a numpy array
# and an integer representing the desired scale factor. 
# The image should be resized using nearest neighbor interpolation.
def resize_img(img, factor):
    pass


# Randomly perturbs the HSV values on an input image by an amount 
#   no greater than the given input value. 
# This should use your code from the first part of the assignment to modify the HSV channels.
def color_jitter(img, hue, saturation, value):
    pass