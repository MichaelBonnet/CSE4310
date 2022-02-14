import sys
import numpy as np
from numpy.lib import stride_tricks
from PIL import Image
from matplotlib import pyplot as plt
import math


# Patch extraction
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


image = np.asarray( Image.open("uss_enterprise.jpeg") )
num_patches = 16
patches = extract_patch( image, num_patches )

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
