import math

import numpy as np
import PIL.Image as Image
import matplotlib.pyplot as plt
import scipy.ndimage as ndi
from matplotlib.patches import ConnectionPatch
from skimage.feature import SIFT, match_descriptors
from skimage.color import rgb2gray, rgba2rgb
from skimage.transform import resize, ProjectiveTransform, SimilarityTransform, warp
from skimage import measure

# Part 2

def keypoint_matching(feature_set_1, feature_set_2):

    detector1 = SIFT()
    detector2 = SIFT()
    detector1.detect_and_extract(dst_img)
    detector2.detect_and_extract(src_img)
    keypoints1 = detector1.keypoints
    descriptors1 = detector1.descriptors
    keypoints2 = detector2.keypoints
    descriptors2 = detector2.descriptors
    
    matches = match_descriptors(descriptors1, descriptors2, cross_check=True)

    return matches, keypoints1, keypoints2

def plot_keypoint_matches(matches, keypoints1, keypoints2):
    
    # Select the points in img1 that match with img2 and vice versa
    dst = keypoints1[matches[:, 0]]
    src = keypoints2[matches[:, 1]]

    fig = plt.figure(figsize=(8, 4))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    ax1.imshow(dst_img, cmap='gray')
    ax2.imshow(src_img, cmap='gray')

    for i in range(src.shape[0]):
        coordB = [dst[i, 1], dst[i, 0]]
        coordA = [src[i, 1], src[i, 0]]
        con = ConnectionPatch(xyA=coordA, xyB=coordB, coordsA="data", coordsB="data",
                            axesA=ax2, axesB=ax1, color="red")
        ax2.add_artist(con)
        ax1.plot(dst[i, 1], dst[i, 0], 'ro')
        ax2.plot(src[i, 1], src[i, 0], 'ro')

# Part 3

def compute_affine_transform(source_points, dest_points):
    pass

def compute_projective_transform(source_points, dest_points):
    pass

def ransac(source_points, dest_points, iterations, min_samples, threshold_boundary):
    pass


# Testing

if __name__ == "__main__":

    dst_img_rgb = np.asarray(Image.open('img/Rainier1.png'))
    src_img_rgb = np.asarray(Image.open('img/Rainier2.png'))

    if dst_img_rgb.shape[2] == 4:
        dst_img_rgb = rgba2rgb(dst_img_rgb)
    if src_img_rgb.shape[2] == 4:
        src_img_rgb = rgba2rgb(src_img_rgb)

    dst_img = rgb2gray(dst_img_rgb)
    src_img = rgb2gray(src_img_rgb)