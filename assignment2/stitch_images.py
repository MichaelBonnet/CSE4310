# imports

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

# Part 1

# For keypoint detection, you may use whichever feature extraction method you would like.
#   One recommendation is to use SIFT from scikit-image (available in v0.19).
def keypoint_and_descriptor_detection(src_img, dst_img):
    
    detector1 = SIFT()
    detector2 = SIFT()
    detector1.detect_and_extract(dst_img)
    detector2.detect_and_extract(src_img)
    keypoints1 = detector1.keypoints
    descriptors1 = detector1.descriptors
    keypoints2 = detector2.keypoints
    descriptors2 = detector2.descriptors

    return keypoints1, descriptors1, keypoints2, descriptors2


##############
### Part 2 ###
##############

# Keypoint Matching
#   Create a matching function that, given two sets of keypoint features, returns a list of
#   indices of matching pairs. That is, pair (i, j) in the list indicates a match between the
#   feature at index i in the source image with the feature at index j in the second image.
def keypoint_matching(keypoints1, descriptors1, keypoints2, descriptors2):
    
    matches = match_descriptors(descriptors1, descriptors2, cross_check=True)

    print(matches.shape)

    return matches, keypoints1, keypoints2

# Plot Keypoint Matches
#   Create a plotting function that combines two input images of the same size side-by-side
#   and plots the keypoints detected in each image. Additionally, plot lines between the
#   keypoints showing which ones have been matched together.
def plot_keypoint_matches(matches, keypoints1, keypoints2, src_img, dst_img):
    
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

    return src, dst

# Part 3

# Estimate Affine Matrix
#   Create a function compute_affine_transform which takes in a set of points from the
#   source image and their matching points in the destination image. Using these samples,
#   compute the affine transformation matrix using the normal equations. This function
#   should return a 3 × 3 matrix.
def compute_affine_transform(src, dst):

    num_samples = src.shape[0]

    src_affine  = np.concatenate((src, np.ones((num_samples, 1))), axis=1)
    zero_buffer = np.zeros_like(src_affine)
    r1 = np.concatenate((src_affine, zero_buffer), axis=1)
    r2 = np.concatenate((zero_buffer, src_affine), axis=1)

    X = np.empty((r1.shape[0] + r2.shape[0], r1.shape[0], r1.shape[1]), dtype=r1.dtype)
    X[0::2] = r1
    X[1::2] = r2

    y = dst.ravel()

    X_inv = np.linalg.pinv(X.T @ X)
    M = X_inv @ X.T @ y
    M = np.reshape(M, (2,3))
    M = np.vstack((M, np.zeros((1, M.shape[1]))))
    M[2, 2] = 1

    return M

# [4:24 PM] Estrada, Jorge
# im not home rn but it’s pretty much the same as the affine matrix

# [4:25 PM] Estrada, Jorge
# pretty much just create a 2xn matrix where n is the number of rows on the X matrix

# [4:26 PM] Estrada, Jorge
# then just loop through the new x after you concatenate and place the values at position 7 and 8 
# with what is shown on his notes in relation to projective matrix


def compute_projective_transform(src, dst):

    num_samples = src.shape[0]

    src_affine  = np.concatenate((src, np.ones((num_samples, 1))), axis=1)
    zero_buffer = np.zeros_like(src_affine)
    r1 = np.concatenate((src_affine, zero_buffer), axis=1)
    r2 = np.concatenate((zero_buffer, src_affine), axis=1)

    X = np.empty((r1.shape[0] + r2.shape[0], r1.shape[0], r1.shape[1]), dtype=r1.dtype)
    X[0::2] = r1
    X[1::2] = r2

    y = dst.ravel()

    X_inv = np.linalg.pinv(X.T @ X)
    M = X_inv @ X.T @ y
    M = np.reshape(M, (2,3))
    M = np.vstack((M, np.zeros((1, M.shape[1]))))
    M[2, 2] = 1

    return M

def ransac(src, dst, iterations, min_samples, threshold_boundary):

    sk_M, sk_best = measure.ransac((src[:, ::-1], dst[:, ::-1]), ProjectiveTransform, min_samples=4, residual_threshold=1, max_trials=300)


    sk_M, sk_best = measure.ransac((src[:, ::-1], dst[:, ::-1]), ProjectiveTransform, min_samples=min_samples, residual_threshold=threshold_boundary, max_trials=iterations)
    
    print(sk_M)

    return sk_M, sk_best

def ransac_2(src, dst, iterations, min_samples, threshold_boundary):
    pass


# Testing

if __name__ == "__main__":

    # Prepping data for use
    dst_img_rgb = np.asarray(Image.open('img/Rainier1.png'))
    src_img_rgb = np.asarray(Image.open('img/Rainier2.png'))

    if dst_img_rgb.shape[2] == 4:
        dst_img_rgb = rgba2rgb(dst_img_rgb)
    if src_img_rgb.shape[2] == 4:
        src_img_rgb = rgba2rgb(src_img_rgb)

    dst_img = rgb2gray(dst_img_rgb)
    src_img = rgb2gray(src_img_rgb)

    # Plotting prepped data
    fig = plt.figure(figsize=(8, 4))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    ax1.imshow(dst_img, cmap='gray')
    ax2.imshow(src_img, cmap='gray')

    # Calling things
    keypoints1, descriptors1, keypoints2, descriptors2 = keypoint_and_descriptor_detection(src_img, dst_img)

    matches, keypoints1, keypoints2 = keypoint_matching(keypoints1, descriptors1, keypoints2, descriptors2)

    src, dst = plot_keypoint_matches(matches, keypoints1, keypoints2, src_img, dst_img)

    affine_transform     = compute_affine_transform(src, dst)
    projective_transform = compute_projective_transform(src, dst)

    sk_M, sk_best = ransac(src, dst, iterations=300, min_samples=4, threshold_boundary=1)

    print(np.count_nonzero(sk_best))
    src_best = keypoints2[matches[sk_best, 1]][:, ::-1]
    dst_best = keypoints1[matches[sk_best, 0]][:, ::-1]

    # Comparison Figure
    fig = plt.figure(figsize=(8, 4))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    ax1.imshow(dst_img_rgb)
    ax2.imshow(src_img_rgb)

    for i in range(src_best.shape[0]):
        coordB = [dst_best[i, 0], dst_best[i, 1]]
        coordA = [src_best[i, 0], src_best[i, 1]]
        con = ConnectionPatch(xyA=coordA, xyB=coordB, coordsA="data", coordsB="data",
                            axesA=ax2, axesB=ax1, color="red")
        ax2.add_artist(con)
        ax1.plot(dst_best[i, 0], dst_best[i, 1], 'ro')
        ax2.plot(src_best[i, 0], src_best[i, 1], 'ro')

    # Compute the Output Shape
    # Transform the corners of img1 by the inverse of the best fit model
    rows, cols = dst_img.shape
    corners = np.array([
        [0, 0],
        [cols, 0],
        [0, rows],
        [cols, rows]
    ])

    corners_proj = sk_M(corners)
    all_corners = np.vstack((corners_proj[:, :2], corners[:, :2]))

    corner_min = np.min(all_corners, axis=0)
    corner_max = np.max(all_corners, axis=0)
    output_shape = (corner_max - corner_min)
    output_shape = np.ceil(output_shape[::-1]).astype(int)
    print(output_shape)

    offset = SimilarityTransform(translation=-corner_min)
    dst_warped = warp(dst_img_rgb, offset.inverse, output_shape=output_shape)

    tf_img = warp(src_img_rgb, (sk_M + offset).inverse, output_shape=output_shape)

    # Combine the images
    foreground_pixels = tf_img[tf_img > 0]
    dst_warped[tf_img > 0] = tf_img[tf_img > 0]

    # Plot the result
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.imshow(dst_warped)