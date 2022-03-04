import sys
from tkinter import X
import numpy as np
import PIL
from PIL import Image
from matplotlib import pyplot as plt

# Takes an RGB image represented as a numpy array (rgb_image)
# And converts it into HSV format.
def rgb_to_hsv(image):

    image_norm = image /255
    R = image_norm[:,:,0]
    G = image_norm[:,:,1]
    B = image_norm[:,:,2]
    
    v_max = np.max(image_norm,axis=2)
    v_min = np.min(image_norm,axis=2)
    C = v_max - v_min
    
    hue_defined = C > 0 
    
    r_is_max = np.logical_and(R == v_max, hue_defined)
    g_is_max = np.logical_and(G == v_max, hue_defined)
    b_is_max = np.logical_and(B == v_max, hue_defined)
    
    H = np.zeros_like(v_max)
    H_r = ((G[r_is_max] - B[r_is_max]) / C[r_is_max]) % 6
    H_g = ((B[g_is_max] - R[g_is_max]) / C[g_is_max]) + 2
    H_b = ((R[b_is_max] - G[b_is_max]) / C[b_is_max]) + 4
    
    H[r_is_max] = H_r
    H[g_is_max] = H_g
    H[b_is_max] = H_b
    H *= 60
    
    V = v_max
    
    sat_defined = V > 0
    print(sat_defined)
    
    S = np.zeros_like(v_max)
    S[sat_defined] = C[sat_defined] / V[sat_defined]
    
    return np.dstack((H, S, V))


# Takes an HSV image represented as a numpy array (hsv_image) 
# And converts it into RGB format.
def hsv_to_rgb( hsv_image ):
    
    # Extracting each channel
    h = hsv_image[:, 0]
    s = hsv_image[:, 1]
    v = hsv_image[:, 2]

    # Getting C (chroma) back
    # Chroma (C) = V * S
    c = v * s

    # Getting H' back
    h_prime = h / 60

    # Getting X
    # X = C * (1 - |H' % 2 - 1| )
    x = c * ( 1 - np.absolute(h_prime % 2 - 1) )

    # making array for RGB
    rgb = np.zeros_like(hsv_image)

    print( "Shape of = "+str( rgb.shape ) )

    # Truth value masks
    mask_0_1 = np.logical_and( 0 <= h_prime, h_prime < 1 )
    mask_1_2 = np.logical_and( 1 <= h_prime, h_prime < 2 )
    mask_2_3 = np.logical_and( 2 <= h_prime, h_prime < 3 )
    mask_3_4 = np.logical_and( 3 <= h_prime, h_prime < 4 )
    mask_4_5 = np.logical_and( 4 <= h_prime, h_prime < 5 )
    mask_5_6 = np.logical_and( 5 <= h_prime, h_prime < 6 )

    # Populating RGB channels with truth value masks
    rgb[mask_0_1] = np.hstack( [ c[mask_0_1], x[mask_0_1], 0           ] )
    rgb[mask_1_2] = np.hstack( [ x[mask_1_2], c[mask_1_2], 0           ] )
    rgb[mask_2_3] = np.hstack( [ 0,           c[mask_2_3], x[mask_2_3] ] )
    rgb[mask_3_4] = np.hstack( [ 0,           x[mask_3_4], c[mask_3_4] ] )
    rgb[mask_4_5] = np.hstack( [ x[mask_4_5], 0,           c[mask_4_5] ] )
    rgb[mask_5_6] = np.hstack( [ c[mask_5_6], 0,           x[mask_5_6] ] )

    # Getting M for final modification
    m = v - c

    # Adding m back into each channel
    rgb[:, 0] = rgb[:, 0] + m
    rgb[:, 1] = rgb[:, 1] + m
    rgb[:, 2] = rgb[:, 2] + m

    rgb = rgb.reshape( shape )

    return rgb


# Accepts the following input from command line via sys.argv:
# • A filename                argv[1]
# • Hue value modification    argv[2]
# • Saturation modification   argv[3]
# • Value modification        argv[4]
# The hue input should be clamped to [0◦, 360◦]. 
# Saturation and value inputs should be within range [0, 1]. 
# If they are not, warn the user and exit the program. 
# Assuming all inputs are accepted, load and modify the image 
# using the functions you wrote above and save the modified image to a new file.
def main(argv, argc):

    if argc != 5:
        print("Improper number of command line arguments given. Exiting program.")
        return

    filename             = argv[1]
    hue_value_mod        = float(argv[2])
    saturation_value_mod = float(argv[3])
    value_modification   = float(argv[4])

    if hue_value_mod < 0 or hue_value_mod > 360:
        print("Given hue value modification is out of range [0, 1]. Exiting program.")
        return
    if saturation_value_mod < 0 or saturation_value_mod > 1:
        print("Given saturation value modification is out of range [0, 1]. Exiting program.")
        return
    if value_modification < 0 or value_modification > 1:
        print("Given value modification is out of range [0, 360]. Exiting program.")
        return

    # load image and convert to numpy array
    image = np.asarray( Image.open(filename) )

    # Capture image shape for reshaping later
    image_shape = image.shape

    # Conversion
    new_image, new_shape = rgb_to_hsv(image)
    image = hsv_to_rgb( new_image, new_shape )

    # show image (will save it later)
    plt.imshow(image)
    plt.show()


# Calling main()
if __name__ == "__main__":
    main( sys.argv, len(sys.argv) )
    # 
    #
    #
    #
    #
    #
    #
    #