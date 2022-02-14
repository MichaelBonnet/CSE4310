import sys
from tkinter import X
import numpy as np
import PIL
from PIL import Image
from matplotlib import pyplot as plt

# Takes an RGB image represented as a numpy array (rgb_image)
# And converts it into HSV format.
def rgb_to_hsv( rgb_image ):

    shape = rgb_image.shape
    
    rgb_image = rgb_image / 255            # Normalizing to [0, 1]
    rgb_image = rgb_image.reshape(-1, 3)   # Changing to a 3-column shape
    r = rgb_image[:, 0]                    # Extracting the R' values to a vector
    g = rgb_image[:, 1]                    # Extracting the G' values to a vector
    b = rgb_image[:, 2]                    # Extracting the B' values to a vector

    max = np.maximum(r, g, b)  # gets vector of the maximum of R', G', and B' for each pixel
    min = np.minimum(r, g, b)  # gets vector of the maximum of R', G', and B' for each pixel

    # Getting V
    v = max

    # Getting C
    c = v - min

    # Getting S
    s = c / v
    s[v==0] = 0  # catching case where S is undefined due to V being 0

    # Holding array for h
    h = np.zeros(rgb_image.shape[0])

    # Populating h
    # h[c==0] = 0
    # h[v==r] = ( ( g[v==r] - b[v==r] ) / c[v==r] ) % 6
    # h[v==g] = ( ( b[v==g] - r[v==g] ) / c[v==g] ) + 2
    # h[v==b] = ( ( r[v==b] - g[v==b] ) / c[v==b] ) + 4

    print("\n\nRange is "+str( len(h) )+"\n\n" )
    for i in range(len(h)):
        if c[i] == 0:
            h[i] = 0
        elif v[i] == r[i]:
            h[i] = ( ( g[i] - b[i] ) / c[i] ) % 6
        elif v[i] == g[i]:
            h[i] = ( ( b[i] - r[i] ) / c[i] ) + 2
        elif v[i] == b[i]:
            h[i] = ( ( r[i] - g[i] ) / c[i] ) + 4

    for i in range(len(h)):
        if not h[i].any():
            print("h at index "+str(i)+" is "+str(h[i]) )
    
    # Modifying to a degree value
    h = h * 60

    hsv_array = np.stack([h, s, v], axis=-1)

    image = hsv_array.reshape(shape)
    plt.imshow(image)
    plt.show()
    
    return hsv_array, shape


# Takes an HSV image represented as a numpy array (hsv_image) 
# And converts it into RGB format.
def hsv_to_rgb( hsv_image, shape ):
    
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

'''
Converting from RGB

assume input RGB from [0,1]; normalize it if needed
Value, V, = max(R,G,B)
^ NP.max across the entire image

Saturation based on chroma

Chroma, C = V - min(R,G,B)
    ^ max(R,G,B) - min(R,G,B)

Given C and V, saturation is the ratio between the two. Undefined if V=0
Saturation, S = C / V
    Will put 0 for S if V=0

Hue measured in [0, 360] degrees

H' =
    0, C=0
    ( (G - B) / C ) mod 6, V=R
    ( (B - R) / C ) + 2,   V=G
    ( (R - G) / C ) + 4,   V=B

at this point, H' will be some value between 0 and 6
the value is then adapted to range [0, 360]:

    H = 60 deg * H'

Converting TO RGB

Chroma (C) = V * S

Hue divided back into a value between [0, 6]:

    H' = H / 60

X = C * (1 - |H' % 2 - 1| )

then

(R', G', B') = 
    (C, X, 0) if 0 <= H' < 1
    (X, C, 0) if 1 <= H' < 2
    (0, C, X) if 2 <= H' < 3
    (0, X, C) if 3 <= H' < 4
    (X, 0, C) if 4 <= H' < 5
    (C, 0, X) if 5 <= H' < 6

then match the original RGB value by adding the same difference m = V - C to each channel:

    (R, G, B) = (R' + m, G' + m, B' + m)

'''