import sys
from tkinter import X
import numpy as np
import PIL
from PIL import Image
from matplotlib import pyplot as plt

# Takes an RGB image represented as a numpy array (rgb_image)
# And converts it into HSV format.
def rgb_to_hsv_1( rgb_image ):

    height, width, c = rgb_image.shape
    rgb_image = rgb_image / 255.0
    r = rgb_image[:,:,0]
    g = rgb_image[:,:,1]
    b = rgb_image[:,:,2]

    maxc = np.maximum(np.maximum(r, g), b)  # gets vector of the maximum of R', G', and B' for each pixel
    minc = np.minimum(np.minimum(r, g), b)  # gets vector of the maximum of R', G', and B' for each pixel
    deltac = maxc - minc                    # gets difference between max(r,g,b) and min(r,g,b) [Δ]

    # getting V
    v = maxc

    # getting S
    s = deltac / v   # proper equation (rather than deltac/maxc)
    s[v==0] = 0      # catching case where S is undefined due to v being 0

    # getting h
    h = np.zeros((height, width), np.float32)

    h[deltac==0] = 0
    h[v==r] = 60 * ( ( (g[v==r] - b[v==r]) / deltac ) % 6)
    h[v==g] = 60 * ( ( (b[v==g] - r[v==g]) / deltac ) + 2)
    h[v==b] = 60 * ( ( (r[v==b] - g[v==b]) / deltac ) + 4)

    res = np.stack([h, s, v], axis=-1)
    return res


# Takes an RGB image represented as a numpy array (rgb_image)
# And converts it into HSV format.
def rgb_to_hsv_2( rgb_image ):

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

    h = np.zeros(rgb_image.shape[0])
    print("\nprinting h\n")
    print(h)
    h[c==0] = 0
    h[v==r] = ( ( g[v==r] - b[v==r] ) / c ) % 6
    h[v==g] = ( ( b[v==g] - r[v==g] ) / c ) + 2
    h[v==b] = ( ( r[v==b] - g[v==b] ) / c ) + 4
    
    h = h * 60

    hsv_array = np.stack([h, s, v], axis=-1)
    
    return hsv_array


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

    rgb[0 <= h_prime < 1] = np.hstack([c, x, 0])[0 <= h_prime < 1]
    rgb[1 <= h_prime < 2] = np.hstack([x, c, 0])[1 <= h_prime < 2]
    rgb[2 <= h_prime < 3] = np.hstack([0, c, x])[2 <= h_prime < 3]
    rgb[3 <= h_prime < 4] = np.hstack([0, x, c])[3 <= h_prime < 4]
    rgb[4 <= h_prime < 5] = np.hstack([x, 0, c])[4 <= h_prime < 5]
    rgb[5 <= h_prime < 6] = np.hstack([c, 0, x])[5 <= h_prime < 6]

    # Getting M for final modification
    m = v - c

    # Adding m back into each channel
    rgb[:, 0] = rgb[:, 0] + m
    rgb[:, 1] = rgb[:, 1] + m
    rgb[:, 2] = rgb[:, 2] + m

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
    image_mod1 = rgb_to_hsv_1(image)
    image_mod2 = hsv_to_rgb(image_mod1)

    # show image (will save it later)
    plt.imshow(image_mod2)
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