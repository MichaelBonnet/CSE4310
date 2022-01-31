import sys
import numpy as np
import PIL
from PIL import Image


def rgb2hsv_np(img_rgb):
    
    height, width, c = img_rgb.shape
    img_rgb = img_rgb / 255.0
    r = img_rgb[:,:,0]
    g = img_rgb[:,:,1]
    b = img_rgb[:,:,2]
    
    min_val = np.min(img_rgb, axis=-1)
    max_val = np.max(img_rgb, axis=-1)
    delta   = max_val - min_val
    v       = max_val
    
    
    s = delta / (max_val + 1e-6)
    s[v==0] = 0
    
    if delta != 0:
        # v==r
        hr = 60 * ( ( (g - b) / (delta + 1e-6) ) % 6 )
        # v==g
        hg = 60 * ( ( (b - r) / (delta + 1e-6) ) + 2 )
        # v==b
        hb = 60 * ( ( (r - g) / (delta + 1e-6) ) + 4 )

    h = np.zeros((height, width), np.float32)
    
    h = h.flatten()
    hr = hr.flatten()
    hg = hg.flatten()
    hb = hb.flatten()
    
    h[(v==b).flatten()] = hb[(v==b).flatten()]
    h[(v==g).flatten()] = hg[(v==g).flatten()]
    h[(v==r).flatten()] = hr[(v==r).flatten()]
    
    h[h<0] += 360
    
    h = h.reshape((height, width))
    
    img_hsv = np.stack([h, s, v], axis=-1)
    
    return img_hsv



# Takes an RGB image represented as a numpy array (rgb_image)
# And converts it into HSV format.
def rgb_to_hsv_1( rgb_image ):

    original_shape = rgb_image.shape
    rgb_image = rgb_image.reshape(-1, 3)
    r = rgb_image[:, 0] / 255  # normalizing r values to [0, 1]
    g = rgb_image[:, 1] / 255  # normalizing g values to [0, 1]
    b = rgb_image[:, 2] / 255  # normalizing b values to [0, 1]

    maxc = np.maximum(np.maximum(r, g), b)  # gets maximum between R', G', and B'
    minc = np.minimum(np.minimum(r, g), b)  # gets minimum between r, g, and b
    deltac = maxc - minc                    # gets difference between max(r,g,b) and min(r,g,b) [Δ]

    # getting V
    v = maxc

    # getting S
    s = deltac / maxc

    # getting h
    if deltac == 0:
        h = 0
    elif v == r:
        h = 60 * ( ( (g - b) / deltac ) % 6)
    elif v == g:
        h = 60 * ( ( (b - r) / deltac ) + 2)
    elif v == b:
        h = 60 * ( ( (r - g) / deltac ) + 4)

    res = np.dstack([h, s, v])
    return res.reshape(original_shape)

# Takes an HSV image represented as a numpy array (hsv_image) 
# And converts it into RGB format.
def hsv_to_rgb( hsv_image ):
    
    original_shape = hsv_image.shape
    hsv = hsv_image.reshape(-1, 3)
    h = hsv[:, 0]
    s = hsv[:, 1]
    v = hsv[:, 2]

    #####

    c = v * s
    h_prime = h / 60
    x = c * (1 - np.absolute(h_prime % 2 - 1) )
    m = v - c

    if 0 <= h_prime < 1:
        r_prime = c
        g_prime = x
        b_prime = 0
    elif 1 <= h_prime < 2:
        r_prime = x
        g_prime = c
        b_prime = 0
    elif 2 <= h_prime < 3:
        r_prime = 0
        g_prime = c
        b_prime = x
    elif 3 <= h_prime < 4:
        r_prime = 0
        g_prime = x
        b_prime = c
    elif 4 <= h_prime < 5:
        r_prime = x
        g_prime = 0
        b_prime = c
    elif 5 <= h_prime < 6:
        r_prime = c
        g_prime = 0
        b_prime = x

    r = r_prime + m
    g = g_prime + m
    b = b_prime + m

    #####

def rgb_to_hsv(rgb):
    """
    >>> from colorsys import rgb_to_hsv as rgb_to_hsv_single
    >>> 'h={:.2f} s={:.2f} v={:.2f}'.format(*rgb_to_hsv_single(50, 120, 239))
    'h=0.60 s=0.79 v=239.00'
    >>> 'h={:.2f} s={:.2f} v={:.2f}'.format(*rgb_to_hsv_single(163, 200, 130))
    'h=0.25 s=0.35 v=200.00'
    >>> np.set_printoptions(2)
    >>> rgb_to_hsv(np.array([[[50, 120, 239], [163, 200, 130]]]))
    array([[[   0.6 ,    0.79,  239.  ],
            [   0.25,    0.35,  200.  ]]])
    >>> 'h={:.2f} s={:.2f} v={:.2f}'.format(*rgb_to_hsv_single(100, 100, 100))
    'h=0.00 s=0.00 v=100.00'
    >>> rgb_to_hsv(np.array([[50, 120, 239], [100, 100, 100]]))
    array([[   0.6 ,    0.79,  239.  ],
           [   0.  ,    0.  ,  100.  ]])
    """
    input_shape = rgb.shape
    rgb = rgb.reshape(-1, 3)
    r, g, b = rgb[:, 0], rgb[:, 1], rgb[:, 2]

    maxc = np.maximum(np.maximum(r, g), b)
    minc = np.minimum(np.minimum(r, g), b)
    v = maxc

    deltac = maxc - minc
    s = deltac / maxc
    deltac[deltac == 0] = 1  # to not divide by zero (those results in any way would be overridden in next lines)
    rc = (maxc - r) / deltac
    gc = (maxc - g) / deltac
    bc = (maxc - b) / deltac

    h = 4.0 + gc - rc
    h[g == maxc] = 2.0 + rc[g == maxc] - bc[g == maxc]
    h[r == maxc] = bc[r == maxc] - gc[r == maxc]
    h[minc == maxc] = 0.0

    h = (h / 6.0) % 1.0
    res = np.dstack([h, s, v])
    return res.reshape(input_shape)

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

    

    # load the image
    # image = Image.open('uss_enterprise.jpeg')
    # convert image to numpy array
    image = np.asarray( Image.open('uss_enterprise.jpeg') )

    # image = rgb2hsv_np(image)
    image = rgb_to_hsv(image)

    from matplotlib import pyplot as plt
    plt.imshow(image, interpolation='nearest')
    plt.show()


    


if __name__ == "__main__":
    main( sys.argv, len(sys.argv) )