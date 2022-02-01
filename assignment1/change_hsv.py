import sys
import numpy as np
import PIL
from PIL import Image
from matplotlib import pyplot as plt

# from Nikolay Polyarni

# def rgb_to_hsv(rgb):

#     input_shape = rgb.shape
#     rgb = rgb.reshape(-1, 3)
#     r, g, b = rgb[:, 0], rgb[:, 1], rgb[:, 2]

#     maxc = np.maximum(np.maximum(r, g), b)
#     minc = np.minimum(np.minimum(r, g), b)
#     v = maxc

#     deltac = maxc - minc
#     s = deltac / maxc
#     deltac[deltac == 0] = 1  # to not divide by zero (those results in any way would be overridden in next lines)
#     rc = (maxc - r) / deltac
#     gc = (maxc - g) / deltac
#     bc = (maxc - b) / deltac

#     h = 4.0 + gc - rc
#     h[g == maxc] = 2.0 + rc[g == maxc] - bc[g == maxc]
#     h[r == maxc] = bc[r == maxc] - gc[r == maxc]
#     h[minc == maxc] = 0.0

#     h = (h / 6.0) % 1.0
#     res = np.dstack([h, s, v])
#     return res.reshape(input_shape)


# def hsv_to_rgb(hsv):

#     input_shape = hsv.shape
#     hsv = hsv.reshape(-1, 3)
#     h, s, v = hsv[:, 0], hsv[:, 1], hsv[:, 2]

#     i = np.int32(h * 6.0)
#     f = (h * 6.0) - i
#     p = v * (1.0 - s)
#     q = v * (1.0 - s * f)
#     t = v * (1.0 - s * (1.0 - f))
#     i = i % 6

#     rgb = np.zeros_like(hsv)
#     v, t, p, q = v.reshape(-1, 1), t.reshape(-1, 1), p.reshape(-1, 1), q.reshape(-1, 1)
#     rgb[i == 0] = np.hstack([v, t, p])[i == 0]
#     rgb[i == 1] = np.hstack([q, v, p])[i == 1]
#     rgb[i == 2] = np.hstack([p, v, t])[i == 2]
#     rgb[i == 3] = np.hstack([p, q, v])[i == 3]
#     rgb[i == 4] = np.hstack([t, p, v])[i == 4]
#     rgb[i == 5] = np.hstack([v, p, q])[i == 5]
#     rgb[s == 0.0] = np.hstack([v, v, v])[s == 0.0]

#     return rgb.reshape(input_shape)

# Takes an RGB image represented as a numpy array (rgb_image)
# And converts it into HSV format.
def rgb_to_hsv_1( rgb_image ):

    height, width, c = rgb_image.shape
    rgb_image = rgb_image / 255.0
    r = rgb_image[:,:,0]
    g = rgb_image[:,:,1]
    b = rgb_image[:,:,2]

    maxc = np.maximum(np.maximum(r, g), b)  # gets maximum between R', G', and B'
    minc = np.minimum(np.minimum(r, g), b)  # gets minimum between R', G', and B'
    deltac = maxc - minc                    # gets difference between max(r,g,b) and min(r,g,b) [Δ]

    # getting V
    v = maxc

    # getting S
    s = deltac / maxc
    s[v==0] = 0

    # getting h
    h = np.zeros((height, width), np.float32)

    h[deltac==0] = 0
    h[v==r] = 60 * ( ( (g[v==r] - b[v==r]) / deltac ) % 6)
    h[v==g] = 60 * ( ( (b[v==g] - r[v==g]) / deltac ) + 2)
    h[v==b] = 60 * ( ( (r[v==b] - g[v==b]) / deltac ) + 4)

    res = np.stack([h, s, v], axis=-1)
    return res

# Takes an HSV image represented as a numpy array (hsv_image) 
# And converts it into RGB format.
def hsv_to_rgb_1( hsv_image ):
    
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

##################



def rgb_to_hsv(image):

    image = image / 255
    image = image.reshape(-1, 3)
    r = image[:, 0]
    g = image[:, 1]
    b = image[:, 2]

    max = np.maximum(r, g, b)
    min = np.minimum(r, g, b)

    v = max
    c = v - min
    s = c / v
    s[v==0] = 0
    print("\nprinting v\n")
    print(v)
    print("\nprinting s\n")
    print(s)

    '''
    H' =
    0, C=0
    ( (G - B) / C ) mod 6, V=R
    ( (B - R) / C ) + 2,   V=G
    ( (R - G) / C ) + 4,   V=B
    '''

    h = np.zeros(image.shape[0])
    # h = h.reshape(-1, 1)
    print("\nprinting h\n")
    print(h)
    h[c==0] = 0
    h[v==r] = ( (g[v==r] - b[v==r]) / c ) % 6
    h[v==g] = ( (b[v==g] - r[v==g]) / c ) + 2
    h[v==b] = ( (r[v==b] - g[v==b]) / c ) + 4
    
    h = h * 60

    res = np.stack([h, s, v], axis=-1)
    print(res)


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
    # print(image[305,223])
    # print(image[259,127])
    print(image)

    image = image.reshape(-1, 3)
    print("\n\n\n\n\n\n\n")
    print(image)

    # choice = input("\nWould you like to convert the image from RGB to HSV (1) or from HSV to RGB (2)? : ")

    # if choice == "1":
    image = rgb_to_hsv(image)
    #     # print(image)
    #     # image = hsv_to_rgb(image)
    # elif choice == "2":
    #     image = hsv_to_rgb(image)

    # show image (will save it later)
    plt.imshow(image)
    # plt.imshow('uss_enterprise.jpeg')
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