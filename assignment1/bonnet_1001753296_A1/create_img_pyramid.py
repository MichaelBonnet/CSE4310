from PIL import Image

def image_pyramid(img, size, filename):
    
    filename = filename.replace('.png', '')

    for x in range(1, size):
        
        H, W, L = img.shape

        resized_width  = round(W / (2 ** x) )
        resized_height = round(H / (2 ** x) )

        img_resize = Image.fromarray(img).resize((resized_width, resized_height), Image.NEAREST)

        img_resize.save(filename + "_" + str(2**x) + "x.png")
