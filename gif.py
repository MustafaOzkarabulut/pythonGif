import numpy as np
from numpngw import write_apng

bits1 = np.array([
    [0,0,0,0,0,0,1],
    [0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0],
    ])

bits2 = np.array([
    [0,0,0,0,0,1,0],
    [0,0,0,0,0,0,1],
    [0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0],
    ])

bits3 = np.array([
    [0,0,0,0,1,0,0],
    [0,0,0,0,0,1,0],
    [0,0,0,0,0,0,1],
    [0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0],
    ])


bits4 = np.array([
    [0,0,0,1,0,0,0],
    [0,0,0,0,1,0,0],
    [0,0,0,0,0,1,0],
    [0,0,0,0,0,0,1],
    [0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0],
    ])

bits5 = np.array([
    [0,0,1,0,0,0,0],
    [0,0,0,1,0,0,0],
    [0,0,0,0,1,0,0],
    [0,0,0,0,0,1,0],
    [0,0,0,0,0,0,1],
    [0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0],
    ])

bits6 = np.array([
    [0,1,0,0,0,0,0],
    [0,0,1,0,0,0,0],
    [0,0,0,1,0,0,0],
    [0,0,0,0,1,0,0],
    [0,0,0,0,0,1,0],
    [0,0,0,0,0,0,1],
    [0,0,0,0,0,0,0],
    ])

bits7 = np.array([
    [1,0,0,0,0,0,0],
    [0,1,0,0,0,0,0],
    [0,0,1,0,0,0,0],
    [0,0,0,1,0,0,0],
    [0,0,0,0,1,0,0],
    [0,0,0,0,0,1,0],
    [0,0,0,0,0,0,1],
    ])

bits8 = np.array([
    [0,0,0,0,0,0,0],
    [1,0,0,0,0,0,0],
    [0,1,0,0,0,0,0],
    [0,0,1,0,0,0,0],
    [0,0,0,1,0,0,0],
    [0,0,0,0,1,0,0],
    [0,0,0,0,0,1,0],
    ])

bits9 = np.array([
    [0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0],
    [1,0,0,0,0,0,0],
    [0,1,0,0,0,0,0],
    [0,0,1,0,0,0,0],
    [0,0,0,1,0,0,0],
    [0,0,0,0,1,0,0],
    ])

bits10 = np.array([
    [0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0],
    [1,0,0,0,0,0,0],
    [0,1,0,0,0,0,0],
    [0,0,1,0,0,0,0],
    [0,0,0,1,0,0,0],
    ])

bits11 = np.array([
    [0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0],
    [1,0,0,0,0,0,0],
    [0,1,0,0,0,0,0],
    [0,0,1,0,0,0,0],
    ])

bits12 = np.array([
    [0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0],
    [1,0,0,0,0,0,0],
    [0,1,0,0,0,0,0],
    ])  

bits13 = np.array([
    [0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0],
    [1,0,0,0,0,0,0],
    ])  

bits14 = np.array([
    [1,0,0,0,0,0,1],
    [1,0,0,0,0,0,1],
    [1,0,0,0,0,0,1],
    [1,1,1,1,1,1,1],
    [1,0,0,0,0,0,1],
    [1,0,0,0,0,0,1],
    [1,0,0,0,0,0,1],
    ])

bits15 = np.array([
    [0,0,0,1,0,0,0],
    [0,0,0,0,0,0,0],
    [0,0,0,1,0,0,0],
    [0,0,0,1,0,0,0],
    [0,0,0,1,0,0,0],
    [0,0,0,1,0,0,0],
    [0,0,0,1,0,0,0],
    ])

bits16 = np.array([
    [0,0,0,1,0,0,0],
    [0,0,0,1,0,0,0],
    [0,0,0,1,0,0,0],
    [0,0,0,1,0,0,0],
    [0,0,0,1,0,0,0],
    [0,0,0,0,0,0,0],
    [0,0,0,1,0,0,0],
    ])    

bits_zeros = np.zeros((7, 7), dtype=bool)
bits_ones = np.ones((7, 7), dtype=bool)


def bits_to_image(bits, blocksize=32, color=None):
    bits = np.asarray(bits, dtype=np.bool)
    if color is None:
        color = np.array([255, 0, 0], dtype=np.uint8)
    else:
        color = np.asarray(color, dtype=np.uint8)

    x = np.linspace(0, 0, blocksize)
    X, Y = np.meshgrid(x, x)
    Z = np.sqrt(np.maximum(1 - (X**16 + Y**16), 0))

    img1 = (Z.reshape(blocksize, blocksize, 1)*color)

    img0 = 0.2*img1

    data = np.where(bits[:, None, :, None, None],
                    img1[:, None, :], img0[:, None, :])
    img = data.reshape(bits.shape[0]*blocksize, bits.shape[1]*blocksize, 3)
    return img.astype(np.uint8)


color = np.array([0, 255, 255])

colora = np.array([255, 255, 255])

colorb = np.array([255,0,0])

colorc = np.array([255,255,0])

blocksize = 40

im3 = bits_to_image(bits3, blocksize=blocksize, color=color)
im2 = bits_to_image(bits2, blocksize=blocksize, color=color)
im1 = bits_to_image(bits1, blocksize=blocksize, color=color)
im_all = bits_to_image(bits_ones, blocksize=blocksize, color=color)
im_none = bits_to_image(bits_zeros, blocksize=blocksize, color=color)
im4 = bits_to_image(bits4, blocksize=blocksize, color=color)
im5 = bits_to_image(bits5, blocksize=blocksize, color=color)
im6 = bits_to_image(bits6, blocksize=blocksize, color=color)
im7 = bits_to_image(bits7, blocksize=blocksize, color=color)
im8 = bits_to_image(bits8, blocksize=blocksize, color=color)
im9 = bits_to_image(bits9, blocksize=blocksize, color=color)
im10 = bits_to_image(bits10, blocksize=blocksize, color=color)
im11 = bits_to_image(bits11, blocksize=blocksize, color=color)
im12 = bits_to_image(bits12, blocksize=blocksize, color=color)
im13 = bits_to_image(bits13, blocksize=blocksize, color=color)
im14 = bits_to_image(bits14, blocksize=blocksize, color=colorb)
im15 = bits_to_image(bits15, blocksize=blocksize, color=colora)
im16 = bits_to_image(bits16, blocksize=blocksize, color=colorc)


seq = [im1, im2, im3,im4,im5,im6,im7, im8, im9,im10,im11,im12,im13,im12, im11, im10,im9,im8,im7,im6, im5, im4,im3,im2,im1,im14,im15,im16, im_all, im_none, im_all, im_none,
       im_none, im_all, im_none]

delay = [5, 5, 5, 5, 5, 5, 5, 5,5, 5, 5, 5, 5, 5, 5,5, 5, 5, 5, 5, 5,
         5, 5, 5,5,500,500,500,5,5,5,5,5,5,5]


write_apng("1.gif", seq, delay=delay, default_image=im_all,
           use_palette=True)