import warnings
from PIL import Image,ImageFilter

def down_sample_fit(pil_img, num):
    ds = [2 ** (i + 1) for i in range(num)]
    for d in ds:
        a, b = pil_img.size
        add = (a % d)
        na = a + add
        add = (b % d)
        nb = b + add
        pil_img = pil_img.resize((na, nb),Image.BICUBIC)
    return pil_img


def max_size_fit(pil_img, max_size):
    max_ = max(pil_img.size)
    if max_ <= max_size:
        return pil_img
    else:
        warnings.warn("The image is over the max size:{}".format(max_size))
        a, b = pil_img.size
        if a >= b:
            na = max_size
            nb = int(na * (b / a))
        else:
            nb = max_size
            na = int(nb * (a / b))

        return pil_img.resize((na, nb),Image.BICUBIC)



def blur_image(pil_img,mode,r):
    if mode == "g":
        kernel = ImageFilter.GaussianBlur(radius=r)
    elif mode == "m":
        kernel = ImageFilter.MedianFilter(size=r)
    else:
        raise ValueError("{} not support".format(mode))
    return pil_img.filter(kernel)