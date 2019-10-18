"""Functions for generating image features related to lumination"""

__author__ = "Victor Mawusi Ayi"


from cv2 import (
    cvtColor,
    COLOR_BGR2HLS,
    COLOR_RGB2GRAY
)
import matplotlib.image as mpimg
from numpy import (
    power,
    sqrt,
    sum
)


def pixelcount(rgb_image):
    i, j, _ = rgb_image.shape
    tpixels = i * j

    return tpixels

def luminance(rgb_image):

    pixelcnt = pixelcount(rgb_image)
    lum_coefs = 0.27, 0.67, 0.06
    lumnance = 0

    for i in range(3):
       avg_pixel_int = sum(rgb_image[:,:,i])/pixelcnt
       lumnance += avg_pixel_int * lum_coefs[i]

    return lumnance

def contrast(rgb_image):

    gray = cvtColor(rgb_image, COLOR_RGB2GRAY)
    pixelcnt = pixelcount(rgb_image)
    
    mean_diff_sq = power((gray - (gray/pixelcnt)), 2)
    stdev = sqrt(
        sum(mean_diff_sq)/pixelcnt
    )

    return stdev

def supracrop(rgb_image, divisor):
    topfourth = int(rgb_image.shape[1]/divisor)
    supracrop = rgb_image[:topfourth, :]

    return supracrop

def supraluminance(rgb_image, divisor=4):
    scrop = supracrop(rgb_image, divisor)

    return luminance(scrop)

def lightness(rgb_image):

    pixelcnt = pixelcount(rgb_image)
    hls = cvtColor(rgb_image, COLOR_BGR2HLS)

    avg_lightness = sum(hls[:,:,1])/pixelcnt
    
    return avg_lightness

def supralightness(rgb_image, divisor=3):
    scrop = supracrop(rgb_image, divisor)

    return lightness(scrop)

def supracontrast(rgb_image, divisor=3):
    scrop = supracrop(rgb_image, divisor)

    return contrast(scrop)
