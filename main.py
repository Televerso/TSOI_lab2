import cv2
import matplotlib.pyplot as plt
import numpy as np

def normalize(image):
    return (255 * ((image-image.min()) / (image.max()-image.min()))).astype(np.uint8)

def show_image(image):
    # image = normalize(image).astype('uint8')
    plt.figure()
    plt.imshow(image, cmap='gray',vmin=0,vmax=256)
    plt.show()

def to_grayscale(image):
    image_gray = np.zeros([image.shape[0],image.shape[1]], dtype=np.uint8)
    image_gray[:,:] = image[:,:,0]
    return image_gray

# Функция для удаления значений, выходящих за промежуток [0,255]
def remove_outliers(image):
    image = np.clip(image, 0, 255).astype(np.uint8)
    return image


def show_hist(hist, doprint=False):
    if doprint:
        print(hist)

    plt.figure()
    plt.bar(range(len(hist)), hist.tolist())
    plt.show()


def norm_hist(image):
    image_norm = normalize(image)
    hist = np.zeros(256, dtype=float)
    N = image.shape[0] * image.shape[1]
    keys, vals = np.unique(image_norm, return_counts=True)

    for key, value in zip(keys, vals):
        hist[key] = value
    return hist / N

def pixel_conversion(image,func):
    image_new = np.zeros_like(image)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            image_new[i,j] = func(image[i,j])
    return image_new

def get_integral_function(image):
    hist = norm_hist(image)
    F = np.cumsum(hist)
    return F

def equalize_gist_of_image(image,g_min,g_max):
    F = get_integral_function(image)
    g = np.zeros_like(F, dtype=float)
    for i in range(256):
        g[i] = ((g_max-g_min)*F[i]+g_min)
    print(g.min(),g.max())
    image_new = pixel_conversion(image,lambda x:g[x])
    return image_new

image_orig = cv2.resize(cv2.imread('img3.png',1).astype('uint8')[:,:,::-1], (256,256))
show_image(image_orig)
image_gray = to_grayscale(image_orig)


def prep_image(image, g_min, g_max):
    F = get_integral_function(image)
    g = np.zeros_like(F, dtype=float)
    for i in range(256):
        if i < g_min:
            g[i] = 255
        elif i > g_max:
            g[i] = i
        else:
            g[i] = (255) * F[i]
    show_hist(g)

    image_new = pixel_conversion(image, lambda x: g[x])
    return image_new


image_prep = prep_image(image_gray, 100, 155)
show_image(image_gray)
show_image(image_prep)
show_hist(norm_hist(image_gray))
show_hist(norm_hist(image_prep))
show_hist(get_integral_function(image_prep))