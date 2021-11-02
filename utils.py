import tensorflow as tf
from matplotlib import pyplot as plt
import numpy as np

#Normalization of the images in intervals between 0 and 1
def biggan_norm(images):
    images = (images + 1) / 2.0
    images = np.clip(images, 0, 1)
    return images

#Denormalization of the given images
def biggan_denorm(images):
    images = images * 2 - 1
    return images