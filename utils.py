import tensorflow as tf
from matplotlib import pyplot as plt

def biggan_norm(images):
    images = (images + 1) / 2.0
    images = images.clip(0, 1)
    return images

def biggan_denorm(images):
    images = images*2 - 1
    return images