import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from StyleGAN2.utils.utils_stylegan2 import convert_images_to_uint8
from StyleGAN2.stylegan2_generator import StyleGan2Generator
from StyleGAN2.stylegan2_discriminator import StyleGan2Discriminator

parser = argparse.ArgumentParser()

parser.add_argument( "--config", type=srt, default="StyleGAN2_ffhq_d" )    #MODEL
parser.add_argument( "--generations", type=int, default=500 )       #Number of images generated
parser.add_argument( "--save-each", type=int, default=50 )          #Images saved each 50 generations
parser.add_argument("--tmp-folder", type=str, default="./tmp")      #Folder in which save the generated images
parser.add_argument("--target", type=str, default="a wolf at night with the moon in the background")    #txt2img

config = parser.parse_args()

iteration = 0

#def save_callback(algorithm):
    
problem = GenerationProblem( config )


def generate_and_save_images(images, it, plot_fig=True):
    plt.close()
    fig = plt.figure(figsize=(9, 9))

    for i in range(images.shape[0]):
        plt.subplot(2, 2, i + 1)
        plt.imshow(images[i])
        plt.axis('off')

    # tight_layout minimizes the overlap between 2 sub-plots
    fig.tight_layout()
    plt.savefig('StyleGAN2/images/generated/image_at_iter_{:04d}.png'.format(it))
    if plot_fig: plt.show()


impl = 'ref'  # 'ref' if cuda is not available in your machine
gpu = False  # False if tensorflow cpu is used
weights_name = 'ffhq'  # face model trained by Nvidia

# instantiating generator network
generator = StyleGan2Generator(weights=weights_name, impl=impl, gpu=gpu)
discriminator = StyleGan2Discriminator(weights=weights_name, impl=impl, gpu=gpu)

# creating random latent vector
rnd = np.random.RandomState()
z = rnd.randn(1, 512).astype('float32')
# z is the latent space

# running generator network
out_generator = generator(z)
out_discriminator = discriminator(out_generator)
print("Discriminator output with GENERATED IMAGE", out_discriminator)

# converting image to uint8
out_image = convert_images_to_uint8(out_generator, nchw_to_nhwc=True, uint8_cast=True)
generate_and_save_images(out_image.numpy(), 0)

target_size = 1024
file_path = "images/real_images/anne.jpg"
img = tf.io.read_file(file_path)
img = tf.io.decode_jpeg(img, channels=3)
img = tf.image.resize(img, [target_size, target_size])
img = tf.transpose(img, [2, 0, 1])  # Convert from [height, width, channel] to [channel, height, width]
img = img * 2 / 255.0 - 1  # Scale each pixel from [0,255] to [-1,1]
img = np.array([img])  # Create a bench

out_discriminator = discriminator(img)
print("Discriminator output with REAL IMAGE", out_discriminator)
