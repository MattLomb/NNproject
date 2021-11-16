import tensorflow as tf
from CLIP import CLIP
import numpy as np
from StyleGAN2.stylegan2_generator import StyleGan2Generator
from StyleGAN2.stylegan2_discriminator import StyleGan2Discriminator
import utils
import matplotlib.pyplot as plt

#Generating the problem related to the given target
class Generator:
    def __init__(self, config):
        self.config = config
        self.clip = CLIP.CLIP(True)

        impl = 'cuda'                   # 'ref' if cuda is not available in your machine
        gpu = True                      # False if tensorflow cpu is used
        weights_name = 'ffhq'           # face model trained by Nvidia
        # instantiating StyleGan2 generator and discriminator network
        self.generator = StyleGan2Generator(weights=weights_name, impl=impl, gpu=gpu)
        if self.config.use_discriminator:
            self.discriminator = StyleGan2Discriminator(weights=weights_name, impl=impl, gpu=gpu)

        self.text_features = self.clip.predict_text(self.config.target)

    #Generation of the images from the latent space
    def generate(self, ls):
        z = ls
        result = self.generator(z)
        result = utils.norm(result)
        return result

    #Discriminate the images
    def discriminate(self, images):
        if self.config.use_discriminator:
            images = utils.denorm(images)
            return self.discriminator(images)

    #Modifies the given image in order to be processed by CLIP
    def process_image(self, image):
        clip_target_size = 218
        img = tf.transpose(image, [0, 2, 3, 1])     # CHW -> HWC
        img = tf.image.resize(img, [clip_target_size, clip_target_size])
        img *= 255
        img = img.numpy().astype(float)
        return img

    #Function that computes the similarity between the given caption and the generated images
    def clip_similarity(self, input):
        processed_image = self.process_image(input)
        image_features = self.clip.predict_image(processed_image)

        return tf.keras.losses.cosine_similarity(image_features, self.text_features)

    #Save the generated images inside the folder specified by the given path
    def save(self, input, path):
        input = np.asarray(input)
        input = np.rollaxis(input, 1, 4)
        input_size = len(input)
        grid_x = 1
        grid_y = input_size
        plt.figure(figsize=(grid_x, grid_y))  # specifying the overall grid size
        figure = plt.gcf()  # get current figure
        figure.set_size_inches(16, 4)
        for i in range(input_size):
            plt.subplot(grid_x, grid_y, i + 1)  # the number of images in the grid is 5*5 (25)
            plt.imshow(input[i], interpolation='nearest')

        plt.savefig(path, dpi=100)

