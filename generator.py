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
        self.augmentation = None
        self.clip = CLIP.CLIP(True)

        impl = 'cuda'                   # 'ref' if cuda is not available in your machine
        gpu = True                      # False if tensorflow cpu is used
        weights_name = 'ffhq'           # face model trained by Nvidia
        # instantiating StyleGan2 generator and discriminator network
        self.generator = StyleGan2Generator(weights=weights_name, impl=impl, gpu=gpu)
        if self.config.use_discriminator:
            self.discriminator = StyleGan2Discriminator(weights=weights_name, impl=impl, gpu=gpu)

    #Generation of the images from the latent space
    def generate(self, ls):
        z = ls
        result = self.generator(z)
        result = utils.biggan_norm(result)
        return result

    #Discriminate the images
    def discriminate(self, images):
        if self.config.use_discriminator:
            images = utils.biggan_denorm(images)
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
        if self.augmentation is not None:
            input = self.augmentation(input)

        text_features = self.config.target

        processed_image = self.process_image(input)

        aggregated_prediction = False
        if aggregated_prediction:
            image_features, text_features = self.clip.predict(processed_image, text_features)
        else:
            image_features = self.clip.predict_image(processed_image)
            text_features = self.clip.predict_text(text_features)

        #cosine_loss = tf.keras.losses.cosine_similarity(image_features, text_features) -> first version of the implementation
        image_features = tf.math.l2_normalize(image_features, axis=1)
        text_features = tf.math.l2_normalize(text_features, axis=1)
        dot_prod = tf.matmul(text_features, image_features, transpose_b=True)   #Function used instead of cosine_similarity
        return dot_prod[0]

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

