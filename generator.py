import tensorflow as tf
from CLIP import CLIP

from StyleGAN2.stylegan2_generator import StyleGan2Generator
from StyleGAN2.stylegan2_discriminator import StyleGan2Discriminator
import utils


class Generator:
    def __init__(self, config):
        self.config = config
        self.augmentation = None
        self.clip = CLIP.CLIP()

        impl = 'ref'  # 'ref' if cuda is not available in your machine
        gpu = False  # False if tensorflow cpu is used
        weights_name = 'ffhq'  # face model trained by Nvidia
        # instantiating generator network
        self.generator = StyleGan2Generator(weights=weights_name, impl=impl, gpu=gpu)
        self.discriminator = StyleGan2Discriminator(weights=weights_name, impl=impl, gpu=gpu)

    def generate(self, ls):
        z = ls
        result = self.generator(z)
        result = utils.biggan_norm(result)
        return result

    def discriminate(self, images):
        images = utils.biggan_denorm(images)
        return self.discriminator(images)

    def process_image(self, image):
        clip_target_size = 224
        print("QUI1", image.shape)

        img = tf.transpose(image, [0, 2, 3, 1])  # C HW -> HWC
        img = tf.image.resize(img, [clip_target_size, clip_target_size])
        img = img.numpy().astype(float)
        img /= 255

        print("QUI2", img.shape)

        img = utils.normalize_image(img,  # PyTorch does it
                                    (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))

        return img

    def clip_similarity(self, input):
        if self.augmentation is not None:
            input = self.augmentation(input)

        text_features = self.config.target
        tokenized_text = self.clip.tokenize(text_features)
        processed_image = self.process_image(input)

        text_features = self.clip.predict_text(tokenized_text)
        image_features = self.clip.predict_image(processed_image)

        cosine_loss = tf.keras.losses.cosine_similarity(image_features, text_features)

        return cosine_loss
