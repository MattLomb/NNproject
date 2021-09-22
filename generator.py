import tensorflow as tf
#import clip
import kornia
from PIL import Image

from StyleGAN2.stylegan2_generator import StyleGan2Generator
from StyleGAN2.stylegan2_discriminator import StyleGan2Discriminator
import utils

#from utils import save_grid, freeze_model

class Generator:
    def __init__(self, config):
        self.config = config
        self.augmentation = None
        impl = 'ref'  # 'ref' if cuda is not available in your machine
        gpu = False  # False if tensorflow cpu is used
        weights_name = 'ffhq'  # face model trained by Nvidia
        # instantiating generator network
        generator = StyleGan2Generator(weights=weights_name, impl=impl, gpu=gpu)
        discriminator = StyleGan2Discriminator(weights=weights_name, impl=impl, gpu=gpu)
        
    def generate( self, ls ):
        z = ls() 
        result = generator( z )
        result = biggan_norm( result )
        return result
        
    def discriminate(self, images ):
        images = biggan_denorm( images )
        return discriminator( images )
