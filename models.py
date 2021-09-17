import os
import sys
import tensorflow as tf
import pickle

gdrive_urls = {
    'gdrive:networks/stylegan2-car-config-a.pkl': 'https://nvlabs-fi-cdn.nvidia.com/stylegan2/networks/stylegan2-car-config-a.pkl',
    'gdrive:networks/stylegan2-car-config-b.pkl': 'https://nvlabs-fi-cdn.nvidia.com/stylegan2/networks/stylegan2-car-config-b.pkl',
    'gdrive:networks/stylegan2-car-config-c.pkl': 'https://nvlabs-fi-cdn.nvidia.com/stylegan2/networks/stylegan2-car-config-c.pkl',
    'gdrive:networks/stylegan2-car-config-d.pkl': 'https://nvlabs-fi-cdn.nvidia.com/stylegan2/networks/stylegan2-car-config-d.pkl',
    'gdrive:networks/stylegan2-car-config-e.pkl': 'https://nvlabs-fi-cdn.nvidia.com/stylegan2/networks/stylegan2-car-config-e.pkl',
    'gdrive:networks/stylegan2-car-config-f.pkl': 'https://nvlabs-fi-cdn.nvidia.com/stylegan2/networks/stylegan2-car-config-f.pkl',
    'gdrive:networks/stylegan2-cat-config-a.pkl': 'https://nvlabs-fi-cdn.nvidia.com/stylegan2/networks/stylegan2-cat-config-a.pkl',
    'gdrive:networks/stylegan2-cat-config-f.pkl': 'https://nvlabs-fi-cdn.nvidia.com/stylegan2/networks/stylegan2-cat-config-f.pkl',
    'gdrive:networks/stylegan2-church-config-a.pkl': 'https://nvlabs-fi-cdn.nvidia.com/stylegan2/networks/stylegan2-church-config-a.pkl',
    'gdrive:networks/stylegan2-church-config-f.pkl': 'https://nvlabs-fi-cdn.nvidia.com/stylegan2/networks/stylegan2-church-config-f.pkl',
    'gdrive:networks/stylegan2-ffhq-config-a.pkl': 'https://nvlabs-fi-cdn.nvidia.com/stylegan2/networks/stylegan2-ffhq-config-a.pkl',
    'gdrive:networks/stylegan2-ffhq-config-b.pkl': 'https://nvlabs-fi-cdn.nvidia.com/stylegan2/networks/stylegan2-ffhq-config-b.pkl',
    'gdrive:networks/stylegan2-ffhq-config-c.pkl': 'https://nvlabs-fi-cdn.nvidia.com/stylegan2/networks/stylegan2-ffhq-config-c.pkl',
    'gdrive:networks/stylegan2-ffhq-config-d.pkl': 'https://nvlabs-fi-cdn.nvidia.com/stylegan2/networks/stylegan2-ffhq-config-d.pkl',
    'gdrive:networks/stylegan2-ffhq-config-e.pkl': 'https://nvlabs-fi-cdn.nvidia.com/stylegan2/networks/stylegan2-ffhq-config-e.pkl',
    'gdrive:networks/stylegan2-ffhq-config-f.pkl': 'https://nvlabs-fi-cdn.nvidia.com/stylegan2/networks/stylegan2-ffhq-config-f.pkl',
    'gdrive:networks/stylegan2-horse-config-a.pkl': 'https://nvlabs-fi-cdn.nvidia.com/stylegan2/networks/stylegan2-horse-config-a.pkl',
    'gdrive:networks/stylegan2-horse-config-f.pkl': 'https://nvlabs-fi-cdn.nvidia.com/stylegan2/networks/stylegan2-horse-config-f.pkl',
    'gdrive:networks/table2/stylegan2-car-config-e-Gorig-Dorig.pkl': 'https://nvlabs-fi-cdn.nvidia.com/stylegan2/networks/table2/stylegan2-car-config-e-Gorig-Dorig.pkl',
    'gdrive:networks/table2/stylegan2-car-config-e-Gorig-Dresnet.pkl': 'https://nvlabs-fi-cdn.nvidia.com/stylegan2/networks/table2/stylegan2-car-config-e-Gorig-Dresnet.pkl',
    'gdrive:networks/table2/stylegan2-car-config-e-Gorig-Dskip.pkl': 'https://nvlabs-fi-cdn.nvidia.com/stylegan2/networks/table2/stylegan2-car-config-e-Gorig-Dskip.pkl',
    'gdrive:networks/table2/stylegan2-car-config-e-Gresnet-Dorig.pkl': 'https://nvlabs-fi-cdn.nvidia.com/stylegan2/networks/table2/stylegan2-car-config-e-Gresnet-Dorig.pkl',
    'gdrive:networks/table2/stylegan2-car-config-e-Gresnet-Dresnet.pkl': 'https://nvlabs-fi-cdn.nvidia.com/stylegan2/networks/table2/stylegan2-car-config-e-Gresnet-Dresnet.pkl',
    'gdrive:networks/table2/stylegan2-car-config-e-Gresnet-Dskip.pkl': 'https://nvlabs-fi-cdn.nvidia.com/stylegan2/networks/table2/stylegan2-car-config-e-Gresnet-Dskip.pkl',
    'gdrive:networks/table2/stylegan2-car-config-e-Gskip-Dorig.pkl': 'https://nvlabs-fi-cdn.nvidia.com/stylegan2/networks/table2/stylegan2-car-config-e-Gskip-Dorig.pkl',
    'gdrive:networks/table2/stylegan2-car-config-e-Gskip-Dresnet.pkl': 'https://nvlabs-fi-cdn.nvidia.com/stylegan2/networks/table2/stylegan2-car-config-e-Gskip-Dresnet.pkl',
    'gdrive:networks/table2/stylegan2-car-config-e-Gskip-Dskip.pkl': 'https://nvlabs-fi-cdn.nvidia.com/stylegan2/networks/table2/stylegan2-car-config-e-Gskip-Dskip.pkl',
    'gdrive:networks/table2/stylegan2-ffhq-config-e-Gorig-Dorig.pkl': 'https://nvlabs-fi-cdn.nvidia.com/stylegan2/networks/table2/stylegan2-ffhq-config-e-Gorig-Dorig.pkl',
    'gdrive:networks/table2/stylegan2-ffhq-config-e-Gorig-Dresnet.pkl': 'https://nvlabs-fi-cdn.nvidia.com/stylegan2/networks/table2/stylegan2-ffhq-config-e-Gorig-Dresnet.pkl',
    'gdrive:networks/table2/stylegan2-ffhq-config-e-Gorig-Dskip.pkl': 'https://nvlabs-fi-cdn.nvidia.com/stylegan2/networks/table2/stylegan2-ffhq-config-e-Gorig-Dskip.pkl',
    'gdrive:networks/table2/stylegan2-ffhq-config-e-Gresnet-Dorig.pkl': 'https://nvlabs-fi-cdn.nvidia.com/stylegan2/networks/table2/stylegan2-ffhq-config-e-Gresnet-Dorig.pkl',
    'gdrive:networks/table2/stylegan2-ffhq-config-e-Gresnet-Dresnet.pkl': 'https://nvlabs-fi-cdn.nvidia.com/stylegan2/networks/table2/stylegan2-ffhq-config-e-Gresnet-Dresnet.pkl',
    'gdrive:networks/table2/stylegan2-ffhq-config-e-Gresnet-Dskip.pkl': 'https://nvlabs-fi-cdn.nvidia.com/stylegan2/networks/table2/stylegan2-ffhq-config-e-Gresnet-Dskip.pkl',
    'gdrive:networks/table2/stylegan2-ffhq-config-e-Gskip-Dorig.pkl': 'https://nvlabs-fi-cdn.nvidia.com/stylegan2/networks/table2/stylegan2-ffhq-config-e-Gskip-Dorig.pkl',
    'gdrive:networks/table2/stylegan2-ffhq-config-e-Gskip-Dresnet.pkl': 'https://nvlabs-fi-cdn.nvidia.com/stylegan2/networks/table2/stylegan2-ffhq-config-e-Gskip-Dresnet.pkl',
    'gdrive:networks/table2/stylegan2-ffhq-config-e-Gskip-Dskip.pkl': 'https://nvlabs-fi-cdn.nvidia.com/stylegan2/networks/table2/stylegan2-ffhq-config-e-Gskip-Dskip.pkl',
}


class StyleGAN2(tf.Module):

    def __init__(self, config):
        #super(StyleGAN2, self).__init__()
        '''
        if not os.path.exists(os.path.join(config.weights, "G.pth")):
            model = "ffhq"
            print("Weights not found!")
        '''

        # self.G = stylegan2.models.load(os.path.join(config.weights, "G.pth")) #Generator
        # self.D = stylegan2.models.load(os.path.join(config.weights, "D.pth")) #Discriminator
        self.G, self.D, self.Gs = self.load_model("gdrive:networks/stylegan2-ffhq-config-f.pkl")

        return 0

    def get_path_or_url(self, path_or_gdrive_path):
        return gdrive_urls.get(path_or_gdrive_path, path_or_gdrive_path)

    def load_model(self, path_or_gdrive_path):
        path_or_url = self.get_path_or_url(path_or_gdrive_path)

        ##if dnnlib.util.is_url(path_or_url):
        ##     stream = dnnlib.util.open_url(path_or_url, cache_dir='.stylegan2-cache')
        ##else:
        stream = open("stylegan2-ffhq-config-d.pkl", 'rb')

        with stream:
            G, D, Gs = pickle.load(stream, encoding='latin1')
        return G, D, Gs

    def has_discriminator(self):
        return True

    # def generate(self, z, minibatch = None):
