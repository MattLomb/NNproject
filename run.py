import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from StyleGAN2.utils.utils_stylegan2 import convert_images_to_uint8
from StyleGAN2.stylegan2_generator import StyleGan2Generator
from StyleGAN2.stylegan2_discriminator import StyleGan2Discriminator


def generate_and_save_images(images, it, plot_fig=True):
    plt.close()
    fig = plt.figure(figsize=(9, 9))

    for i in range(images.shape[0]):
        plt.subplot(2, 2, i + 1)
        plt.imshow(images[i])
        plt.axis('off')

    # tight_layout minimizes the overlap between 2 sub-plots
    fig.tight_layout()
    plt.savefig('images/generated/image_at_iter_{:04d}.png'.format(it))
    if plot_fig: plt.show()


def normalize_image(image, mean, std):
    for channel in range(3):
        image[:, :, channel] = (image[:, :, channel] - mean[channel]) / std[channel]
    return image


impl = 'ref'  # 'ref' if cuda is not available in your machine
gpu = False  # False if tensorflow cpu is used
weights_name = 'ffhq'  # face model trained by Nvidia
'''
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
'''
target_size = 1024
file_path = "images/real_images/anne.jpg"
img = tf.io.read_file(file_path)
img_original = tf.io.decode_jpeg(img, channels=3)
img = tf.image.resize(img_original, [target_size, target_size])
img = tf.transpose(img, [2, 0, 1])  # Convert from [height, width, channel] to [channel, height, width]
img = img * 2 / 255.0 - 1  # Scale each pixel from [0,255] to [-1,1]
img_array = np.array([img])  # Create a bench

# out_discriminator = discriminator(img_array)
# print("Discriminator output with REAL IMAGE", out_discriminator)

clip = CLIP.CLIP()
clip_target_size = 224

img = tf.image.resize(img_original, [clip_target_size, clip_target_size])
# Check pixels range, it should be [0, 255]
img_array = np.array([img])  # Create a bench
img_array = normalize_image(img_array,  # PyTorch does it
                            (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
# print(img_latent_space)  # Array of 512

parole = ["a diagram", "a dog", "a cat", "a neural network"]
tokenized = clip.tokenize(parole)
print("tokenized", tokenized)
text = np.expand_dims(tokenized, 0).astype(int)  # grml... keras doesnt like different cardinality in batch dim
logits_per_image, logits_per_text = clip.predict(img_array, text)
tf_probs = tf.nn.softmax(logits_per_image, axis=1)
tf_probs = np.array(tf_probs)
print(tf_probs)
