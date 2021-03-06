import pandas as pd
import tensorflow as tf

'''

Download celeb_a from here
https://www.kaggle.com/jessicali9530/celeba-dataset

'''

base_path = "celeb_a"  # Path of celeb_a
path = base_path + "/list_attr_celeba.csv"  # Path to csv file
image_path = base_path + "/img_align_celeba/img_align_celeba"  # Path to the folder that contains the images
target_size = [218, 218]

# Attributes of celeb_a images
features = ["5_o_Clock_Shadow", "Arched_Eyebrows", "Attractive", "Bags_Under_Eyes", "Bald", "Bangs",
            "Big_Lips", "Big_Nose", "Black_Hair", "Blond_Hair", "Blurry", "Brown_Hair", "Bushy_Eyebrows",
            "Chubby",
            "Double_Chin", "Eyeglasses", "Goatee", "Gray_Hair", "Heavy_Makeup", "High_Cheekbones", "Male",
            "Mouth_Slightly_Open", "Mustache", "Narrow_Eyes", "No_Beard", "Oval_Face", "Pale_Skin",
            "Pointy_Nose",
            "Receding_Hairline", "Rosy_Cheeks", "Sideburns", "Smiling", "Straight_Hair", "Wavy_Hair",
            "Wearing_Earrings", "Wearing_Hat", "Wearing_Lipstick", "Wearing_Necklace", "Wearing_Necktie",
            "Young"]

opposite_features = ["", "", "Ugly", "", "", "",
                     "Small_Lips", "Small_Nose", "", "", "", "", "",
                     "Thin",
                     "", "", "", "", "", "", "Woman",
                     "", "", "wide_Eyes", "", "", "",
                     "",
                     "", "", "", "", "", "",
                     "", "", "", "", "",
                     "Adult"]


class DatasetLoader:

    def __init__(self):
        self.length = 0
        dict = []
        csv = pd.read_csv(path)

        #Parse and concatenate attributes of each image of the dataset
        for row in range(len(csv)):
            img_name = csv.at[row, 'image_id']
            img_features = ""
            for colIndex in range(len(csv.columns[1:])):
                if csv.at[row, features[colIndex]] == 1:
                    img_features += features[colIndex].replace("_", " ") + " "
                else:
                    to_add = opposite_features[colIndex].replace("_", " ")
                    img_features += to_add
                    if len(to_add) > 0:
                        img_features += " "

            img_features = img_features.lower()
            tf_feature = {"caption": [img_features.strip()], "image": img_name}
            dict.append(tf_feature)
            self.length += 1

        self.ds = tf.data.Dataset.from_generator(lambda: dict, {"caption": tf.string, "image": tf.string})

    def getDataset(self, batch_size):
        def load_images(e):
            file_name = e['image']
            complete_path = image_path + "/" + file_name
            image_string = tf.io.read_file(complete_path)
            image_decoded = tf.image.decode_jpeg(image_string, channels=3)
            image_resized = tf.image.resize(image_decoded, target_size)
            e['image'] = image_resized
            e['caption'] = tf.convert_to_tensor(e['caption'])
            return e

        return self.ds.map(load_images, num_parallel_calls=tf.data.AUTOTUNE) \
            .shuffle(batch_size * 10) \
            .prefetch(buffer_size=tf.data.AUTOTUNE) \
            .batch(batch_size)

    def count(self):
        return self.length
