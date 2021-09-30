import pandas as pd

path = "/home/luca/tensorflow_datasets/download/manual/celeb_a/list_attr_celeba.csv"
csv = pd.read_csv(path)

features = ["5_o_Clock_Shadow", "Arched_Eyebrows", "Attractive", "Bags_Under_Eyes", "Bald", "Bangs",
            "Big_Lips", "Big_Nose", "Black_Hair", "Blond_Hair", "Blurry", "Brown_Hair", "Bushy_Eyebrows",
            "Chubby",
            "Double_Chin", "Eyeglasses", "Goatee", "Gray_Hair", "Heavy_Makeup", "High_Cheekbones", "Male",
            "Mouth_Slightly_Open", "Mustache", "Narrow_Eyes", "No_Beard", "Oval_Face", "Pale_Skin",
            "Pointy_Nose",
            "Receding_Hairline", "Rosy_Cheeks", "Sideburns", "Smiling", "Straight_Hair", "Wavy_Hair",
            "Wearing_Earrings", "Wearing_Hat", "Wearing_Lipstick", "Wearing_Necklace", "Wearing_Necktie",
            "Young"]

dict = {}
for row in range(len(csv)):
    img_name = csv.at[row, 'image_id']
    dict[img_name] = []
    for colIndex in range(len(csv.columns[1:])):
        if csv.at[row, features[colIndex]]==1:
            dict[img_name].append(features[colIndex])

print(dict['000001.jpg'])