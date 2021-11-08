import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

file = "../celeb_a/list_attr_celeba.csv"
csv = pd.read_csv(file)
distribution = {}

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

for element in features:
    distribution[element] = 0

for element in opposite_features:
    if len(element) > 0:
        distribution[element] = 0


def generate_colors(num):
    step = 1.0/num
    colors = []
    for i in range(num):
        colors.append(matplotlib.colors.hsv_to_rgb([step*i,1,1]))

    return colors

size = len(csv)
for row in range(size):
    img_name = csv.at[row, 'image_id']
    for colIndex in range(len(csv.columns[1:])):
        if csv.at[row, features[colIndex]] == 1:
            distribution[features[colIndex]] += 1
        else:
            if len(opposite_features[colIndex]) > 0:
                distribution[opposite_features[colIndex]] += 1


colors = generate_colors(len(distribution))
str_values = list(map(lambda x:str(x),distribution.values()))

plt.figure(figsize=[15,8])
plt.xticks(rotation=90)
plt.grid(True,axis='y',color="lightgrey")
bars = plt.bar(distribution.keys(),distribution.values(),color=colors)

for bar in bars:    #Adds text on top of the bars
    yval = bar.get_height()
    plt.text(bar.get_x()+0.1, yval + size*0.01, yval,rotation=90)

plt.subplots_adjust(bottom=0.25) # or whatever
plt.show()
