import os
import random
import math
import nltk
from nltk import pos_tag
aspect_file = r"C:\Users\16488\Desktop\Iter_DG\new_aspect_set\few_shot_com\clean\laptop14_few_clean.txt"
aspect_de_repeat = []
save_path = r'/Extend_aspect'
filename = os.path.basename(aspect_file)
name, extension = os.path.splitext(filename)
nltk.download('punkt')
pos_save = { 'NN'}
pos_del = {'JJ','VBD','CC','VB','MD','VBZ','IN'}
positive_aspect = []
negative_aspect = []
neutral_aspect = []

with open(aspect_file, "r",encoding='utf-8') as file:
    for line in file:
        aspect_de_repeat.append(line)

    aspect_pos = []
    tagged_tokens = pos_tag(aspect_de_repeat)
    for token in tagged_tokens:
        if token[1] in pos_save and len(token[0].split(' ')) < 2:
            aspect_pos.append(token[0])
for line in aspect_pos:
    txt_file_neg = open(save_path + "/" + name + "_nn_at" + ".txt", "a", encoding="utf-8")
    txt_file_neg.write(line)
    txt_file_neg.close()
