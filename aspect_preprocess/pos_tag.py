import os
import nltk
from textblob import TextBlob
from nltk import pos_tag
aspect_file = r"/mams/mams_clean.txt"
aspect_de_repeat = []
save_path = r'/mams'
filename = os.path.basename(aspect_file)
name, extension = os.path.splitext(filename)
nltk.download('punkt')
pos_save = {'NNP', 'NN'}
pos_del = {'JJ','VBD','CC','VB','MD','VBZ','IN'}
positive_aspect = []
negative_aspect = []
neutral_aspect = []

with open(aspect_file, "r",encoding='utf-8') as file:
    for line in file:
        if line in aspect_de_repeat:
            continue
        if len(line.split(' ')) > 5:
            continue
        aspect_de_repeat.append(line)
    aspect_pos = []
    tagged_tokens = pos_tag(aspect_de_repeat)
    for token in tagged_tokens:
        if token[1] in pos_del:
            continue
        polarity = TextBlob(token[0]).sentiment.polarity
        if polarity < 0:
            negative_aspect.append(token[0])
        elif polarity > 0:
            positive_aspect.append(token[0])
        else:
            neutral_aspect.append(token[0])

    aspect_pos.append(token[0])
for line in negative_aspect:
    txt_file_neg = open(save_path + "/" + name + "_process_neg" + ".txt", "a", encoding="utf-8")
    txt_file_neg.write(line)
    txt_file_neg.close()

for line in positive_aspect:
    txt_file_pos = open(save_path + "/" + name + "_process_pos" + ".txt", "a", encoding="utf-8")
    txt_file_pos.write(line)
    txt_file_pos.close()

for line in neutral_aspect:
    txt_file_neu = open(save_path + "/" + name + "_process_neu" + ".txt", "a", encoding="utf-8")
    txt_file_neu.write(line)
    txt_file_neu.close()
    txt_file_neg = open(save_path + "/" + name + "_process_neg" + ".txt", "a", encoding="utf-8")
    txt_file_neg.write(line)
    txt_file_neg.close()
    txt_file_pos = open(save_path + "/" + name + "_process_pos" + ".txt", "a", encoding="utf-8")
    txt_file_pos.write(line)
    txt_file_pos.close()
