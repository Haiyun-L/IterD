import json
import os
import re



sentiment_list = {}
sentiment_list[-1] = '/n '
sentiment_list[1] = '/p '
sentiment_list[0] = '/0 '
path = r'C:\Users\16488\Desktop\Data_process\KGAN-DATA\twitter\sou_total.raw'

filename = os.path.basename(path)
name, extension = os.path.splitext(filename)
save_path = rf'C:\Users\16488\Desktop\Data_process\KGAN-DATA\twitter\{name}.txt'
datasets = []
datasets_0 = []
target_sample = []
flag = 0
with open(path, encoding='utf-8') as fp:
    for line in fp:
        if flag == 0:
            target_sample.append(line)
        if flag == 1:
            target_sample.append(line)
        if flag == 2:
            target_sample.append(line)
        flag += 1
        if flag == 3:
            flag = 0
            datasets.append(target_sample)
            target_sample = []
for sample in datasets:
    sentence = sample[0].replace('\n','')
    aspect = sample[1].replace('\n','')
    try:
        sentiment = sentiment_list[int(sample[2].replace('\n',''))]
    except:
        continue
    aspect_sentiment = ""
    for aspect_line in aspect.split():
        aspect_sentiment += aspect_line + sentiment
    if "$T$" in sentence:
        sentence = sentence.replace('$T$',aspect_sentiment)
    else:
        sentence = sentence.replace(f"{aspect}",aspect_sentiment,1)
    if '/n' in sentence or '/0' in sentence or '/p' in sentence:
        if sentence[0] == ' ':
            sentence = sentence[1:]
        with open(save_path, 'a', encoding='utf-8') as f:
            f.write(sentence + '\n')
            f.close()

