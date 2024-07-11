import json
import re

sentiment_polarity = {'negative':-1, 'neutral':0, 'positive':1}

jsonl_file = r"C:\Users\16488\Desktop\Data_process\single_pseudo_label\rest16\sim\rest16_one_pos_sim_add.jsonl"
with open(jsonl_file, "r", encoding="utf-8") as file:
    for line in file:
        flag = 0
        try:
            aspect_num = 0
            json_obj = json.loads(line)
            signal = sentiment_polarity[json_obj['sentiment'].replace(']', '').strip()]
            # if signal != 1:
            #     continue
            # 对 JSON 对象进行处理
            pattern = r'[\n,.?;!\'\[\]]'
            # 使用空格替换所有匹配的特殊字符
            sentence = re.sub(pattern, lambda x: f' {x.group()} ', json_obj['sentence'])
            aspect = json_obj['aspect'].replace('\n','')
            aspect_signal = re.search(rf'\b{aspect.lower()}\b', sentence)
            # if aspect_signal:
            if "$T$" in sentence:
                sentence = sentence
            else:
                sentence = sentence.lower().replace(f'{aspect.lower()}', "$T$", 1)
            if sentence[0] != '$':
                sentence = sentence.replace('$T$', " $T$ ", 1)
            else:
                sentence = sentence.replace('$T$', "$T$ ", 1)
            sentence = sentence.replace('  ', " ")
            sentence = sentence[:1].capitalize() + sentence[1:]
            save_path = r'C:\Users\16488\Desktop\Data_process\data_process'
            txt_file = open(save_path + "/" + "new" + ".raw", "a", encoding="utf-8")
            if aspect != "$T$" and "$T$" in sentence:
                txt_file.write(sentence.split('\n')[0])
                txt_file.write("\n")
                txt_file.write(json_obj['aspect'])
                txt_file.write("\n")
                txt_file.write(str(sentiment_polarity[json_obj['sentiment'].replace(']', '').strip()]))
                txt_file.write("\n")

        except:
            continue