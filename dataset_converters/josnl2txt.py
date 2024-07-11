import json
import re

jsonl_file = r"/domain_accuracy/JUDGE_LAPTOP_negative.jsonl"

with open(jsonl_file, "rb") as file:
    with open(jsonl_file, "r") as file:
        for line in file:
            ##get data
            json_obj = json.loads(line)
            sentence = json_obj["sentence"]
            aspect = json_obj["aspect"]
            sentiment = json_obj["sentiment"]
            ##get sentiment
            insert_obj=''
            if(sentiment=="positive"):
                insert_obj = '/p '
            elif(sentiment=="neutral"):
                insert_obj= '/0 '
            elif (sentiment == "negative"):
                insert_obj = '/n '

            # ##format
            # sample=sample.replace(',',' ,')
            # sample=sample.replace('.',' .')
            # sample=sample.replace(':', ' : ')
            # sample=sample.replace("'", " ' ")
            # sample=sample.replace(';', ' ; ')
            # sample=sample.replace('!', ' ! ')
            # sample=sample.replace('?', ' ? ')
            # sample=sample.replace('(', ' ( ')
            # sample=sample.replace(')', ' ) ')

            ##transform
            # if "$T$" in sentence:
            #     sentence = sentence.replace('$T$',insert_obj)
            # else:
            #     for token in aspect.split(' '):
            #         sentence = sentence.lower().replace(f'{token.lower()}', token+insert_obj)
            # formatted_sentence = ' '.join([f"{word}/{sentiment}" if word.lower() == aspect.lower() else word for word in sentence.split()])



            # 处理数据
            aspect_sentiment = ""
            for aspect_line in aspect.split():
                aspect_sentiment += aspect_line + insert_obj
            sentence = sentence.replace(f"{aspect}",aspect_sentiment,1)

            # index_top=sentence.find(aspect)
            # length=len(aspect)
            # index_tail=index_top+length
            # insert_temp=list(sentence)
            # insert_temp.insert(index_tail,insert_obj)
            # sentence = ''.join(insert_temp)
            if '/n' in sentence or '/0' in sentence or '/p' in sentence:
                with open('../KGAN-DATA/laptop_few_shot_new.txt', 'a', encoding='utf-8') as f:
                    f.write(sentence+'\n')
                    f.close()


