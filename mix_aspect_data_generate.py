import random
import re
import jsonlines
from tqdm import tqdm
from src.api import *
from src.templates import *

def format_generate_instructions(aspect_sentiment_set, template, example):
    instructions = []
    if example == 0:
        example = ""
        ex_input = ""
        for input in aspect_sentiment_set:
            instructions.append(template.format(input=input,example_input=ex_input,example_output=example))
    else:
        for input in aspect_sentiment_set:
            example_ = random.choice(example)
            ex_input = example_.split('#')[1].replace(']','')
            ex_output = example_.split('#')[0].replace('[','')
            instructions.append(template.format(input=input,example_input=ex_input,example_output=ex_output))
    return instructions

def process_data_in_batches(data, batch_size):
    num_samples = len(data)
    num_batches = num_samples // batch_size
    bc = num_batches
    dataset = []
    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = (batch_idx + 1) * batch_size
        batch_data = data[start_idx:end_idx]
        dataset.append(batch_data)
        print(f"Processing Batch {batch_idx}: {batch_data}")
    remainder_data = data[num_batches * batch_size:]
    if remainder_data:
        bc += 1
        print(f"Processing Remainder Batch: {remainder_data}")
        dataset.append(remainder_data)
    print(bc)
    epoch = 0
    for batch_idx in dataset:
        if epoch == 0:
            print('epoch:',epoch+1)
            instruntion = format_generate_instructions(batch_idx, template, epoch)
            result_pseudo_sample = pseudo_sample_generate(instruntion)
            example, high_score_pseudo_sample = select(result_pseudo_sample, epoch)
            save_mix_pseudo_sample(high_score_pseudo_sample)
        elif epoch < bc - 1 :
            print('epoch:',epoch+1)
            instruntion = format_generate_instructions(batch_idx, template, example)
            result_pseudo_sample = pseudo_sample_generate(instruntion)
            example, high_score_pseudo_sample = select(result_pseudo_sample, epoch)
            save_mix_pseudo_sample(high_score_pseudo_sample)
        else:
            print('epoch:',epoch+1)
            instruntion = format_generate_instructions(batch_idx, template, example)
            result_pseudo_sample = pseudo_sample_generate(instruntion)
            example, high_score_pseudo_sample = select(result_pseudo_sample, epoch)
            save_mix_pseudo_sample(high_score_pseudo_sample)
        epoch += 1

def select(pseudo_sample, epoch):
    qualified_samples = open(save_path_score + "/" + "sample_score_" + category + ".txt", "a", encoding="utf-8")
    discarded_samples = open(save_path_score + "/" + "low_score_remain_" + category + ".txt", "a", encoding="utf-8")
    eval_score_high = Eval_score
    example_str = ''
    select_p = []
    mutl_fl = 0
    for i,line in enumerate(pseudo_sample):
        if mutl_fl == 0:
            true_sentiment = line.split('#')[1].split(',')[1].replace(']', '').strip()
            true_aspect = line.split('#')[1].split(',')[0]
            sentence = line.split('#')[0].replace("$T$", true_aspect).replace('[', '')
            line_p = sentence + '#' + true_aspect + ',' + true_sentiment
            mutl_fl += 1
        elif mutl_fl < aspect_num -1 :
            true_sentiment = line.split('#')[1].split(',')[1].replace(']', '').strip()
            true_aspect = line.split('#')[1].split(',')[0]
            line_p = line_p + ';' + true_aspect + ',' + true_sentiment
            mutl_fl += 1
        else:
            mutl_fl = 0
            true_sentiment = line.split('#')[1].split(',')[1].replace(']', '').strip()
            true_aspect = line.split('#')[1].split(',')[0]
            line_p = line_p + ';' + true_aspect + ',' + true_sentiment
            select_p.append(line_p)
            example_str += f'sample {(i + 1)/aspect_num}: "{line_p}"\n'
    instruction = eval_score_high.format(example=example_str)
    sample_score_dict = {}
    while True:
        try:
            num = 0
            score_example = judge_sample(instruction)
            high_quality_sample = []
            high_score = 0.00
            for i, example_line in enumerate(score_example.split('\n'), start=1):
                assert len(score_example.split('\n')) == len(pseudo_sample) // aspect_num
                comprehensive_score_match = re.search(r'comprehensive score:(\d+(\.\d+)?)', example_line)
                comprehensive_score = float(comprehensive_score_match.group(1))
                if comprehensive_score >= 6.00:
                    sample_score_dict[i] = comprehensive_score
                    for i in range(aspect_num):
                        high_quality_sample.append(pseudo_sample[num+i])
                else:
                    aspect_sentiment = pseudo_sample[num].split('#')[1].replace(']','')
                    discarded_samples.write(f"{aspect_sentiment}\n")
                if comprehensive_score >= high_score:
                    high_score = comprehensive_score
                num += aspect_num
            break
        except Exception as e:
            print('error',e)
    sorted_samples = sorted(sample_score_dict.items(), key=lambda x: x[1], reverse=True)
    high_sample_idex = [sample[0] for sample in sorted_samples[:k]]
    high_sample = ["[" + select_p[i - 1] + "]" for i in high_sample_idex]
    qualified_res = f"sample: {example_str}\nscore: {score_example}\nepoch: {epoch + 1}\n{example_str}\n{score_example}"
    print(qualified_res)
    qualified_samples.write(qualified_res)
    return high_sample, high_quality_sample

def save_mix_pseudo_sample(result_pseudo_sample):
    pseudo_sample_total = []
    with jsonlines.open(save_path, mode='a') as writer:
        for line in result_pseudo_sample:
            pseudo_sample = {'sentence': [], 'aspect': [], 'sentiment': []}
            sentiment = line.split('#')[1].split(',')[1].replace(']','').strip()
            aspect = line.split('#')[1].split(',')[0]
            sentence = line.split('#')[0].replace('[','')
            pseudo_sample['sentence'] = sentence
            pseudo_sample['aspect'] = aspect
            pseudo_sample['sentiment'] = sentiment
            writer.write(pseudo_sample)
            pseudo_sample_total.append(pseudo_sample)
    return pseudo_sample_total

def eval_filter_mix_sample(pseudo_sample):
    sample = []
    domain_num = 0
    for line in pseudo_sample:
        eval_filter_template = Eval_filter
        true_sentiment = line.split('#')[1].split(',')[1].split(']')[0].strip()
        true_aspect = line.split('#')[1].split(',')[0]
        sentence = line.split('#')[0].replace("$T$", true_aspect)
        input = f'input :\nsentence: {sentence}#aspect: {true_aspect}\n'
        instruction = eval_filter_template.format(input=input)
        senti_epoch = 0
        while True:
            try:
                if senti_epoch < 3:
                    result = judge_sample(instruction)
                    sentiment = re.search(r'\b(positive|negative|neutral)\b', result.lower())
                    domain_accuracy = re.search(r'\b(Y|N)\b', result).group(1)
                    senti_epoch += 1
                    pred_sentiment = sentiment.group(1)
                    if domain_accuracy == 'Y':
                        domain_num += 1
                    break
                else:
                    break
            except Exception as e:
                print('error judge', e)
                print("Re-judge……")
        if pred_sentiment == true_sentiment:
            if domain_num >= 1:
                sample.append(line)
    if len(sample) == len(pseudo_sample):
        return sample

def generate_sample(query):
    '''
     pseudo samples generation
    '''
    res = invoke_gpt_turbo_generate(query)
    return res

def judge_sample(query):
    '''
    judgement moddule
    '''
    res = invoke_gpt_turbo(query)
    return res

def mix_generate_sample(query):
    import re
    result = generate_sample(query)
    pseudo_sample_total = []
    num = 0
    if len(result.split('#')[0].split(' ')) > 100:
        return False
    for sample_line in re.split("#|;", result)[1:]:
        aspect = sample_line.split(',')[0]
        try:
            match = re.search(r'\b{}\b'.format(re.escape(aspect.lower())), result.split('#')[0].lower())
            if match:
                sentence = result.split('#')[0].replace(f"{aspect}", '$T$')
                sample_line = sentence + '#' + sample_line
                pseudo_sample_total.append(sample_line)
                num += 1
        except:
            return False
    return pseudo_sample_total if num == aspect_num else False

def pseudo_sample_generate(instructions):
    pseudo_sample = []
    for query in tqdm(instructions):
        filter_num = 0
        while True:
            try:
                if filter_num < 5:
                    filter_num += 1
                    result = mix_generate_sample(query)
                    eval_signal = eval_filter_mix_sample(result)
                else:
                    break
                if eval_signal:
                    for line in eval_signal: pseudo_sample.append(line)
                    break
                else:
                    print("Filtered sample: ", result)
            except Exception as e:
                print("Error: ", e)
                print('Regenerate...')
    return pseudo_sample

def aspect_sentiment_comb_mix(aspect_set, sentiment_polarity):
    num = 0
    aspect_sentiment = ''
    aspect_sentiment_set = []
    aspect_sentiment_two = []
    selected_sentiment = sentiment_polarity.copy()
    with open(aspect_set, 'r', encoding='UTF-8') as asp:
        for aspect in asp:
            sentiment = random.choice(selected_sentiment)
            aspect_sentiment += aspect.split('\n')[0] + ',' + sentiment + ';'
            num += 1
            aspect_sentiment_two.append(aspect_sentiment)
            if num >= aspect_num:
                aspect_sentiment_set.append(aspect_sentiment_two)
                num = 0
                aspect_sentiment_two = []
                selected_sentiment = sentiment_polarity.copy()
            aspect_sentiment = ''
    return aspect_sentiment_set

if __name__ == "__main__":
    batch_size = 10
    k = 3 # num of additional examples
    aspect_num = 2
    category = 'number_sentiment' #two_mix,tow_pos
    aspect_set_path = r'.\aspect.txt'
    save_path = rf'.\score\{category}.jsonl'
    save_path_score = rf'.\score'
    domain = 'laptop'
    template = ITAT_template
    sentiment_polarity = ['positive', 'positive', 'neutral']
    aspect_sentiment_set_mix = aspect_sentiment_comb_mix(aspect_set_path, sentiment_polarity)
    process_data_in_batches(aspect_sentiment_set_mix, batch_size)
