import re
import jsonlines
from tqdm import tqdm
from src.api import *
from src.templates import *

def format_generate_instructions(aspect_sentiment_set, template, example):
    instructions = []
    if example == 0:
        example = ""
        for input in aspect_sentiment_set:
            instructions.append(template.format(input=input, example_output=example))
    else:
        for input in aspect_sentiment_set:
            instructions.append(template.format(input=input,example_output = example[0]))
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
    example = ""
    for batch_idx in dataset:
        if epoch == 0:
            print('epoch:', epoch + 1)
            instruntion = format_generate_instructions(batch_idx, template, epoch)
            result_pseudo_sample = pseudo_sample_generate(instruntion)
            example, high_score_pseudo_sample = select(result_pseudo_sample, epoch)
            save_single_pseudo_sample(high_score_pseudo_sample)
            return example
        elif epoch <= bc - 1:
            print('epoch:', epoch + 1)
            print()
            instruntion = format_generate_instructions(batch_idx, template, example)
            result_pseudo_sample = pseudo_sample_generate(instruntion)
            example, high_score_pseudo_sample = select(result_pseudo_sample, epoch)
            save_single_pseudo_sample(high_score_pseudo_sample)
            return example
        else:
            print('epoch:',epoch+1)
            instruntion = format_generate_instructions(batch_idx, template, example)
            result_pseudo_sample = pseudo_sample_generate(instruntion)
            example, high_score_pseudo_sample = select(result_pseudo_sample, epoch)
            save_single_pseudo_sample(high_score_pseudo_sample)
        epoch += 1

def select(pseudo_sample, epoch):
    qualified_samples = open(save_path_score + "/" + "sample_score_" + category + ".txt", "a", encoding="utf-8")
    discarded_samples = open(save_path_score + "/" + "low_score_remain_" + category + ".txt", "a", encoding="utf-8")
    eval_score_high = Eval_score
    example_str = ''
    for i,line in enumerate(pseudo_sample):
        true_sentiment = line[0].split('#')[1].split(',')[1].replace(']','').strip()
        true_aspect = line[0].split('#')[1].split(',')[0]
        sentence = line[0].split('#')[0].replace("$T$", true_aspect).replace('[','')
        line = sentence + '#' + true_aspect + ',' + true_sentiment
        example_str += f'sample {i + 1}: "{line}"\n'
    instruction = eval_score_high.format(example=example_str)
    num = 0
    sample_score_dict = {}
    score_num = 0
    while True:
        try:
            if score_num < 5:
                score_example = judge_sample(instruction)
                score_num += 1
                high_quality_sample = []
                for i,example_line in enumerate(score_example.split('\n'),start=1):
                    assert len(score_example.split('\n')) == len(pseudo_sample)
                    comprehensive_score_match = re.search(r'comprehensive score:(\d+(\.\d+)?)', example_line) #(\d+\.\d+)
                    comprehensive_score = float(comprehensive_score_match.group(1))
                    sample_score_dict[i] = comprehensive_score
                    if comprehensive_score >= 6.00:
                        high_quality_sample.append(pseudo_sample[num])
                    else:
                        aspect_sentiment = pseudo_sample[num][0].split('#')[1].replace(']','')
                        discarded_samples.write(f"{aspect_sentiment}\n")
                    num += 1
                break
            else:
                high_quality_sample = pseudo_sample
                high_sample = pseudo_sample[0]
                return high_sample, high_quality_sample
        except Exception as e:
            print('error',e)
    sorted_samples = sorted(sample_score_dict.items(), key=lambda x: x[1], reverse=True)
    high_sample_idex = [sample[0] for sample in sorted_samples[:1]]
    high_sample = ["["+pseudo_sample[i-1][0]+"]" for i in high_sample_idex]
    qualified_res = f"sample: {example_str}\nscore: {score_example}\nepoch: {epoch + 1}\n{example_str}\n{score_example}"
    print(qualified_res)
    qualified_samples.write(qualified_res)
    return high_sample, high_quality_sample

def save_single_pseudo_sample(result_pseudo_sample):
    pseudo_sample_total = []
    with jsonlines.open(save_path, mode='a') as writer:
        for line in result_pseudo_sample:
            pseudo_sample = {'sentence': [], 'aspect': [], 'sentiment': []}
            sentiment = line[0].split('#')[1].split(',')[1].replace(']','').strip()
            aspect = line[0].split('#')[1].split(',')[0]
            sentence = line[0].split('#')[0].replace('[','')
            pseudo_sample['sentence'] = sentence
            pseudo_sample['aspect'] = aspect
            pseudo_sample['sentiment'] = sentiment
            writer.write(pseudo_sample)
            pseudo_sample_total.append(pseudo_sample)
    return pseudo_sample_total

def eval_filter_single_sample(pseudo_sample):
    eval_filter_template = Eval_filter
    true_sentiment = pseudo_sample[0].split('#')[1].split(',')[1].split(']')[0].strip()
    true_aspect = pseudo_sample[0].split('#')[1].split(',')[0]
    sentence = pseudo_sample[0].split('#')[0].replace("$T$", true_aspect)
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
                break
            else:
                break
        except Exception as e:
            print('Error judge', e)
            print("Re-judge……")
    if pred_sentiment == true_sentiment:
        if domain_accuracy == 'Y':
            return True

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

def single_generate_sample(query):
    result = generate_sample(query)
    aspect = result.split('#')[1].split(',')[0]
    if len(result.split('#')[0].split(' ')) > 30:
        return False
    if "$T$" in result.split('#')[0]:
        return result.split('\n')
    else:
        try:
            match = re.search(r'\b{}\b'.format(re.escape(aspect.lower())), result.split('#')[0].lower())
            if match:
                sentence = result.split('#')[0].replace(f"{aspect}", '$T$')
                result = sentence + '#' + result.split('#')[1]
                return result.split('\n')
        except:
            return False

def pseudo_sample_generate(instructions):
    pseudo_sample = []
    for query in tqdm(instructions):
        filter_num = 0
        while True:
            try:
                if filter_num < 5:
                    result = single_generate_sample(query)
                    filter_num += 1
                    eval_signal = eval_filter_single_sample(result)
                else:
                    break
                if eval_signal:
                    pseudo_sample.append(result)
                    break
                else:
                    print("Filtered sample: ", result)
            except Exception as e:
                print("Error: ", e)
                print('Regenerate...')
    return pseudo_sample

def aspect_sentiment_comb_single(aspect_set, sentiment_polarity):
    num = 0
    aspect_sentiment = ''
    aspect_sentiment_set = []
    with open(aspect_set, 'r', encoding='UTF-8') as asp:
        for aspect in asp:
            aspect_sentiment += aspect.split('\n')[0] + ',' + sentiment_polarity + ';'
            num += 1
            if num >= 1:
                aspect_sentiment_set.append(aspect_sentiment)
                aspect_sentiment = ''
                num = 0
    return aspect_sentiment_set

if __name__ == "__main__":
    batch_size = 10
    category = 'number_setiment' #one_pos
    aspect_set_path = r'.\aspect.txt'
    save_path_score = rf'.\score'
    save_path = rf'.\{category}.jsonl'
    domain = 'laptop'
    template = ITAT_template
    sentiment_polarity = ['neutral', 'neutral', 'neutral']
    aspect_sentiment_set_single = aspect_sentiment_comb_single(aspect_set_path, sentiment_polarity[0])
    process_data_in_batches(aspect_sentiment_set_single, batch_size)


