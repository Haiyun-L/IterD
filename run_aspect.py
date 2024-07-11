import random
from typing import re

from src.data_utils import *
from src.templates import *
from src.api import *
from tqdm import tqdm
import pandas as pd
import random
from src.data_utils import *
import re
from src.api import *
from src.templates import *
from eval import *


def load_datasets(task, dataset, k=5):
    data_dir = f'./IterD/original_data/{dataset}/'
    types = ['unlabeled_corpus.txt', 'examples.txt']

    for type in types:
        path = data_dir + type

        # TODO 实现所有任务
        if task == 'EX':
            inputs, targets = prepare_EX_extraction(path)
            datasets = [[i, t] for i, t in zip(inputs, targets)]
        else:
            raise NotImplementedError

        if type == 'unlabeled_corpus.txt':
            train_dataset = datasets
        elif type == 'examples.txt':
            test_dataset = datasets

    # In-context Learning 指令构建
    inputs, outputs = zip(*train_dataset)
    train_instructions = format_instructions(test_dataset, inputs, task, k)
    train_dataloader = [{'instruction': train_instruction, 'sent': input, 'output': output} for
                        train_instruction, input, output in zip(train_instructions, inputs, outputs)]
    return train_dataloader

def format_instructions(choose_dataset, inputs, task, k=5):
    # 随机选择 k 个示例
    example_set = random.sample(choose_dataset, k)
    example_str = ""
    for i, example in enumerate(example_set):
        example_str += f'example {i + 1}:\nInput: "{example[0]}"\nOutput: "{example[1]}"\n'
    if task == 'EX':
        template = EX_template
    else:
        raise NotImplementedError
    instructions = []
    for input in inputs:
        instructions.append(template.format(example=example_str, input=input))

    return instructions

def format_ET_instructions(aspect_set, template):
    instructions = []

    for input in aspect_set:
        instructions.append(template.format(input=input,example_in=ET_example_in,example_out=ET_example_out))
    return instructions

def parse_output(s):
    groups = re.findall(r'\[(.*?)\]', s)
    elements = [re.split(r';\s*', group) for group in groups]
    return elements


def check_format(s, task, is_NULL):
    '''
    给定一个 GPT 生成的结果，判断是否符合任务的格式要求
    '''
    elements = parse_output(s)

    # 如果某个元素长度不为 3
    if task == 'EX':
        if is_NULL is False and len(elements) == 0:
            return False
        else:
            return True
    elif task == 'ET':
        if is_NULL is False and len(elements) == 0:
            return False
        else:
            return True
    else:
        raise NotImplementedError

def generate_format(query, history, task, is_NULL):
    res = invoke_gpt_turbo(query, history=history)
    if check_format(res, task, is_NULL):
        return res
    else:
        return generate_format(query, history, task, is_NULL)

def generate(query):
    '''
    生成 AE 任务的预测结果
    '''
    res = invoke_gpt_turbo(query)
    return res

def source_aspect(dataloader):
    for data in tqdm(dataloader):
        aspect_set = data['output']
        aspect_set = parse_output(aspect_set)
        for aspects in aspect_set:
            GT_file.write("".join(aspects[0]))
            GT_file.write("\n")
        GT_file.close()

def aspect_ex(dataloader):
    preds = []
    for data in tqdm(dataloader):
        instruction, true = data['instruction'], data['output']
        is_NULL = False if len(data['output']) > 0 else True
        pred = generate(instruction)
        if pred == None:
            print("Error: get_response failed!")
            continue
        try:
            pred_list = parse_output(pred)
        except Exception as e:
            pred_list = []
        preds.append(pred_list)

    for aspects in preds:
        for aspect in aspects:
            EX_file.write("".join(aspect[0]))
            EX_file.write("\n")
    EX_file.close()

def aspect_et(aspect_set_path):
    aspect_set = []
    with open(aspect_set_path, 'r', encoding='UTF-8') as aspects:
        for aspect in aspects:
            aspect_set.append(aspect)
    instruction = format_ET_instructions(aspect_set,template=ET_template)
    Extend_aspects = generate(instruction)
    for line in Extend_aspects:
        for Extend_aspect in line:
            ET_file.write("".join(Extend_aspect))
            ET_file.write("\n")
    ET_file.close()


if __name__ == "__main__":
    task = 'EX'
    dataset = 'laptop14' # rest14 rest15 rest16 laptop14
    if task == 'EX':
        save_path = r'F:\实验室电脑-C盘\Iter_DG\IterD\aspect_set'
        EX_file = open(save_path + "/" + dataset + ".txt", "a", encoding="utf-8")
        dataloader = load_datasets(task, dataset)
        aspect_ex(dataloader, task, dataset)
    elif task == 'ET':
        save_path = r'IterD/aspect_set'
        ET_file = open(save_path + "/" + dataset + "_" + ".txt", "a", encoding="utf-8")
        ET_example_in = 'salad'
        ET_example_out = '#fish#noodles#bread#fruit salads' #扩展方面词示例
        aspect_set_path = r'.\aspect.txt' #名词方面词集
        aspect_et(aspect_set_path)
    elif task == 'GT':
        save_path = r'C:\Users\16488\Desktop\LLM_DA\ODA_LLM\source_new_aspect\ori'
        GT_file = open(save_path + "/" + dataset + "_" + "source_aspect" + ".txt", "a", encoding="utf-8") #提取人工标注数据集的标注
        dataloader = load_datasets(task, dataset)
        source_aspect(dataloader)

