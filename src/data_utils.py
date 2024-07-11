import logging
import random

TAG_TO_WORD = {'POS': 'positive', 'NEG': 'negative', 'NEU': 'neutral'}
NONE_TOKEN = "[none]"
ASPECT_TOKEN = "<aspect>"
OPINION_TOKEN = "<opinion>"
TAG_TO_SPECIAL = {"POS": ("<pos>", "</pos>"), "NEG": ("<neg>", "</neg>"), "NEU": ("<neu>", "</neu>")}
senttag2word = {'POS': 'positive', 'NEG': 'negative', 'NEU': 'neutral'}
logger = logging.getLogger(__name__)

def read_line_examples_from_file(data_path):
    """
    Read data from file, each line is: sent####labels
    Return List[List[word]], List[Tuple]
    """
    sents, labels = [], []
    with open(data_path, 'r', encoding='UTF-8') as fp:
        words, labels = [], []
        for line in fp:
            line = line.strip()
            if line != '':
                words, tuples = line.split('####')
                sents.append(words.split())
                labels.append(eval(tuples))
    logger.info(f"{data_path.split('/')[-1]}\tTotal examples = {len(sents)} ")
    return sents, labels


def get_inputs(args, data_type_file="train"):
    """
        train_inputs: ["hi", "I love apples."],
    """
    data_path = f"{args.data_dir}/{data_type_file}.txt"
    inputs, _ = read_line_examples_from_file(data_path)
    inputs = [" ".join(i) for i in inputs]
    return inputs

def prepare_EX_extraction(data_path):
    sents, labels = read_line_examples_from_file(data_path)
    inputs = [" ".join(s) for s in sents]

    targets = []
    for i, label in enumerate(labels):
        if label == []:
            targets.append('None')
        else:
            all_tri = []
            for tri in label:
                # single aspect
                if len(tri[0]) == 1:
                    try:
                        a = sents[i][tri[0][0]]
                    except:
                        continue
                else:
                    start_idx, end_idx = tri[0][0], tri[0][-1]
                    a = ' '.join(sents[i][start_idx:end_idx+1])
                try:
                    c = TAG_TO_WORD[tri[1]]
                except:
                    c = TAG_TO_WORD[tri[2]]
                all_tri.append((a, c))
            label_strs = ['['+', '.join(l)+']' for l in all_tri]
            targets.append('; '.join(label_strs))
    return inputs, targets
