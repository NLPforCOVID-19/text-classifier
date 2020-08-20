import argparse
import json
import os
import re
from random import random
from langdetect import detect
from transformers import BertTokenizer

parser = argparse.ArgumentParser(
        description='Bert Classifier Project for Covid-19')
    # data arguments
parser.add_argument('--input', default='crowdsourcing20200420.processed.jsonl',
                        help='path to annotated data')
parser.add_argument('--input_retreated', default='crowdsourcing20200420.processed.jsonl',
                        help='path to annotated data')
parser.add_argument('--output', default='0421',
                        help='path to annotated data')
args = parser.parse_args()

divided_dataset = [[] for _ in range(3)]

with open(args.input,"r") as f:
    dataset = f.readlines()
with open(args.input_retreated,"r") as f:
    dataset_retreated = json.loads(f.read())
dataset = [json.loads(x) for x in dataset]
dataset = list(filter(lambda x: "cleaned_text" in x.keys() and len(x["title"])>0, dataset))
tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")
for x in dataset:
    sents = re.split('; |\*|\n|\. |。',x["cleaned_text"])
    sents = [tokenizer.tokenize(x) for x in sents]
    x["cleaned_text"] = list(filter(lambda x: len(x)> 3, sents))
    # print(x)
# dataset = list(dataset) + dataset_retreated
# dataset = list(filter(lambda x: "cleaned_text" in x.keys() and x["title"]!=r'', dataset))
print(len(dataset), len(dataset_retreated))
for sample in dataset:
    try:
        lang = detect(sample["title"])
        if lang in ['de', 'ja', 'fa', 'it', 'pt', 'es', 'fr', 'en', 'vi', 'zh-cn', 'ru', 'ar', 'ko']:
            sample['lang'] = lang
        else:
            print(lang)
            print(sample["title"])
            del sample
    except Exception as e:
        print(e)
        del sample
        continue

for sample in dataset_retreated:
    sample['lang'] = 'ja'
print(len(dataset), len(dataset_retreated))
dataset = dataset_retreated+dataset
for sample in dataset:
    dice = random()
    if dice < 0.8:
        divided_dataset[0].append(json.dumps(sample))
    elif dice < 0.9:
        divided_dataset[1].append(json.dumps(sample))
    else:
        divided_dataset[2].append(json.dumps(sample))
# for lang in divided_dataset.keys():
#     if len(divided_dataset[lang][0]) < 600:
#         continue
#     if not os.path.exists(os.path.join(args.output, lang)):
#         os.makedirs(os.path.join(args.output, lang))
#     with open(os.path.join(os.path.join(args.output, lang), "crowdsourcing.train"),"w") as f:
#         f.write("\n".join(divided_dataset[lang][0]))
#     with open(os.path.join(os.path.join(args.output, lang), "crowdsourcing.dev"),"w") as f:
#         f.write("\n".join(divided_dataset[lang][1]))
#     with open(os.path.join(os.path.join(args.output, lang), "crowdsourcing.test"),"w") as f:
#         f.write("\n".join(divided_dataset[lang][2]))
with open(os.path.join(args.output, "crowdsourcing.train"),"w") as f:
    f.write("\n".join(divided_dataset[0]))
with open(os.path.join(args.output, "crowdsourcing.dev"),"w") as f:
    f.write("\n".join(divided_dataset[1]))
with open(os.path.join(args.output, "crowdsourcing.test"),"w") as f:
    f.write("\n".join(divided_dataset[2]))