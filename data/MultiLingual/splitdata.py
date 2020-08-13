import argparse
import json
import os
from random import random
from langdetect import detect

parser = argparse.ArgumentParser(
        description='Bert Classifier Project for Covid-19')
    # data arguments
parser.add_argument('--input', default='crowdsourcing20200420.processed.jsonl',
                        help='path to annotated data')
parser.add_argument('--output', default='0421',
                        help='path to annotated data')
args = parser.parse_args()

divided_dataset = [[] for _ in range(3)]

with open(args.input,"r") as f:
    dataset = f.readlines()
dataset = [json.loads(x) for x in dataset]
dataset = filter(lambda x: "cleaned_text" in x.keys() and x["title"]!=r'', dataset)
for sample in dataset:
    try:
        lang = detect(sample["cleaned_text"])
        sample['lang'] = lang
    except Exception:
        print(sample["title"])
        # print(
        continue
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