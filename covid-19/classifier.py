"""Classify new texts into pre-defined classes"""

import argparse
from xml.etree import ElementTree
import json
from collections import defaultdict
from pyknp import Juman
import os

target_pos = ["名詞" ,"動詞", "未定義語"]

keywords = {
    "感染状況": "感染状況 感染者 重症 死者",
    "予防": "予防 手洗い うがい 防ぐ",
    "検査": "検査 PCR",
    "渡航制限": "渡航制限 渡航禁止 検疫 防疫",
    "休校": "休校",
    "イベント中止": "イベント 自粛",
    "経済への影響": "株価 為替",
    "モノの不足": "買い占め 不足",
}

def to_text(juman, standard_format, target_pos=[]):
    "Convert standard format into a simple text (list of word lists)"
    target_pos = set(target_pos)
    annotations = etree.findall(".//Annotation")
    sentences = []
    for annotation in annotations:
        mlist = juman.result(annotation.text)
#        print(mlist)
#        input()
        sentences.append([m.midasi for m in mlist.mrph_list()])
    return sentences

def classify(text, keyword_dict):
    "Classify a text into pre-defined classes"
    text = [sentence for sentence in text if len(sentence) > 3]  # remove sentences with <=3 content words
    text = set(sum(text, []))
    classes = []
    for clss, keywords in keyword_dict.items():
        if len(keywords & text) >= 1:
            classes.append((clss, 1))
        else:
            classes.append((clss, 0))
    return classes

def inquire_exclude(sent):

    keywords = ['すべての結果を表示',
               'その他']
    for keyword in keywords:
        if keyword in sent:
            return True
    startingKeySymbols = ["＃", "＊", "©", "￥＋", "［","＊"]
    for keyword in startingKeySymbols:
        if keyword in sent:
            return True
    return False
def doc_exclude(doc):

    keywords = ['コロナ',
               'COVID', 'covid', "ｃｏｖｉｄ", "ＣＯＶＩＤ", "Covid", "Ｃｏｖｉｄ"]
    for keyword in keywords:
        if keyword in doc:
            return False
    return True

def content_filter(text):
    # print(text)
    tmp = list(filter(lambda x: not inquire_exclude(x), text))
    # text = list(filter(lambda x: not "その他" in x, text))
    # print(text)
    # input()
    return tmp

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("-d", default=".", help="Path prefix of XML files")
    argparser.add_argument("metadata_file", help="Metadata file (JSON)")
    argparser.add_argument("output_file", help="Output file (JSON)")
    args = argparser.parse_args()

    # initialize keyword dict and juman
    keywords = {key: set(value.split()) for key, value in keywords.items()}
    juman = Juman()

    # read metadata file
    with open(args.metadata_file, "r") as metadata_file:
        metadata = json.load(metadata_file)

    # classify each file
    for lang, pages in metadata.items():
        for page in pages:
            xml_file = page["meta"]["xml_file"]
            # print(f"{args.d}/{xml_file}")
            if os.stat(f"{args.d}/{xml_file}").st_size == 0:
                page["text"] = "EMPTY"
            else:
                etree = ElementTree.parse(f"{args.d}/{xml_file}")
                text = to_text(juman, etree, target_pos)
                text = filter(lambda x:len(x)>3, text)
                text = ["".join(i) for i in text]

                text = content_filter(text)
                page["text"] = " <SEP> ".join(text) if not doc_exclude(". ".join(text)) else "EMPTY"
                # print(page["text"])
                # input()




    # output the results into JSON file
    # with open(args.output_file, "w") as of:
    #     json.dump(metadata, of, ensure_ascii=False)

    # print(metadata)
    with open(args.output_file,"w") as f:
        for i in metadata.keys():
            for j in metadata[i]:
                # print(i)
                if j["text"] == "EMPTY" or j["text"] == "EMPTY FILE":
                    continue
                f.write(json.dumps(j))
                # print(j)
                # input()
                f.write("\n")
