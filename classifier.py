"""Classify new texts into pre-defined classes"""

import argparse
from xml.etree import ElementTree
import json
from collections import defaultdict
from pyknp import Juman

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
    texts = etree.findall(".//Title") + etree.findall(".//Description") + etree.findall(".//S")
    wordlists, rawsentences = [], []
    for text in texts:
        annotation = text.find("Annotation")
        if annotation is None: continue
        mlist = juman.result(annotation.text)
        wordlists.append([m.midasi for m in mlist.mrph_list() if m.hinsi in target_pos])
        rawsentence = text.find("RawString")
        rawsentences.append(rawsentence.text)
    return wordlists, rawsentences

def classify(text, keyword_dict):
    "Classify a text into pre-defined classes"
    classes = {}
    snippets = {}
    for clss, keywords in keyword_dict.items():
        classes[clss] = 0
        snippet_scores = []
        for i, sentence in enumerate(text):
            if len(sentence) <= 3: continue   # ignore sentences with <=3 content words
            num_keywords = len(keywords & set(sentence))
            if num_keywords >= 1:
                classes[clss] = 1
                snippet_scores.append((i, num_keywords / len(sentence)))
        snippets[clss] = [a[0] for a in sorted(snippet_scores, key=lambda x: x[1], reverse=True)]
    return classes, snippets

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("-d", "--directory", default=".", help="Path prefix of XML files")
    argparser.add_argument("metadata_file", help="Metadata file (JSON)")
    argparser.add_argument("output_file", help="Output file (JSON)")
    args = argparser.parse_args()

    # initialize keyword dict and juman
    keywords = {key: set(value.split()) for key, value in keywords.items()}
    juman = Juman()

    with open(args.metadata_file, "r") as metadata_file, open(args.output_file, "w") as of:
        for line in metadata_file:
            line = line.strip()
            page = json.loads(line)
            xml_file = page["xml_file"]
            etree = ElementTree.parse(f"{args.directory}/{xml_file}")
            wordlists, rawsentences = to_text(juman, etree, target_pos)

            # apply classifier
            classes, snippet_ids = classify(wordlists, keywords)
            page["classes"] = classes

            # snippets
            snippets = { clss: [rawsentences[i] for i in ids] for clss, ids in snippet_ids.items() }
            page["snippets"] = snippets

            # output the results into JSONL file
            json.dump(page, of, ensure_ascii=False)
            of.write("\n")

