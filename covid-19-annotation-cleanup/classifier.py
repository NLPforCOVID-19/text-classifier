"""Classify new texts into pre-defined classes"""

import logging
import argparse
from xml.etree import ElementTree
import unicodedata
import json
from collections import defaultdict
from pyknp import Juman

logger = logging.getLogger(__name__)

target_pos = ["名詞" ,"動詞", "未定義語"]

def read_keywords(keyword_file):
    keywords = {}
    for line in keyword_file:
        clss, *words = line.strip().split()
        keywords[clss] = [unicodedata.normalize("NFKC", k) for k in words]
    return keywords

def inquire_exclude(sent):

    keywords = ['すべての結果を表示',
               'その他']
    for keyword in keywords:
        if keyword in sent:
            return True
    startingKeySymbols = ["＃", "＊", "©", "￥＋", "［","＊", "http"]
    for keyword in startingKeySymbols:
        if keyword in sent:
            return True
    return False

def to_text(juman, standard_format, target_pos=[]):
    "Convert standard format into a simple text (list of word lists)"
    target_pos = set(target_pos)
    texts = etree.findall(".//Title") + etree.findall(".//Description") + etree.findall(".//S")
    wordlists, rawsentences = [], []
    for text in texts:
        # print(text.attrib)
        if text.tag == "S" and  text.attrib["BlockType"] == "unknown_block": continue
        annotation = text.find("Annotation")
        if annotation is None: continue
        mlist = juman.result(annotation.text)
        wordlists.append([unicodedata.normalize("NFKC", m.midasi) for m in mlist.mrph_list() if m.hinsi in target_pos])
        rawsentence = text.find("RawString")
        rawsentences.append(rawsentence.text)
    return wordlists, rawsentences

def classify(text, rawtext, keyword_dict):
    "Classify a text into pre-defined classes"
    classes = {}
    snippets = {}
    for clss, keywords in keyword_dict.items():
        classes[clss] = 0
        snippet_scores = []
        for wordlist, rawsentence in zip(text, rawtext):
            if len(wordlist) <= 3: continue   # ignore sentences with <=3 content words
            string = ''.join(wordlist)
            num_keywords = sum([1 for key in keywords if key in string])
            if num_keywords >= 1:
                classes[clss] = 1
                snippet_scores.append((rawsentence, num_keywords / len(wordlist)))
        snippets[clss] = [a[0] for a in sorted(snippet_scores, key=lambda x: x[1], reverse=True)]
    return classes, snippets

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s:%(name)s:%(levelname)s: %(message)s')
    
    argparser = argparse.ArgumentParser()
    argparser.add_argument("-d", "--directory", default=".", help="Path prefix of XML files")
    argparser.add_argument("-t", "--target", default="ja_translated", help="JSON attribute of target language")
    argparser.add_argument("metadata_file", help="Metadata file (JSON)")
    argparser.add_argument("keyword_file", help="Keyword file (space-separated text)")
    argparser.add_argument("output_file", help="Output file (JSON)")
    args = argparser.parse_args()

    # initialize keyword dict and juman
    with open(args.keyword_file, "r") as f:
        keywords = read_keywords(f)
    juman = Juman()

    num_pages = 0
    num_ignored = 0
    statistics = defaultdict(int)
    with open(args.metadata_file, "r") as metadata_file, open(args.output_file, "w") as of:
        for line in metadata_file:
            line = line.strip()
            item = json.loads(line)
            page = item["meta"]
            xml_file = page[args.target]["xml_file"]
            #print(xml_file)
            try:
                etree = ElementTree.parse(f"{args.directory}/{xml_file}")
                num_pages += 1
            except ElementTree.ParseError:
                logger.error("XML file broken: %s", xml_file)
                num_ignored += 1
                del item
                continue
            wordlists, rawsentences = to_text(juman, etree, target_pos)
            rawsentences = list(filter(lambda x: not inquire_exclude(x), rawsentences))
            cleansentences = []
            slicecontainer = []
            for slice in rawsentences:
                slicecontainer.append(slice)
                if slice.endswith("。"):
                    cleansentences.append("".join(slicecontainer))
                    slicecontainer = []

            item["text"] = "\n".join(cleansentences)

            # apply classifier
            classes, snippets = classify(wordlists, rawsentences, keywords)
            page["classes"] = classes
            item["labels"] = [key for key in classes.keys() if classes[key] > 0]
            page["snippets"] = snippets
            for clss, output in classes.items():
                statistics[clss] += output

            # output the results into JSONL file
            json.dump(item, of, ensure_ascii=False)
            of.write("\n")

    logger.info("Pages: %s", num_pages)
    logger.info("Ignored: %s", num_ignored)
    for clss, num in statistics.items():
        logger.info("%s: %s", clss, num)


