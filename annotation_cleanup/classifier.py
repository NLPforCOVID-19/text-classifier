"""Classify new texts into pre-defined classes"""

import logging
import argparse
from xml.etree import ElementTree
import unicodedata
import json
from collections import defaultdict

from pyknp import Juman

import sys
sys.path.append('..')
from preprocess.shorten_sentence import shorten

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

def classify(text, rawtext, keyword_dict, page):
    "Classify a text into pre-defined classes"
    classes = {}
    snippets = {}
    snippets_en = {}
    for clss, keywords in keyword_dict.items():
        classes[clss] = 0
        snippet_scores = []
        for idx, (wordlist, rawsentence) in enumerate(zip(text, rawtext)):
            if len(wordlist) <= 3: continue   # ignore sentences with <=3 content words
            string = ''.join(wordlist)
            num_keywords = sum([1 for key in keywords if key in string])
            if num_keywords >= 1:
                classes[clss] = 1
                snippet_scores.append((rawsentence, num_keywords / len(wordlist), idx))
        # snippets[clss] = [shorten(a[0], is_title=False) for a in sorted(snippet_scores, key=lambda x: x[1], reverse=True)]
        snippets[clss] = []
        snippets_en[clss] = []
        used_eidxs = {}
        for rawsentence, ratio, idx in sorted(snippet_scores, key=lambda x: x[1], reverse=True):
            snippets[clss].append(shorten(rawsentence, is_title=False))
            eidx = page["idxmap"][idx]
            if eidx in used_eidxs:
                pass
                #sys.stderr.write("sentence already usd: {}d\n".format(page["en_translated"]["rawsentences"][eidx]))
            else:
                snippets_en[clss].append(page["en_translated"]["rawsentences"][eidx])
                used_eidxs[eidx] = True
    return classes, snippets, snippets_en

def extract_meta_add_keyword_classify(keyword_file, meta):
    with open(keyword_file, "r") as f:
        keywords = read_keywords(f)

    wordlists, rawsentences = meta["wordlists"], meta["rawsentences"]
    classes, snippets, snippets_en = classify(wordlists, rawsentences, keywords, meta)
    meta["classes"] = classes
    meta["labels"] = [key for key in classes.keys() if classes[key] > 0]
    meta["snippets"] = snippets
    meta["snippets_en"] = snippets_en

    return meta


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
            # # page = item["meta"]
            page = item # HACK
            wordlists, rawsentences = page["wordlists"], page["rawsentences"]

            sys.stderr.write("{}\n".format(page["orig"]["file"]))

            # apply classifier
            classes, snippets, snippets_en = classify(wordlists, rawsentences, keywords, page)
            page["classes"] = classes
            item["labels"] = [key for key in classes.keys() if classes[key] > 0]
            page["snippets"] = snippets
            page["snippets_en"] = snippets_en
            for clss, output in classes.items():
                statistics[clss] += output

            # output the results into JSONL file
            json.dump(item, of, ensure_ascii=False)
            of.write("\n")

    logger.info("Pages: %s", num_pages)
    logger.info("Ignored: %s", num_ignored)
    for clss, num in statistics.items():
        logger.info("%s: %s", clss, num)


