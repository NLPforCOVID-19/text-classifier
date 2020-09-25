"""Classify new texts into pre-defined classes"""

import logging
import argparse
import unicodedata
import json
from collections import defaultdict

logger = logging.getLogger(__name__)

target_pos = ["名詞" ,"動詞", "未定義語"]

def read_keywords(keyword_file):
    """Read keyword file"""
    keywords = {}
    for line in keyword_file:
        clss, *words = line.strip().split()
        keywords[clss] = [unicodedata.normalize("NFKC", k) for k in words]
    return keywords

def filter_words(wordlists, poslists, target_pos=[]):
    """Filter out words not used by classifier"""
    return [[word for word, pos in zip(words, poss) if pos in target_pos] for words, poss in zip(wordlists, poslists)]

def classify(text, rawtext, keyword_dict):
    "Classify a text into pre-defined classes"
    classes = {}
    snippets = {}
    for clss, keywords in keyword_dict.items():
        classes[clss] = 0
        snippet_scores = []
        for wordlist, rawsentence in zip(text, rawtext):
            if len(wordlist) <= 3: continue   # ignore sentences with <=3 content words
            # concatenate all tokens and look for keywords
            string = ''.join(wordlist)
            num_keywords = sum([1 for key in keywords if key in string])
            if num_keywords >= 1:
                # found keywords -> compute scores
                classes[clss] = 1
                snippet_scores.append((rawsentence, num_keywords / len(wordlist)))
        # sort snippets by scores
        snippets[clss] = [a[0] for a in sorted(snippet_scores, key=lambda x: x[1], reverse=True)]
    return classes, snippets

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s:%(name)s:%(levelname)s: %(message)s')
    
    argparser = argparse.ArgumentParser()
    argparser.add_argument("text_file", help="Text file (JSONL)")
    argparser.add_argument("keyword_file", help="Keyword file (space-separated text)")
    argparser.add_argument("output_file", help="Output file (JSONL)")
    args = argparser.parse_args()

    # initialize keyword dict
    with open(args.keyword_file, "r") as f:
        keywords = read_keywords(f)

    num_pages = 0
    statistics = defaultdict(int)
    with open(args.text_file, "r") as text_file, open(args.output_file, "w") as of:
        for line in text_file:
            line = line.strip()
            page = json.loads(line)
            wordlists = page['wordlists']
            poslists = page['poslists']
            rawsentences = page['rawsentences']

            # apply classifier
            wordlists = filter_words(wordlists, poslists, target_pos)
            classes, snippets = classify(wordlists, rawsentences, keywords)
            page["classes"] = classes
            page["snippets"] = snippets
            for clss, output in classes.items():
                statistics[clss] += output

            # output the results into JSONL file
            json.dump(page, of, ensure_ascii=False)
            of.write("\n")
            num_pages += 1

    logger.info("Pages: %s", num_pages)
    for clss, num in statistics.items():
        logger.info("%s: %s", clss, num)


