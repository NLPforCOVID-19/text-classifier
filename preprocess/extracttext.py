"""Extract text data from HTML files in standard format
"""

import logging
import argparse
import unicodedata
import json
from xml.etree import ElementTree
from pyknp import Juman
from collections import defaultdict

def to_text(juman, standard_format):
    "Convert standard format into a simple text (list of word lists)"
    texts = etree.findall(".//Title") + etree.findall(".//Description") + etree.findall(".//S")
    wordlists, poslists, rawsentences = [], [], []
    for text in texts:
        annotation = text.find("Annotation")
        if annotation is None: continue
        mlist = juman.result(annotation.text)
        wordlists.append([unicodedata.normalize("NFKC", m.midasi) for m in mlist.mrph_list()])
        poslists.append([unicodedata.normalize("NFKC", m.hinsi) for m in mlist.mrph_list()])
        rawsentence = text.find("RawString")
        rawsentences.append(rawsentence.text)
    return wordlists, poslists, rawsentences

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s:%(name)s:%(levelname)s: %(message)s')
    
    argparser = argparse.ArgumentParser()
    argparser.add_argument("-d", "--directory", default=".", help="Path prefix of XML files")
    argparser.add_argument("-t", "--target", default="ja_translated", help="JSON attribute of target language")
    argparser.add_argument("metadata_file", help="Metadata file (JSONL)")
    argparser.add_argument("output_file", help="Output file (JSONL)")
    args = argparser.parse_args()

    # initialize juman (used only for parsing standard format)
    juman = Juman("cat")  # use dummy command to avoid installing jumanpp...

    num_pages = 0
    num_ignored = 0
    statistics = defaultdict(int)
    with open(args.metadata_file, "r") as metadata_file, open(args.output_file, "w") as of:
        for line in metadata_file:
            line = line.strip()
            page = json.loads(line)
            xml_file = page[args.target]["xml_file"]
            #print(xml_file)
            try:
                etree = ElementTree.parse(f"{args.directory}/{xml_file}")
                num_pages += 1
            except ElementTree.ParseError:
                logger.error("XML file broken: %s", xml_file)
                num_ignored += 1
                continue
            wordlists, poslists, rawsentences = to_text(juman, etree)

            # output the results into JSONL file
            page["rawsentences"] = rawsentences
            page["wordlists"] = wordlists
            page["poslists"] = poslists
            json.dump(page, of, ensure_ascii=False)
            of.write("\n")
