"""Extract text data from HTML files in standard format
"""

import logging
import argparse
import unicodedata
import json
from xml.etree import ElementTree
from pyknp import Juman
from collections import defaultdict
import multiprocessing
import functools

logger = logging.getLogger(__name__)

def to_text(juman, etree):
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

def extract_text(line, target, directory):
    line = line.strip()
    page = json.loads(line)
    if target not in page or "xml_file" not in page[target]: return None
    xml_file = page[target]["xml_file"]
    #print(xml_file)
    try:
        etree = ElementTree.parse(f"{directory}/{xml_file}")
    except ElementTree.ParseError:
        logger.error("XML file broken: %s", xml_file)
        return None
    wordlists, poslists, rawsentences = to_text(juman, etree)

    # output the results into JSONL file
    page["rawsentences"] = rawsentences
    page["wordlists"] = wordlists
    page["poslists"] = poslists
    return page

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s:%(name)s:%(levelname)s: %(message)s')
    
    argparser = argparse.ArgumentParser()
    argparser.add_argument("-d", "--directory", default=".", help="Path prefix of XML files")
    argparser.add_argument("-t", "--target", default="ja_translated", help="JSON attribute of target language")
    argparser.add_argument("metadata_file", help="Metadata file (JSONL)")
    argparser.add_argument("output_file", help="Output file (JSONL)")
    argparser.add_argument("-j", "--num-processes", default=1, type=int, help="Number of processes to run")
    args = argparser.parse_args()

    # initialize juman (used only for parsing standard format)
    juman = Juman("cat")  # use dummy command to avoid installing jumanpp...

    with open(args.metadata_file, "r") as metadata_file, open(args.output_file, "w") as of:
        if args.num_processes == 1:
            for line in metadata_file:
                page = extract_text(line, args.target, args.directory)
                if page is not None:
                    json.dump(page, of, ensure_ascii=False)
                    of.write("\n")
        else:
            # run multiprocessing
            extract_text_func = functools.partial(extract_text, target=args.target, directory=args.directory)
            with multiprocessing.Pool(args.num_processes) as pool:
                for page in pool.imap(extract_text_func, metadata_file, chunksize=10):
                    if page is not None:
                        json.dump(page, of, ensure_ascii=False)
                        of.write("\n")

                    
