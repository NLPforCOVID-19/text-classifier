"""Extract text data from HTML files in standard format
"""

import sys
import logging
import argparse
import unicodedata
import json
from xml.etree import ElementTree
from pyknp import Juman
from collections import defaultdict
import pathlib
import bs4
def to_text(juman, etree, htmlsentences, htmlfile):
    '''
    Convert standard format into a simple text (list of word lists)

    translation HTML was converted to SF
    during this step, extra sentence splitting is performed, making sentence alignment hard
    to address this problem, we look at SF's Offset and Length attributes, read the corresponding text chunk in the HTML

    idxmap: idx in SF -> idx in HTML (based on the ja file but the en file has the same number of sentences)
    '''
    m_len = 100 
    sfoffset = 0
    htmloffset = 0
    htmlidx = 0
    idxmap = []

    texts = etree.findall(".//Title") + etree.findall(".//Description") + etree.findall(".//S")
    wordlists, poslists, rawsentences = [], [], []

    for text in texts:
        rawsentence = text.find("RawString")
        cur_htmlidx = htmlidx
        offset, length = int(text.attrib["Offset"]), int(text.attrib["Length"])
        if offset < 0:
            continue
            raise Exception("no valid offset")
        htmlfile.seek(offset)
        dat = htmlfile.read(length)
        htmlstr = dat.decode("utf-8")
        sfoffset += len(htmlstr)

        # sys.stderr.write("{}\t{}\t{}\t{}\n".format(sfoffset, htmloffset, rawsentence.text, htmlsentences[htmlidx]))
        current_sentence = htmlsentences[htmlidx]
        if (current_sentence == None): continue
        if sfoffset >= htmloffset + len(htmlsentences[htmlidx]):
            htmloffset += len(htmlsentences[htmlidx])
            htmlidx += 1            
        
        annotation = text.find("Annotation")
        if annotation is None: continue
        try:
            mlist = juman.result(annotation.text)
        except:
            sys.stderr.write(f"juman failed{annotation}\n")
            continue
        wordlists.append([unicodedata.normalize("NFKC", m.midasi) for m in mlist.mrph_list()][:m_len])
        poslists.append([unicodedata.normalize("NFKC", m.hinsi) for m in mlist.mrph_list()][:m_len])

        idxmap.append(cur_htmlidx)
        rawsentences.append(rawsentence.text[:m_len])
    return wordlists, poslists, rawsentences, idxmap


def extract_sentences_from_html_file(filepath: pathlib.Path) -> str:
    with filepath.open() as f:
        htmltree = bs4.BeautifulSoup(f.read(), "html.parser")
        sentences = []
        for ent in htmltree.find_all('title') + htmltree.find_all('p'):
            sentences.append(ent.string)
        return sentences

def extract_meta_add_text(data_dir, meta, juman):
    data_dir_str = data_dir
    data_dir = pathlib.Path(data_dir)

    en_filepath = data_dir / meta["en_translated"]["file"]
    en_sentences = extract_sentences_from_html_file(en_filepath)
    meta["en_translated"]["rawsentences"] = en_sentences

    ja_filepath = data_dir / meta["ja_translated"]["file"]
    ja_sentences = extract_sentences_from_html_file(ja_filepath)
    ja_htmlfile = ja_filepath.open('rb')

    xml_file = meta["ja_translated"]["xml_file"]

    xml_file_path = "{}/{}".format(data_dir_str, xml_file)
    etree = ElementTree.parse(xml_file_path)
    wordlists, poslists, rawsentences, idxmap = to_text(juman, etree, ja_sentences, ja_htmlfile)

    meta["rawsentences"] = rawsentences
    meta["wordlists"] = wordlists
    meta["poslists"] = poslists
    meta["idxmap"] = idxmap

    return meta

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s:%(name)s:%(levelname)s: %(message)s')
    
    argparser = argparse.ArgumentParser()
    argparser.add_argument("-d", "--directory", default=".", help="Path prefix of XML files")
    argparser.add_argument("-t", "--target", default="ja_translated", help="JSON attribute of target language")
    argparser.add_argument("metadata_file", help="Metadata file (JSONL)")
    argparser.add_argument("output_file", help="Output file (JSONL)")
    args = argparser.parse_args()

    data_dir = pathlib.Path(args.directory)
    # initialize juman (used only for parsing standard format)
    juman = Juman("cat")  # use dummy command to avoid installing jumanpp...

    num_pages = 0
    num_ignored = 0
    statistics = defaultdict(int)
    with open(args.metadata_file, "r") as metadata_file, open(args.output_file, "w") as of:
        for line in metadata_file:
            line = line.strip()
            page = json.loads(line)


            en_filepath = data_dir / page["en_translated"]["file"]
            en_sentences = extract_sentences_from_html_file(en_filepath)
            page["en_translated"]["rawsentences"] = en_sentences

            ja_htmlpath = data_dir / page["ja_translated"]["file"]
            ja_sentences_html = extract_sentences_from_html_file(ja_htmlpath)
            ja_htmlfile = ja_htmlpath.open('rb')

            xml_file = page[args.target]["xml_file"]
            try:
                etree = ElementTree.parse(f"{args.directory}/{xml_file}")
                num_pages += 1
            except ElementTree.ParseError:
                logging.error("XML file broken: %s", xml_file)
                num_ignored += 1
                continue
            try:
                wordlists, poslists, rawsentences, idxmap = to_text(juman, etree, ja_sentences_html, ja_htmlfile)
            except Exception as e:
                sys.stderr.write("{}\t{}\n".format(xml_file, e))
                continue

            # output the results into JSONL file
            page["rawsentences"] = rawsentences
            page["wordlists"] = wordlists
            page["poslists"] = poslists
            page["idxmap"] = idxmap
            json.dump(page, of, ensure_ascii=False)
            of.write("\n")
