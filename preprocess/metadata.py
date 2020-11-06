"""Collect metadata and output it to json

Requirements:
   pip install beautifulsoup4

"""

import sys
import re
import os
import pathlib
import argparse
import bs4
import json
import datetime
from collections import defaultdict

from preprocess.shorten_sentence import shorten


def extract_url_from_url_file(filepath: pathlib.Path) -> str:
    with filepath.open() as f:
        return f.readline().strip()


def extract_title_from_html_file(filepath: pathlib.Path, do_shorten="ja") -> str:
    with filepath.open() as f:
        title = bs4.BeautifulSoup(f.read(), "html.parser").title
        if title is None:
            return ""
        else:
            if do_shorten == "ja":
                return shorten(title.string)
            else:
                return title.string


def extract_timestamp_from_file(filepath: pathlib.Path) -> str:
    return datetime.datetime.fromtimestamp(os.path.getmtime(str(filepath))).isoformat()


def extract_meta(data_dir, sourceinfo_file, url_path):
    sourceinfo = {}
    with open(sourceinfo_file, "r") as f:
        for line in f:
            line = line.rstrip()
            domain, sourcelabel, sourcelabel_en = line.split("\t", maxsplit=2)
            sourceinfo[domain] = (sourcelabel, sourcelabel_en)

    _, country, _, domain, *url_parts = pathlib.Path(url_path.strip()).parts
    url_filename = pathlib.Path(*url_parts)

    # list file paths
    data_dir = pathlib.Path(data_dir)
    url_filepath = data_dir / "html" / country / "orig" / domain / url_filename.with_suffix(".url")
    orig_filepath = data_dir / "html" / country / "orig" / domain / url_filename.with_suffix(".html")
    ja_filepath = data_dir / "html" / country / "ja_translated" / domain / url_filename.with_suffix(".html")
    xml_filepath = data_dir / "xml" / country / "ja_translated" / domain / url_filename.with_suffix(".xml")
    en_filepath = data_dir / "html" / country / "en_translated" / domain / url_filename.with_suffix(".html")

    orig_url = extract_url_from_url_file(url_filepath)
    orig_title = extract_title_from_html_file(orig_filepath, do_shorten=None)
    ja_title = extract_title_from_html_file(ja_filepath, do_shorten="ja")
    en_title = extract_title_from_html_file(en_filepath, do_shorten=None)
    orig_timestamp = extract_timestamp_from_file(orig_filepath)
    ja_timestamp = extract_timestamp_from_file(ja_filepath)
    en_timestamp = extract_timestamp_from_file(en_filepath)
    xml_timestamp = extract_timestamp_from_file(xml_filepath)

    # ad-hoc bug fix
    if re.match(r'bundle\...', domain) is not None:
        domain = url_parts[0]
    elif domain == "www.cdc.gov.kr":
        domain = "www.cdc.go.kr"
    
    # append the metadata
    meta = {
        "country": country,
        "orig": {
            "file": str(orig_filepath.relative_to(data_dir)),
            "title": orig_title,
            "timestamp": orig_timestamp
        },
        "ja_translated": {
            "file": str(ja_filepath.relative_to(data_dir)),
            "title": ja_title,
            "timestamp": ja_timestamp,
            "xml_file": str(xml_filepath.relative_to(data_dir)),
            "xml_timestamp": xml_timestamp
        },
        "en_translated": {
            "file": str(en_filepath.relative_to(data_dir)),
            "title": en_title,
            "timestamp": en_timestamp,
        },
        "url": orig_url,
        "domain": domain
    }
    for _domain, (sourcelabel, sourcelabel_en) in sourceinfo.items():
        if domain.endswith(_domain):
            meta["domain_label"] = sourcelabel
            meta["domain_label_en"] = sourcelabel_en
            break
    #json.dump(meta, of, ensure_ascii=False)
    return meta


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("-d", "--directory", default=".", help="Path prefix of XML files")
    argparser.add_argument("-s", "--sourceinfo", default=None, help="domain labels")
    argparser.add_argument("url_path", help="File paths of input URL files (*.url; one path per line)")
    argparser.add_argument("output_file", help="Output file (JSONL)")
    args = argparser.parse_args()

    sourceinfo = {}
    if args.sourceinfo is not None:
        with open(args.sourceinfo, "r") as f:
            for line in f:
                line = line.rstrip()
                domain, sourcelabel, sourcelabel_en = line.split("\t", maxsplit=2)
                sourceinfo[domain] = (sourcelabel, sourcelabel_en)
    
    with open(args.output_file, "w") as of:
        # './html/fr/en_translated/www.lemonde.fr/signataires/francois-beguin/?page=3/2020/11/03-08-45/?page=3.html'
        # decompose a url, which is like `./html/<country>/ja_translated/<domain>/<filename>`, into its parts
        _, country, _, domain, *url_parts = pathlib.Path(url_path.strip()).parts
        url_filename = pathlib.Path(*url_parts)

        # list file paths
        data_dir = pathlib.Path(args.directory)
        url_filepath = data_dir / "html" / country / "orig" / domain / url_filename.with_suffix(".url")
        orig_filepath = data_dir / "html" / country / "orig" / domain / url_filename.with_suffix(".html")
        ja_filepath = data_dir / "html" / country / "ja_translated" / domain / url_filename.with_suffix(".html")
        xml_filepath = data_dir / "xml" / country / "ja_translated" / domain / url_filename.with_suffix(".xml")
        en_filepath = data_dir / "html" / country / "en_translated" / domain / url_filename.with_suffix(".html")

        try:
            # extract metadata by reading the files
            orig_url = extract_url_from_url_file(url_filepath)

            orig_title = extract_title_from_html_file(orig_filepath, do_shorten=None)
            ja_title = extract_title_from_html_file(ja_filepath, do_shorten="ja")
            en_title = extract_title_from_html_file(en_filepath, do_shorten=None)

            orig_timestamp = extract_timestamp_from_file(orig_filepath)
            ja_timestamp = extract_timestamp_from_file(ja_filepath)
            en_timestamp = extract_timestamp_from_file(en_filepath)
            xml_timestamp = extract_timestamp_from_file(xml_filepath)
        except Exception as e:
            sys.stderr.write(f"file not found error...skip: {line}\t{e}\n")


        # ad-hoc bug fix
        if re.match(r'bundle\...', domain) is not None:
            domain = url_parts[0]
        elif domain == "www.cdc.gov.kr":
            domain = "www.cdc.go.kr"
        
        # append the metadata
        meta = {
            "country": country,
            "orig": {
                "file": str(orig_filepath.relative_to(data_dir)),
                "title": orig_title,
                "timestamp": orig_timestamp
            },
            "ja_translated": {
                "file": str(ja_filepath.relative_to(data_dir)),
                "title": ja_title,
                "timestamp": ja_timestamp,
                "xml_file": str(xml_filepath.relative_to(data_dir)),
                "xml_timestamp": xml_timestamp
            },
            "en_translated": {
                "file": str(en_filepath.relative_to(data_dir)),
                "title": en_title,
                "timestamp": en_timestamp,
            },
            "url": orig_url,
            "domain": domain
        }
        for _domain, (sourcelabel, sourcelabel_en) in sourceinfo.items():
            if domain.endswith(_domain):
                meta["domain_label"] = sourcelabel
                meta["domain_label_en"] = sourcelabel_en
                break
        else:
            sys.stderr.write(f"unrecognized domain name {domain}\n")
        
        # output the metadata as a JSONL file
        json.dump(meta, of, ensure_ascii=False)
        of.write("\n")
