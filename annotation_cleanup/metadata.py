"""Collect metadata and output it to json

Requirements:
   pip install beautifulsoup4

"""

import os
import pathlib
import argparse
import bs4
import json
import datetime
from collections import defaultdict


def extract_url_from_url_file(filepath: pathlib.Path) -> str:
    with filepath.open() as f:
        return f.readline().strip()


def extract_title_from_html_file(filepath: pathlib.Path) -> str:
    with filepath.open() as f:
        return bs4.BeautifulSoup(f.read(), "html.parser").title.string


def extract_timestamp_from_file(filepath: pathlib.Path) -> str:
    return datetime.datetime.fromtimestamp(os.path.getmtime(str(filepath))).isoformat()


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("-d", "--directory", default=".", help="Path prefix of URL files")
    argparser.add_argument("url_files", help="File paths of input URL files (*.url; one path per line)")
    argparser.add_argument("output_file", help="Output file (JSONL)")
    args = argparser.parse_args()

    with open(args.url_files, "r") as f, open(args.output_file, "w") as of:
        for line in f:
            # decompose a url, which is like `./html/<country>/orig/<domain>/<filename>`, into its parts
            _, country, _, domain, *url_parts = pathlib.Path(line.strip()).parts
            url_filename = pathlib.Path(*url_parts)

            # list file paths
            data_dir = pathlib.Path(args.directory)
            url_filepath = data_dir / "html" / country / "orig" / domain / url_filename.with_suffix(".url")
            orig_filepath = data_dir / "html" / country / "orig" / domain / url_filename.with_suffix(".html")
            ja_filepath = data_dir / "html" / country / "ja_translated" / domain / url_filename.with_suffix(".html")
            xml_filepath = data_dir / "xml" / country / "ja_translated" / domain / url_filename.with_suffix(".xml")

            # extract metadata by reading the files
            orig_url = extract_url_from_url_file(url_filepath)

            orig_title = extract_title_from_html_file(orig_filepath)
            ja_title = extract_title_from_html_file(ja_filepath)

            orig_timestamp = extract_timestamp_from_file(orig_filepath)
            ja_timestamp = extract_timestamp_from_file(ja_filepath)
            xml_timestamp = extract_timestamp_from_file(xml_filepath)

            # append the metadata
            item = {
                "text": "",
                "labels": [],
                "meta":{
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
                    "orig_title": orig_title,
                    "ja_title": ja_title,
                "url": orig_url,
                "domain": domain
            }}
            
            # output the metadata as a JSONL file
            json.dump(item, of, ensure_ascii=False)
            of.write("\n")

