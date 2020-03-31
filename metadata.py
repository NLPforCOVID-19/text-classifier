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
    argparser.add_argument("-d", default=".", help="Path prefix of URL files")
    argparser.add_argument("url_files", help="File paths of input URL files (*.url; one path per line)")
    argparser.add_argument("output_file", help="Output file (JSON)")
    args = argparser.parse_args()

    metadata = defaultdict(list)
    with open(args.url_files, "r") as f:
        for line in f:
            # decompose a url, which is like `./html/<lang>/orig/<domain>/<filename>`, into its parts
            _, lang, _, domain, url_filename = pathlib.Path(line.strip()).parts

            # list file paths
            data_dir = pathlib.Path(args.d)
            basename = pathlib.Path(url_filename).stem
            url_filepath = data_dir / "html" / lang / "orig" / domain / (basename + ".url")
            orig_filepath = data_dir / "html" / lang / "orig" / domain / (basename + ".html")
            ja_filepath = data_dir / "html" / lang / "ja_translated" / domain / (basename + ".html")
            xml_filepath = data_dir / "xml" / lang / "ja_translated" / domain / (basename + ".xml")

            # extract metadata by reading the files
            orig_url = extract_url_from_url_file(url_filepath)

            orig_title = extract_title_from_html_file(orig_filepath)
            ja_title = extract_title_from_html_file(ja_filepath)

            orig_timestamp = extract_timestamp_from_file(orig_filepath)
            ja_timestamp = extract_timestamp_from_file(ja_filepath)
            xml_timestamp = extract_timestamp_from_file(xml_filepath)

            # append the metadata
            metadata[lang].append({
                "ja_title": ja_title,
                "orig_title": orig_title,
                "ja_file": str(ja_filepath.relative_to(data_dir)),
                "orig_file": str(orig_filepath.relative_to(data_dir)),
                "xml_file": str(xml_filepath.relative_to(data_dir)),
                "ja_timestamp": ja_timestamp,
                "orig_timestamp": orig_timestamp,
                "xml_timestamp": xml_timestamp,
                "orig_url": orig_url,
                "domain": domain
            })

    # output the metadata as a JSON file
    with open(args.output_file, "w") as of:
        json.dump(dict(metadata), of, ensure_ascii=False)
