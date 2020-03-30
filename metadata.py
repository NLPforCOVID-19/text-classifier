"""Collect metadata and output it to json
Requirements:
   pip install beautifulsoup4
"""

import os
import argparse
import bs4
import json
import datetime
from urllib.parse import urlparse
from collections import defaultdict

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("-d", default=".", help="Path prefix of URL files")
    argparser.add_argument("url_files", help="File paths of input URL files (*.url; one path per line)")
    argparser.add_argument("output_file", help="Output file (JSON)")
    args = argparser.parse_args()

    metadata = defaultdict(list)
    with open(args.url_files, "r") as url_files:
        for url_file in url_files:
            url_file = url_file.strip()
            lang_pos = url_file.find("html/") + 5      # the position right after "html/"
            lang = url_file[lang_pos:lang_pos + 2]     # extract language info; this method is too ad-hoc...
            orig_file = os.path.splitext(url_file)[0] + ".html"            # .url -> # .html
            ja_file = orig_file.replace("/orig/", "/ja_translated/", 1)    # .../orig/... -> .../ja_translated/...
            xml_file = os.path.splitext(ja_file.replace("html/", "xml/", 1))[0] + ".xml"  # html/.../...html -> xml/.../...xml
            with open(f"{args.d}/{url_file}", "r") as f:
                orig_url = f.readline().strip()
            domain = urlparse(orig_url).netloc
            # read <title> from orig html
            with open(f"{args.d}/{orig_file}", "r") as f:
                orig_html = f.read()
            orig_bs = bs4.BeautifulSoup(orig_html, "html.parser")
            orig_title = orig_bs.title.string
            # read <title> from ja html
            with open(f"{args.d}/{ja_file}", "r") as f:
                ja_html = f.read()
            ja_bs = bs4.BeautifulSoup(ja_html, "html.parser")
            ja_title = ja_bs.title.string
            # file timestamp
            orig_timestamp = datetime.datetime.fromtimestamp(os.stat(f"{args.d}/{orig_file}").st_mtime)
            ja_timestamp = datetime.datetime.fromtimestamp(os.stat(f"{args.d}/{ja_file}").st_mtime)
            xml_timestamp = datetime.datetime.fromtimestamp(os.stat(f"{args.d}/{xml_file}").st_mtime)
            # output
            new_elem = {
                "ja_title": ja_title,
                "orig_title": orig_title,
                "ja_file": ja_file,
                "orig_file": orig_file,
                "xml_file": xml_file,
                "ja_timestamp": ja_timestamp.isoformat(),
                "orig_timestamp": orig_timestamp.isoformat(),
                "xml_timestamp": xml_timestamp.isoformat(),
                "orig_url": orig_url,
                "domain": domain
            }
            #print(new_elem)
            metadata[lang].append(new_elem)

    # output into JSON file
    with open(args.output_file, "w") as of:
        json.dump(dict(metadata), of, ensure_ascii=False)

