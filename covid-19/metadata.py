"""Collect metadata and output it to json
Requirements:
   pip install beautifulsoup4
"""

import os
import argparse
import bs4
import json
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
            with open(args.url_files, "r") as f:
                orig_url = f.readline().strip()
            domain = urlparse(orig_url).netloc
            # read <title> from orig html
            with open("{}/{}".format(args.d, orig_file[2:]), "r") as f:
            #with open(f"{args.d}/{orig_file}", "r") as f:
                orig_html = f.read()
            orig_bs = bs4.BeautifulSoup(orig_html, "html.parser")
            orig_title = orig_bs.title.string
            # read <title> from ja html
            with open("{}/{}".format(args.d, ja_file), "r") as f:
                ja_html = f.read()
            ja_bs = bs4.BeautifulSoup(ja_html, "html.parser")
            ja_title = ja_bs.title.string
            # output
            new_elem = {
                "ja_title": ja_title,
                "orig_title": orig_title,
                "ja_file": ja_file,
                "orig_file": orig_file,
                "xml_file": xml_file,
                "orig_url": orig_url,
                "domain": domain
            }
            my_elem = {
                "text": "",
                "labels": [],
                "meta":{
                    "ja_title": ja_title,
                    "orig_title": orig_title,
                    "ja_file": ja_file,
                    "orig_file": orig_file,
                    "xml_file": xml_file[2:],
                    "orig_url": orig_url,
                    "domain": domain
                }
                    }
            # print(my_elem)
            metadata[lang].append(my_elem)

    # output into JSON file
    with open(args.output_file, "w") as of:
        json.dump(dict(metadata), of, ensure_ascii=False)

    with open("data/annotation","w") as f:
        for i in metadata["en"]:
            print(i)
            f.write(json.dumps(i))
            f.write("\n")

