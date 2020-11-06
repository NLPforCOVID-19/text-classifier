"""Collect metadata and output it to json
"""

import sys
import os
import pathlib
import argparse
import json
import datetime
from collections import defaultdict

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("-s", "--sourceinfo", default=None, help="domain labels")
    argparser.add_argument("input_file", help="Input file (JSONL)")
    argparser.add_argument("output_file", help="Output file (JSONL)")
    args = argparser.parse_args()

    sourceinfo = {}
    if args.sourceinfo is not None:
        with open(args.sourceinfo, "r") as f:
            for line in f:
                line = line.rstrip()
                domain, sourcelabel, sourcelabel_en = line.split("\t", maxsplit=2)
                sourceinfo[domain] = (sourcelabel, sourcelabel_en)

    with open(args.input_file, "r") as in_f, open(args.output_file, "w") as of:
        for line in in_f:
            line = line.strip()
            page = json.loads(line)
            domain = page["domain"]

            for _domain, (sourcelabel, sourcelabel_en) in sourceinfo.items():
                if domain.endswith(_domain):
                    page["domain_label"] = sourcelabel
                    page["domain_label_en"] = sourcelabel_en
                    break
            else:
                sys.stderr.write(f"unrecognized domain name {domain}\n")
            
            # output the metadata as a JSONL file
            json.dump(page, of, ensure_ascii=False)
            of.write("\n")
