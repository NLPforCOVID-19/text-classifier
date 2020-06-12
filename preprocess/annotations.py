"""Add labels to texts"""

import logging
import argparse
import json

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s:%(name)s:%(levelname)s: %(message)s')
    
    argparser = argparse.ArgumentParser()
    argparser.add_argument("text_file", help="Text file (JSONL)")
    argparser.add_argument("label_file", help="Label file (JSONL)")
    argparser.add_argument("output_file", help="Output file (JSONL)")
    args = argparser.parse_args()

    # read label file
    labels = {}
    with open(args.label_file, "r") as lf:
        for line in lf:
            js = json.loads(line)
            tags = js['tags']
            if 'topics' in tags:
                topics = tags['topics']
                del tags['topics']
                tags.update(topics)
            labels[js['url']] = tags
    print("Number of labeled data: {}".format(len(labels)))

    # read text file and output it if labels exist
    num_output = 0
    with open(args.text_file, "r") as f, open(args.output_file, "w") as of:
        for line in f:
            js = json.loads(line)
            url = js['url']
            if url in labels:
                js['classes'] = labels[url]
                of.write(json.dumps(js, ensure_ascii=False))
                of.write("\n")
                num_output += 1
    print("Number of output data: {}".format(num_output))
