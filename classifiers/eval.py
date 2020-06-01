"""Compute accuracy of text classification results"""

import sys
import argparse
import json
import numpy
from sklearn.metrics import precision_recall_fscore_support, average_precision_score

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--gold", required=True, help="Gold file (JSONL)")
    argparser.add_argument("--pred", required=True, help="Prediction file (JSONL)")
    argparser.add_argument("--classes", required=True, help="Classes file (text)")
    argparser.add_argument("-t", "--threshold", default=0.5, help="Probability threshold for precision/recall/F-score")
    args = argparser.parse_args()

    # read classes file
    with open(args.classes, "r") as f:
        classes = [line.strip() for line in f]

    # read gold/pred files
    gold = []
    with open(args.gold, "r") as f:
        for line in f:
            tags = json.loads(line)["classes"]
            gold.append([tags[cl] for cl in classes])
    pred = []
    with open(args.pred, "r") as f:
        for line in f:
            tags = json.loads(line)["classes"]
            pred.append([tags[cl] for cl in classes])
    if len(gold) != len(pred):
        sys.stderr.write("Numbers of lines different: {}, {}\n".format(len(gold), len(pred)))
        sys.exit(1)
    gold_matrix = numpy.array(gold)
    pred_matrix = numpy.array(pred)

    # show accuracy
    sys.stdout.write("{} instances\n".format(len(gold)))
    sys.stdout.write("{} classes: {}\n".format(len(classes), ' '.join(classes)))
    precision, recall, fscore, _ = precision_recall_fscore_support(gold_matrix >= 1.0, pred_matrix > args.threshold, zero_division=0)
    average_precision = average_precision_score(gold_matrix >= 1.0, pred_matrix, average=None)
    sys.stdout.write("Precision/Recall/F-score/Average precision\n")
    for cl, p, r, f, a in zip(classes, precision, recall, fscore, average_precision):
        sys.stdout.write(f"{cl}: {p:.3f}/{r:.3f}/{f:.3f}/{a:.3f}\n")
    
