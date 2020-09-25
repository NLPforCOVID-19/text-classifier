"""Split data into train and test"""

import logging
import argparse

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s:%(name)s:%(levelname)s: %(message)s')
    
    argparser = argparse.ArgumentParser()
    argparser.add_argument("input_file", help="Text file (one instance per line)")
    argparser.add_argument("train_file", help="Output file for train")
    argparser.add_argument("test_file", help="Output file for test")
    argparser.add_argument("--test-size", type=float, default=0.1, help="Test set size (default: 0.1)")
    args = argparser.parse_args()

    test_iter = int(1.0 / args.test_size)
    with open(args.input_file, "r") as f, open(args.train_file, "w") as train, open(args.test_file, "w") as test:
        for i, line in enumerate(f):
            if i % test_iter == 0:
                test.write(line)
            else:
                train.write(line)

