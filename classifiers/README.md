# Text classifiers

Train/run a classifier using `make`.

- Classifiers:
  - `kwclassifier.py`: Keyword-match classifier.  Uses `keywords.txt` as a keyword list.
  - `bertsimple.py`: Classifier using BERT features.
- Make goals:
  - `train-...`: Train a classifier using `../data/annotations-train.jsonl`
  - `test-...`: Run a classifier on `../data/annotations-test.jsonl` and evaluate accuracy.
  - `run-...`: Run a classifier on entire text data (`../data/textdata.jsonl`).

